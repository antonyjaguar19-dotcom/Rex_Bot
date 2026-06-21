"""
Claw Bot — ACE-Step 1.5 Song Adapter (via ComfyUI API)

Music-video audio backend. Generates a full song (vocals + instrumental) from a
style-tag string + lyrics using the ACE-Step 1.5 turbo model in ComfyUI.

Mirrors the comfyui_wan22_14B.py submit/poll/download pattern. Workflow is the
API-format export `audio_ace_step_1_5_split.json`. Injection targets node
class_types (ids ignored), so a re-export won't silently break:
  - TextEncodeAceStepAudio1.5  → tags, lyrics, bpm, duration, keyscale, language, seed
  - EmptyAceStep1.5LatentAudio → seconds
  - KSampler                   → seed (steps/cfg keep workflow/config values)
  - SaveAudioMP3               → produced output is fetched from /history
"""

import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

_AGENT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules.audio_backend import AudioBackend

log = logging.getLogger("claw_bot.audio_backend.ace_step")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "songs"


class Backend(AudioBackend):
    """ACE-Step 1.5 song generator via ComfyUI API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.steps = int(config.get("steps", 8))
        self.cfg = float(config.get("cfg", 1))
        self.default_duration = float(config.get("default_duration", 120))

        workflow_file = config.get("workflow_file", "audio_ace_step_1_5_split.json")
        self._workflow_path = PROJECT_ROOT / "05_Config" / "workflows" / workflow_file
        if not self._workflow_path.exists():
            raise FileNotFoundError(
                f"ACE-Step API workflow not found: {self._workflow_path}. "
                f"Export it from ComfyUI GUI (Workflow → Export (API Format))."
            )
        self._workflow_template = json.loads(self._workflow_path.read_text(encoding="utf-8"))
        self._last_seed: int = -1
        log.info(f"ACE-Step adapter initialized — workflow: {self._workflow_path.name}")

    # --------------------------------------------------------------
    # AudioBackend contract
    # --------------------------------------------------------------

    def health_check(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.server_url}/system_stats", timeout=5)
            if r.status_code == 200:
                return True, f"ComfyUI alive at {self.server_url}"
            return False, f"ComfyUI returned HTTP {r.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to ComfyUI at {self.server_url}. Is it running?"
        except Exception as e:
            return False, f"Health check failed: {e}"

    def generate(
        self,
        tags: str,
        lyrics: str,
        duration_sec: float = 120.0,
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        language: str = "en",
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
    ) -> Path:
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        self._last_seed = int(seed)
        duration = float(duration_sec) if duration_sec else self.default_duration

        log.info(
            f"Generating song — seed={seed}, dur={duration:.0f}s, bpm={bpm}, "
            f"key={keyscale}, tags='{(tags or '')[:60]}'"
        )

        workflow = self._build_workflow(
            tags=tags or "",
            lyrics=lyrics or "",
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            language=language or "en",
            seed=seed,
        )

        client_id = str(uuid.uuid4())
        prompt_id = self._submit_prompt(workflow, client_id)
        log.info(f"Submitted prompt_id={prompt_id}")

        # ACE-Step turbo (8 steps) is fast, but cold model load off disk plus a
        # 120s render needs headroom.
        history = self._wait_for_completion(prompt_id, timeout=1200)
        audio_path = self._download_output(history, prompt_id, output_filename)
        log.info(f"Song saved: {audio_path}")
        return audio_path

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    def _build_workflow(self, tags: str, lyrics: str, duration: float,
                        bpm: Optional[int], keyscale: Optional[str],
                        language: str, seed: int) -> dict:
        wf = json.loads(json.dumps(self._workflow_template))

        injected = {"text_encode": False, "latent": False, "seed": False}

        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type", "")
            inputs = node.setdefault("inputs", {})

            if ctype == "TextEncodeAceStepAudio1.5":
                inputs["tags"] = tags
                inputs["lyrics"] = lyrics
                inputs["duration"] = float(duration)
                inputs["seed"] = int(seed)
                inputs["language"] = language
                if bpm is not None:
                    inputs["bpm"] = int(bpm)
                if keyscale:
                    inputs["keyscale"] = keyscale
                injected["text_encode"] = True

            elif ctype == "EmptyAceStep1.5LatentAudio":
                inputs["seconds"] = float(duration)
                injected["latent"] = True

            elif ctype == "KSampler":
                inputs["seed"] = int(seed)
                inputs["steps"] = int(self.steps)
                inputs["cfg"] = float(self.cfg)
                injected["seed"] = True

        log.info(
            f"ACE workflow injection: text_encode={injected['text_encode']} "
            f"latent={injected['latent']} seed={injected['seed']}"
        )
        critical = [k for k, v in injected.items() if not v]
        if critical:
            raise RuntimeError(
                f"ACE-Step workflow injection failed for: {critical}. "
                f"Workflow file ({self._workflow_path.name}) likely changed shape. "
                f"Re-export from ComfyUI GUI (Workflow → Export (API Format))."
            )
        return wf

    def _submit_prompt(self, workflow: dict, client_id: str) -> str:
        url = f"{self.server_url}/prompt"
        payload = {"prompt": workflow, "client_id": client_id}
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected prompt (HTTP {r.status_code}): {r.text[:400]}")
        data = r.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"ComfyUI response missing prompt_id: {data}")
        return data["prompt_id"]

    def _wait_for_completion(self, prompt_id: str, timeout: int = 1200) -> dict:
        history_url = f"{self.server_url}/history/{prompt_id}"
        queue_url = f"{self.server_url}/queue"
        start = time.time()
        last_heartbeat = start
        consecutive_poll_failures = 0

        while time.time() - start < timeout:
            try:
                r = requests.get(history_url, timeout=60)
                consecutive_poll_failures = 0
                if r.status_code == 200:
                    data = r.json()
                    if prompt_id in data:
                        entry = data[prompt_id]
                        status = entry.get("status", {})
                        if status.get("completed") is True:
                            log.info(f"Prompt {prompt_id[:8]} completed in {int(time.time()-start)}s")
                            return entry
                        if status.get("status_str") == "error":
                            raise RuntimeError(
                                f"ComfyUI reported error: {status.get('messages', [])}"
                            )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                consecutive_poll_failures += 1
                log.warning(
                    f"Poll {consecutive_poll_failures}/5 failed for {prompt_id[:8]}: "
                    f"{type(e).__name__}"
                )
                if consecutive_poll_failures >= 5:
                    raise RuntimeError(
                        f"Lost connection to ComfyUI for {prompt_id[:8]} after 5 timeouts."
                    ) from e
                time.sleep(10)
                continue

            now = time.time()
            if now - last_heartbeat >= 30:
                last_heartbeat = now
                try:
                    q = requests.get(queue_url, timeout=10).json()
                    log.info(
                        f"Waiting for {prompt_id[:8]}: {int(now-start)}s elapsed, "
                        f"running={len(q.get('queue_running', []))}, "
                        f"pending={len(q.get('queue_pending', []))}"
                    )
                except Exception:
                    log.info(f"Waiting for {prompt_id[:8]}: {int(now-start)}s elapsed")
            time.sleep(2)

        raise TimeoutError(f"Timed out after {timeout}s waiting for prompt {prompt_id}")

    def _download_output(self, history_entry: dict, prompt_id: str,
                         output_filename: Optional[str]) -> Path:
        outputs = history_entry.get("outputs", {})
        if not outputs:
            raise RuntimeError("ComfyUI finished but produced no outputs.")

        audio_info = None
        for node_id, output in outputs.items():
            items = output.get("audio")
            if items:
                audio_info = items[0]
                break
        if audio_info is None:
            raise RuntimeError("No audio in outputs.")

        filename = audio_info["filename"]
        subfolder = audio_info.get("subfolder", "")
        ftype = audio_info.get("type", "output")

        view_url = (
            f"{self.server_url}/view?"
            f"filename={filename}&subfolder={subfolder}&type={ftype}"
        )
        r = requests.get(view_url, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Could not download audio: HTTP {r.status_code}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(filename).suffix or ".mp3"
        target = OUTPUT_DIR / (output_filename or f"song_{prompt_id[:12]}{ext}")
        target.write_bytes(r.content)
        return target


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from modules import audio_backend as ab
    backend = ab.get_active_backend()
    print(f"Backend: {backend.backend_id}")
    ok, msg = backend.health_check()
    print(f"Health: {msg}")
