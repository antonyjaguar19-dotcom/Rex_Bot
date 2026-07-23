"""
Claw Bot — LTX-2.3 Video Adapter (via ComfyUI API)

Drives the PROVEN, frontend-exported LTX-2.3 API graph (05_Config/workflows/
IA2V_LTX2.3.json) — the same 22B fp8 + distilled-LoRA two-stage pipeline that
already renders on this 16 GB card (streams via offload, no OOM).

Two modes, one graph:
  • image-to-video (i2v): one start frame + a SILENT audio track. LTX-2.3 is
    fundamentally an audio-video model, so the clean way to get pure video is to
    feed it silence (the manual pipeline re-muxes narration/music at assembly
    anyway, so the embedded silent track is never heard).
  • first-last-frame (flf2v): start frame + END frame guide. Set via `end_image`
    (config workflow_file -> the flf2v API graph); falls back to i2v when absent.

Why not the old GGUF `comfyui_ltx2`: that entry named a 19B GGUF that isn't
installed and loads incomplete (missing AV-connector weights) and offloads to
~2h/clip. This one uses the weights that are actually on disk.

Slow by nature: ~7-12 min per 7-9 s clip on the 5080 (distilled 4-step base +
upsample refine). Single pass, 24 fps, no chaining.
"""

import json
import logging
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

_AGENT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules.video_backend import VideoBackend
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.video_backend.ltx23")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
WORKFLOWS_DIR = PROJECT_ROOT / "05_Config" / "workflows"
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "videos"
SILENT_WAV = WORKFLOWS_DIR / "ltx_silent_20s.wav"      # covers any clip <= 20s (graph trims)

DEFAULT_WORKFLOW_FILE = "IA2V_LTX2.3.json"
# Dims must be multiples of 32 (LTX latent stride); match the Qwen still aspect.
_AR_DIMS = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1024, 1024)}
DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, blurry, static, frozen"


class Backend(VideoBackend):
    """LTX-2.3 22B fp8 (distilled) I2V / FLF2V via ComfyUI API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        self.default_width  = int(config.get("default_width", 1280))
        self.default_height = int(config.get("default_height", 720))
        self.default_fps    = int(config.get("default_fps", 24))
        self.default_duration = float(config.get("default_duration", 5.0))
        self.max_clip_seconds = float(config.get("max_clip_seconds", 12.0))
        # IA2V variant: the shot's real audio drives the clip (lip-sync). i2v keeps
        # feeding silence so the video has no imposed motion from sound.
        self.audio_driven = bool(config.get("audio_driven", False))
        # flf2v backends point at their own exported API graph; i2v uses the ia2v one.
        wf_name = config.get("workflow_file", DEFAULT_WORKFLOW_FILE)
        wf_path = WORKFLOWS_DIR / wf_name
        if not wf_path.exists():
            raise FileNotFoundError(
                f"LTX-2.3 API workflow not found: {wf_path}. Export it from ComfyUI "
                f"GUI (Workflow -> Export (API Format)).")
        self._workflow_template = json.loads(wf_path.read_text(encoding="utf-8"))
        self._workflow_name = wf_name
        # Does this graph take an END-frame guide? (a 2nd LoadImage titled end/last)
        self._has_end_frame = any(
            isinstance(n, dict) and n.get("class_type") == "LoadImage"
            and any(w in (n.get("_meta", {}).get("title", "").lower()) for w in ("end", "last"))
            for n in self._workflow_template.values())
        # Does it take an audio track? (i2v/ia2v do; flf2v does not)
        self._has_audio = any(
            isinstance(n, dict) and n.get("class_type") == "LoadAudio"
            for n in self._workflow_template.values())
        self._last_seed = -1
        log.info(f"LTX-2.3 adapter initialized — workflow {wf_name}, "
                 f"end_frame={self._has_end_frame}")

    # ---------------- contract ----------------

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

    def format_prompt_for_backend(self, raw_prompt: str) -> str:
        return raw_prompt.strip()

    def generate(
        self,
        prompt: str,
        input_image: Path,
        negative_prompt: Optional[str] = None,
        frame_count: int = None,
        fps: int = None,
        aspect_ratio: str = "16:9",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
        end_image: Optional[Path] = None,
        mid_images: Optional[list] = None,        # flf2v: keyframes BETWEEN first and last
        audio_path: Optional[Path] = None,        # IA2V: real audio drives the clip
        width: Optional[int] = None,
        height: Optional[int] = None,
        cfg_override: Optional[float] = None,      # accepted, unused (distilled cfg pinned)
        lora_4step_override: Optional[bool] = None,  # accepted, unused
    ) -> Path:
        input_image = Path(input_image)
        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        self._last_seed = int(seed)
        if fps is None:
            fps = self.default_fps
        _ar = (aspect_ratio or "16:9").replace("x", ":").strip()
        _aw, _ah = _AR_DIMS.get(_ar, (self.default_width, self.default_height))
        eff_w = int(width) if width else _aw
        eff_h = int(height) if height else _ah
        # duration seconds from frame_count (graph is duration-based: frames = dur*fps+1)
        if frame_count is None:
            duration = self.default_duration
        else:
            duration = max((frame_count - 1) / fps, 1.0)
        # IA2V: the audio drives the clip, so its length wins (capped).
        use_audio = bool(self.audio_driven and audio_path and Path(audio_path).exists())
        if use_audio:
            adur = self._audio_duration(Path(audio_path))
            if adur and adur > 0:
                duration = adur
        duration = min(duration, self.max_clip_seconds)

        negative = (negative_prompt or DEFAULT_NEGATIVE).strip()

        log.info(f"LTX-2.3 gen — seed={seed}, {duration:.2f}s @{fps}fps, "
                 f"{eff_w}x{eff_h}, end_frame={bool(end_image)}, image={input_image.name}")

        if self._has_end_frame and end_image is None:
            raise ValueError(
                "This LTX-2.3 backend is first-last-frame — it needs an END frame. "
                "Set the shot's 'last frame' image, or pick the plain i2v backend.")

        # 1) upload start frame (+ optional end frame) + silent audio (if graph uses it)
        img_name = self._upload(input_image, "image/png")
        end_name = None
        if end_image is not None and self._has_end_frame:
            end_image = Path(end_image)
            if not end_image.exists():
                raise FileNotFoundError(f"End frame not found: {end_image}")
            end_name = self._upload(end_image, "image/png")
        # Audio track: IA2V uses the shot's real audio (lip-sync); i2v uses silence.
        audio_name = None
        if self._has_audio:
            if use_audio:
                audio_name = self._upload(Path(audio_path), "audio/wav")
                log.info(f"IA2V — driving with audio {Path(audio_path).name} ({duration:.2f}s)")
            else:
                if self.audio_driven:
                    log.info("IA2V backend but shot has no audio — falling back to silence.")
                audio_name = self._ensure_silent_audio()

        # optional in-between keyframes (flf2v only): upload each middle still
        mid_names = []
        if mid_images and self._has_end_frame:
            for mp in mid_images:
                mp = Path(mp)
                if mp.exists():
                    mid_names.append(self._upload(mp, "image/png"))

        # 2) inject
        wf = self._build_workflow(
            prompt=self.format_prompt_for_backend(prompt), negative=negative,
            image_name=img_name, end_name=end_name, audio_name=audio_name,
            width=eff_w, height=eff_h, duration=duration, fps=fps, seed=seed)

        # 2b) insert middle keyframe guides (evenly spaced between first and last)
        if mid_names:
            self._insert_midframes(wf, mid_names, duration, fps)

        # 3) submit + wait + download
        client_id = str(uuid.uuid4())
        prompt_id = self._submit(wf, client_id)
        log.info(f"Submitted LTX-2.3 prompt_id={prompt_id}")
        history = self._wait(prompt_id, timeout=int(self.max_clip_seconds * 180 + 1200))
        return self._download(history, prompt_id, output_filename)

    # ---------------- helpers ----------------

    def _audio_duration(self, path: Path) -> float:
        try:
            from modules.assembly import _probe_duration
            return float(_probe_duration(path))
        except Exception as e:
            log.warning(f"could not probe audio duration ({e})")
            return 0.0

    def _ensure_silent_audio(self) -> str:
        if not SILENT_WAV.exists():
            # regenerate a 20s mono silence if the asset went missing
            from modules.assembly import FFMPEG_EXE
            subprocess.run([str(FFMPEG_EXE), "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", "anullsrc=r=24000:cl=mono", "-t", "20",
                            str(SILENT_WAV)], timeout=60)
        return self._upload(SILENT_WAV, "audio/wav")

    def _upload(self, path: Path, mime: str) -> str:
        url = f"{self.server_url}/upload/image"   # ComfyUI stages any file into input/
        with open(path, "rb") as f:
            files = {"image": (path.name, f, mime)}
            r = requests.post(url, files=files, data={"overwrite": "true"}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Upload failed for {path.name} (HTTP {r.status_code}): {r.text[:300]}")
        return r.json().get("name", path.name)

    def _build_workflow(self, prompt, negative, image_name, end_name, audio_name,
                        width, height, duration, fps, seed) -> dict:
        wf = json.loads(json.dumps(self._workflow_template))
        inj = {"image": False, "audio": False, "prompt": False, "negative": False,
               "width": False, "height": False, "duration": False, "fps": False,
               "seed": False, "end": False}
        start_image_done = False

        for node in wf.values():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type", "")
            title = node.get("_meta", {}).get("title", "").lower()
            ins = node.setdefault("inputs", {})

            if ct == "LoadImage":
                if any(w in title for w in ("end", "last")) and end_name:
                    ins["image"] = end_name
                    inj["end"] = True
                elif not start_image_done:
                    ins["image"] = image_name
                    inj["image"] = True
                    start_image_done = True
            elif ct == "LoadAudio" and audio_name:
                ins["audio"] = audio_name
                ins.pop("audioUI", None)
                inj["audio"] = True
            elif ct == "PrimitiveStringMultiline" and "prompt" in title:
                ins["value"] = prompt
                inj["prompt"] = True
            elif ct == "CLIPTextEncode" and "positive" in title:
                # flf2v: positive prompt is a literal CLIPTextEncode (titled by surgery)
                ins["text"] = prompt
                inj["prompt"] = True
            elif ct == "CLIPTextEncode" and "negative" in title:
                ins["text"] = negative
                inj["negative"] = True
            elif ct == "CLIPTextEncode" and not isinstance(ins.get("text"), list) \
                    and not inj["negative"]:
                # i2v/ia2v: the one literal CLIPTextEncode is the negative
                ins["text"] = negative
                inj["negative"] = True
            elif ct == "PrimitiveInt" and title == "width":
                ins["value"] = int(width); inj["width"] = True
            elif ct == "PrimitiveInt" and title == "height":
                ins["value"] = int(height); inj["height"] = True
            elif ct == "PrimitiveInt" and "frame rate" in title:
                ins["value"] = int(fps); inj["fps"] = True
            elif ct in ("PrimitiveFloat", "PrimitiveInt") and "duration" in title:
                ins["value"] = (float(duration) if ct == "PrimitiveFloat"
                                else int(round(duration)))
                inj["duration"] = True
            elif ct == "RandomNoise":
                ins["noise_seed"] = int(seed)
                inj["seed"] = True

        log.info("LTX-2.3 injection: " + " ".join(f"{k}={v}" for k, v in inj.items()))
        for k in ("image", "prompt", "width", "height", "duration", "fps", "seed"):
            if not inj[k]:
                raise RuntimeError(
                    f"LTX-2.3 workflow injection failed for '{k}' — {self._workflow_name} "
                    f"changed shape. Re-export from ComfyUI (Export API Format).")
        return wf

    def _insert_midframes(self, wf: dict, mid_names: list,
                          duration: float, fps: int) -> None:
        """Add in-between keyframe guides to the flf2v graph. Each middle still
        gets its own LoadImage → ResizeImageMaskNode → LTXVPreprocess → LTXVAddGuide
        (a copy of the first/last frame pre-chain), placed at an evenly-spaced frame
        index, and threaded into the guide chain BEFORE the last-frame guide.

        Chain becomes: first(0) → mid_1 → … → mid_k → last(-1) → CFGGuider.
        Mutates wf in place. No-op if the graph shape isn't the expected flf2v one.
        """
        # Only the BASE guide chain — skip the refine-stage re-guides (two-stage HQ
        # graph), which carry a "Refine" title. Mids shape the base latent; the
        # refine pass re-guides first+last only and sharpens what the base produced.
        guides = {nid: n for nid, n in wf.items()
                  if isinstance(n, dict) and n.get("class_type") == "LTXVAddGuide"
                  and "refine" not in (n.get("_meta", {}).get("title", "").lower())}
        first = next((nid for nid, n in guides.items()
                      if n["inputs"].get("frame_idx") == 0), None)
        last = next((nid for nid, n in guides.items()
                     if n["inputs"].get("frame_idx") == -1), None)
        if not first or not last:
            log.warning("mid-frames: no first/last guide found — skipping keyframes.")
            return
        # A sample Resize + Preprocess to copy width/height links + params from.
        sample_resize = next((n for n in wf.values() if isinstance(n, dict)
                              and n.get("class_type") == "ResizeImageMaskNode"), None)
        sample_prep = next((n for n in wf.values() if isinstance(n, dict)
                            and n.get("class_type") == "LTXVPreprocess"), None)
        if not sample_resize or not sample_prep:
            log.warning("mid-frames: resize/preprocess template missing — skipping.")
            return
        vae = wf[last]["inputs"].get("vae")
        strength = wf[first]["inputs"].get("strength", 0.7)

        # Frame index of each middle, evenly spaced over the clip length.
        total = max(int(round(duration * fps)) + 1, len(mid_names) + 2)
        k = len(mid_names)
        idxs = [max(1, min(total - 2, int(round((i + 1) / (k + 1) * (total - 1)))))
                for i in range(k)]

        uid = f"mf{uuid.uuid4().hex[:6]}"
        prev = first
        for i, (name, fidx) in enumerate(zip(mid_names, idxs)):
            lid, rid, pid, gid = (f"{uid}_l{i}", f"{uid}_r{i}", f"{uid}_p{i}", f"{uid}_g{i}")
            wf[lid] = {"class_type": "LoadImage", "inputs": {"image": name},
                       "_meta": {"title": f"Load Mid {i+1}"}}
            rz = dict(sample_resize["inputs"])
            rz["input"] = [lid, 0]
            wf[rid] = {"class_type": "ResizeImageMaskNode", "inputs": rz,
                       "_meta": {"title": f"Resize Mid {i+1}"}}
            pp = dict(sample_prep["inputs"])
            pp["image"] = [rid, 0]
            wf[pid] = {"class_type": "LTXVPreprocess", "inputs": pp,
                       "_meta": {"title": f"Preprocess Mid {i+1}"}}
            wf[gid] = {"class_type": "LTXVAddGuide", "_meta": {"title": f"AddGuide Mid {i+1}"},
                       "inputs": {"positive": [prev, 0], "negative": [prev, 1],
                                  "vae": vae, "latent": [prev, 2], "image": [pid, 0],
                                  "frame_idx": int(fidx), "strength": strength}}
            prev = gid
        # Re-thread the last-frame guide to sit AFTER the middles.
        wf[last]["inputs"]["positive"] = [prev, 0]
        wf[last]["inputs"]["negative"] = [prev, 1]
        wf[last]["inputs"]["latent"] = [prev, 2]
        log.info(f"mid-frames: inserted {k} keyframe guide(s) at frames {idxs} "
                 f"(clip length ~{total}).")

    def _submit(self, workflow: dict, client_id: str) -> str:
        r = requests.post(f"{self.server_url}/prompt",
                          json={"prompt": workflow, "client_id": client_id}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected prompt (HTTP {r.status_code}): {r.text[:600]}")
        data = r.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"ComfyUI response missing prompt_id: {data}")
        return data["prompt_id"]

    def _wait(self, prompt_id: str, timeout: int) -> dict:
        history_url = f"{self.server_url}/history/{prompt_id}"
        queue_url = f"{self.server_url}/queue"
        start = time.time(); last = start; fails = 0
        while time.time() - start < timeout:
            try:
                r = requests.get(history_url, timeout=60)
                fails = 0
                if r.status_code == 200:
                    data = r.json()
                    if prompt_id in data:
                        entry = data[prompt_id]
                        st = entry.get("status", {})
                        if st.get("completed") is True:
                            log.info(f"LTX-2.3 {prompt_id[:8]} done in {int(time.time()-start)}s")
                            return entry
                        if st.get("status_str") == "error":
                            raise RuntimeError(f"ComfyUI error: {st.get('messages')}")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                fails += 1
                if fails >= 5:
                    raise RuntimeError(f"Lost ComfyUI for {prompt_id[:8]} after 5 timeouts") from e
                time.sleep(10); continue
            now = time.time()
            if now - last >= 30:
                last = now
                try:
                    q = requests.get(queue_url, timeout=10).json()
                    log.info(f"Waiting LTX-2.3 {prompt_id[:8]}: {int(now-start)}s, "
                             f"running={len(q.get('queue_running', []))} "
                             f"pending={len(q.get('queue_pending', []))}")
                except Exception:
                    log.info(f"Waiting LTX-2.3 {prompt_id[:8]}: {int(now-start)}s")
            time.sleep(2)
        raise TimeoutError(f"Timed out after {timeout}s waiting for {prompt_id}")

    def _download(self, history_entry: dict, prompt_id: str,
                  output_filename: Optional[str]) -> Path:
        outputs = history_entry.get("outputs", {})
        if not outputs:
            raise RuntimeError("ComfyUI finished but produced no outputs.")
        info = None
        for _nid, out in outputs.items():
            for key in ("videos", "gifs", "images"):
                items = out.get(key)
                if items:
                    info = next((i for i in items
                                 if str(i.get("filename", "")).lower().endswith((".mp4", ".webm"))),
                                items[0])
                    break
            if info:
                break
        if info is None:
            raise RuntimeError("No video in LTX-2.3 outputs.")
        view = (f"{self.server_url}/view?filename={info['filename']}"
                f"&subfolder={info.get('subfolder', '')}&type={info.get('type', 'output')}")
        r = requests.get(view, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"Could not download video: HTTP {r.status_code}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(info["filename"]).suffix or ".mp4"
        target = OUTPUT_DIR / (output_filename or f"vid_ltx23_{prompt_id[:12]}{ext}")
        target.write_bytes(r.content)
        log.info(f"LTX-2.3 saved: {target}")
        return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from modules import model_registry, video_backend as vb
    cfg = model_registry.get_available("video_backend", "comfyui_ltx23_i2v")
    b = Backend(cfg)
    print("health:", b.health_check())
