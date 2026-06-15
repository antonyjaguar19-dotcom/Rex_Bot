"""
Claw Bot — Clip Generator

Orchestrates a single shot from script + storyboard → finished narrated MP4.

Strategy A (TTS-first):
  1. Synthesize narration with Kokoro
  2. Measure audio duration
  3. Generate video matching that duration (frame_count = duration * fps)
  4. Mux video + audio with ffmpeg → final MP4

This module is pure orchestration — no Discord, no LLM, no auto-trigger.
The bot calls this for each approved shot.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure 02_Agent on sys.path for consistent imports
_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules import video_backend as vb
from modules.tts_engine import TTSEngine
from modules import runtime_settings as rs
from modules import beat_policy as bp
from modules import gpu_utils

log = logging.getLogger("claw_bot.clip_generator")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FFMPEG_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffprobe.exe"
CLIPS_DIR = PROJECT_ROOT / "04_Outputs" / "clips"

# Sync mode — exposed for runtime override later
SYNC_MODE_STRICT = "strict"   # Video frames = exact audio duration
SYNC_MODE_LOOSE  = "loose"    # Video has padding/buffer, trim later
DEFAULT_SYNC_MODE = SYNC_MODE_STRICT
DEFAULT_FPS = 16              # Fallback only — real fps comes from the active backend's default_fps.
DEFAULT_BUFFER_SECS = 0.5     # Used in 'loose' mode


class ClipGenerator:
    """
    Build one fully-narrated clip per shot.
    """

    def __init__(
        self,
        video_backend=None,
        tts_engine: Optional[TTSEngine] = None,
        sync_mode: str = DEFAULT_SYNC_MODE,
        fps: Optional[int] = None,
    ):
        self.video = video_backend or vb.get_active_backend()
        self.tts = tts_engine or TTSEngine()
        self.sync_mode = sync_mode
        # fps is model-specific — derive from the active backend's config so each
        # model runs at its own recommended frame rate.
        backend_cfg = getattr(self.video, "config", {}) or {}
        self.fps = int(fps) if fps is not None else int(backend_cfg.get("default_fps", DEFAULT_FPS))

        if not FFMPEG_EXE.exists():
            raise FileNotFoundError(
                f"ffmpeg not found at {FFMPEG_EXE}. "
                f"Run install_ffmpeg.ps1 first."
            )
        if not FFPROBE_EXE.exists():
            raise FileNotFoundError(f"ffprobe not found at {FFPROBE_EXE}")

        # Lazy S2V (lip-sync) backend — loaded on first character shot when
        # lip-sync is enabled. _s2v_failed latches so we don't retry every shot.
        self._s2v = None
        self._s2v_failed = False

        log.info(
            f"ClipGenerator ready — sync={sync_mode}, fps={self.fps}, "
            f"video={self.video.backend_id} (single video per shot, no splitting)"
        )

    def _get_s2v_backend(self):
        """Return the Wan-S2V lip-sync backend, or None when disabled/unavailable.
        Loaded once and cached; failure latches to a graceful I2V fallback."""
        if not rs.get_lipsync_enabled() or self._s2v_failed:
            return None
        if self._s2v is not None:
            return self._s2v
        try:
            from modules import model_registry as _mr
            cfg = _mr.get_available("video_backend", rs.get_lipsync_backend_id())
            if not cfg:
                log.warning("Lip-sync enabled but S2V backend not in registry; "
                            "using I2V.")
                self._s2v_failed = True
                return None
            backend = vb.build_backend(cfg)
            ok, msg = backend.health_check()
            if not ok:
                log.warning(f"S2V backend health check failed: {msg}; using I2V.")
                self._s2v_failed = True
                return None
            self._s2v = backend
            log.info(f"S2V lip-sync backend ready: {cfg.get('_id')}")
            return self._s2v
        except Exception as e:
            log.warning(f"S2V backend load failed ({e}); using I2V.")
            self._s2v_failed = True
            return None

    # ----------------- Public API -----------------

    def generate_clip(
        self,
        shot_id: str,
        narration: str,
        action_prompt: str,
        storyboard_image: Path,
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
        beat: Optional[str] = None,
        voice: Optional[str] = None,
        lipsync: bool = False,
    ) -> Path:
        """
        Produce one finished narrated clip for a shot.

        Args:
            shot_id: Identifier for filenames (e.g. "shot1").
            narration: The voice-over line for this shot.
            action_prompt: Description of the visual motion (for video model).
            storyboard_image: Path to the starting frame.
            output_filename: Override final MP4 name. Default: clip_{shot_id}.mp4
            seed: Optional fixed seed for video gen reproducibility.

        Returns:
            Path to the muxed MP4.
        """
        if not storyboard_image.exists():
            raise FileNotFoundError(f"Storyboard missing: {storyboard_image}")

        log.info(f"=== Generating clip: {shot_id} ===")
        log.info(f"Narration ({len(narration)} chars): {narration[:80]}...")
        if voice:
            log.info(f"Speaker voice: {voice}")

        # Step 1 — TTS first (Strategy A). One audio file for the whole shot.
        # `voice` selects the speaker's Kokoro voice (narrator vs character);
        # None falls back to the engine's configured default.
        audio_path = self.tts.synthesize(
            narration,
            output_path=CLIPS_DIR / "_temp" / f"audio_{shot_id}.wav",
            voice=voice,
        )
        audio_duration = self._probe_duration(audio_path)
        log.info(f"Audio duration: {audio_duration:.2f}s")

        # ONE video per shot — no splitting, no continuation chaining. LTX
        # generates the whole narration length in a single pass. Frame count is
        # sized to the full narration; the mux caps video to audio so matching
        # frames to audio is safe.
        round_fn = getattr(self.video, "round_frame_count", None)

        def _frames_for(seconds: float) -> int:
            f = self._frames_for_duration(seconds)
            if callable(round_fn):
                return round_fn(f)
            return self._round_to_ltx2_frames(f)

        frame_count = _frames_for(audio_duration)
        log.info(
            f"Single-pass shot — {frame_count} frames @ {self.fps}fps "
            f"(~{frame_count/self.fps:.2f}s for {audio_duration:.2f}s narration)"
        )

        # Beat-aware video knobs: cfg + 4-step LoRA toggle.
        backend_default_cfg = float(getattr(self.video, "cfg", 3.5))
        effective_video_cfg = bp.video_cfg_for(beat, backend_default_cfg)
        effective_lora_4step = bp.lora_4step_for(beat)

        temp_dir = CLIPS_DIR / "_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # VRAM pre-flight before the heavy video gen.
        # Unload Ollama ONLY — never call ComfyUI /free here. /free evicts the
        # video model and forces a full cold reload on every shot (7-10min load
        # tax each time). Keeping it resident makes shot 1 cold (~7-10min) and
        # shots 2+ cached (~4-6min). Ollama unload frees room without touching it.
        try:
            gpu_utils.free_ollama_vram()
        except Exception:
            pass

        # Step 2 — Generate the video (try with beat overrides, fallback gracefully)
        gen_kwargs = dict(
            prompt=action_prompt,
            input_image=storyboard_image,
            frame_count=frame_count,
            fps=self.fps,
            output_filename=f"video_{shot_id}.mp4",
            seed=seed,
            cfg_override=effective_video_cfg,
            lora_4step_override=effective_lora_4step,
        )
        # Per-run video resolution override (None = backend default dims)
        video_res = rs.get_effective_video_resolution()
        if video_res:
            gen_kwargs["width"], gen_kwargs["height"] = video_res

        # ── Lip-sync routing (P4): CHARACTER shots → Wan-S2V (audio-driven mouth).
        # Narrator shots and any S2V failure fall through to the I2V path below.
        used_s2v = False
        s2v = self._get_s2v_backend() if lipsync else None
        if s2v is not None:
            try:
                s2v_kwargs = dict(
                    prompt=action_prompt,
                    input_image=storyboard_image,
                    audio_path=audio_path,
                    fps=self.fps,
                    seed=seed,
                    cfg_override=effective_video_cfg,
                    output_filename=f"video_{shot_id}.mp4",
                )
                if video_res:
                    s2v_kwargs["width"], s2v_kwargs["height"] = video_res
                log.info(f"{shot_id}: routing to S2V lip-sync backend")
                video_path = s2v.generate(**s2v_kwargs)
                self.last_seed = int(getattr(s2v, "_last_seed", -1) or -1)
                used_s2v = True
            except Exception as e:
                log.warning(f"{shot_id}: S2V lip-sync failed ({e}); "
                            f"falling back to I2V (no lip-sync).")

        if not used_s2v:
            try:
                video_path = self.video.generate(**gen_kwargs)
            except TypeError:
                # Older adapters may not accept every override kwarg — drop the
                # optional ones and retry with the core interface.
                for k in ("cfg_override", "lora_4step_override", "width", "height"):
                    gen_kwargs.pop(k, None)
                video_path = self.video.generate(**gen_kwargs)

        # Move video into temp area (uniform location)
        temp_video = temp_dir / f"video_{shot_id}.mp4"
        if video_path.resolve() != temp_video.resolve():
            video_path.replace(temp_video)
            video_path = temp_video

        # Step 3 — Mux video + audio (freeze-pad if video shorter) → final clip
        if output_filename is None:
            output_filename = f"clip_{shot_id}.mp4"
        final_path = CLIPS_DIR / output_filename
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        self._mux(
            video_path, audio_path, final_path,
            audio_duration=audio_duration,
        )

        if not used_s2v:
            self.last_seed = int(getattr(self.video, "_last_seed", -1) or -1)

        # Integrity guard: the finished clip must carry the full narration.
        # If it's shorter than the measured narration, the last word(s) were
        # trimmed somewhere upstream — surface it instead of shipping silently.
        final_dur = self._probe_duration(final_path)
        if final_dur < audio_duration - 0.15:
            log.warning(
                f"⚠️ {shot_id}: final clip {final_dur:.2f}s is SHORTER than "
                f"narration {audio_duration:.2f}s — narration may be trimmed."
            )
        log.info(
            f"Clip ready: {final_path.name} "
            f"({final_path.stat().st_size/(1024*1024):.2f} MB), "
            f"dur={final_dur:.2f}s vs narration={audio_duration:.2f}s, "
            f"seed={self.last_seed}"
        )
        return final_path

    # ----------------- Helpers -----------------

    def _probe_duration(self, media_path: Path) -> float:
        """Return media duration in seconds via ffprobe."""
        cmd = [
            str(FFPROBE_EXE),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                timeout=60)
        return float(result.stdout.strip())

    def _frames_for_duration(self, seconds: float) -> int:
        return max(int(round(seconds * self.fps)), self.fps)   # min 1 sec

    def _round_to_ltx2_frames(self, frames: int) -> int:
        """
        LTX-2 latent video requires frame_count = 8*N + 1 (8 = temporal stride).
        Round UP to the next valid count so we don't truncate audio.
        """
        if (frames - 1) % 8 == 0:
            return frames
        return ((frames - 1) // 8 + 1) * 8 + 1

    def _mux(self, video_path: Path, audio_path: Path, out_path: Path,
             audio_duration: Optional[float] = None):
        """
        Combine silent video + audio into one MP4.
        - Narration (audio) is the source of truth for duration.
        - If the rendered video is shorter than the audio, the last frame is
          held (frozen) until the audio ends. No truncation of narration.
        - Video is re-encoded only when padding is required (cheap on short
          clips, avoids missing-frame artefacts).
        - Audio always re-encoded to AAC for web compatibility.
        """
        if audio_duration is None:
            audio_duration = self._probe_duration(audio_path)
        video_duration = self._probe_duration(video_path)
        pad_seconds = max(0.0, audio_duration - video_duration)
        log.info(
            f"Muxing -> {out_path.name} "
            f"(audio={audio_duration:.2f}s, video={video_duration:.2f}s, "
            f"freeze-pad={pad_seconds:.2f}s)"
        )

        if pad_seconds > 0.05:
            # Pad video tail by holding the last frame for `pad_seconds`,
            # then mux. tpad re-encodes the video stream so c:v cannot copy.
            cmd = [
                str(FFMPEG_EXE),
                "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-filter_complex",
                f"[0:v]tpad=stop_mode=clone:stop_duration={pad_seconds:.3f},"
                f"fps={self.fps}[v]",
                "-map", "[v]",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                # No -shortest: audio is the source of truth, video already padded
                str(out_path),
            ]
        else:
            # Video already >= audio. Keep video stream-copy; cap to the
            # narration length so trailing silent video doesn't leave the user
            # staring at a frozen image after narration ends.
            # NOTE: use explicit `-t audio_duration`, NOT `-shortest`. With
            # `-c:v copy` a frame-timing quirk can make -shortest stop on the
            # video stream and clip the last word of narration. Capping by the
            # measured audio duration guarantees the narration is never trimmed.
            cmd = [
                str(FFMPEG_EXE),
                "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", f"{audio_duration:.3f}",
                str(out_path),
            ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mux failed:\n{result.stderr.strip()}"
            )


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Pick first storyboard as test image
    storyboard_dir = PROJECT_ROOT / "04_Outputs" / "storyboards"
    candidates = sorted(
        list(storyboard_dir.glob("*.png")) + list(storyboard_dir.glob("*.jpg"))
    )
    if not candidates:
        raise SystemExit(f"No storyboards in {storyboard_dir}")

    test_image = candidates[0]
    print(f"Using storyboard: {test_image.name}")

    gen = ClipGenerator()
    out = gen.generate_clip(
        shot_id="test1",
        narration=(
            "Today, we'll learn about being kind. "
            "When we share with friends, everyone feels happy."
        ),
        action_prompt=(
            "The character moves naturally within the scene. "
            "Soft camera drift, warm lighting, gentle motion."
        ),
        storyboard_image=test_image,
    )
    print(f"\n✅ Done — final clip: {out}")
