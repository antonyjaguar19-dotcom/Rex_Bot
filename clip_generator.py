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

log = logging.getLogger("claw_bot.clip_generator")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FFMPEG_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffprobe.exe"
CLIPS_DIR = PROJECT_ROOT / "04_Outputs" / "clips"

# Sync mode — exposed for runtime override later
SYNC_MODE_STRICT = "strict"   # Video frames = exact audio duration
SYNC_MODE_LOOSE  = "loose"    # Video has padding/buffer, trim later
DEFAULT_SYNC_MODE = SYNC_MODE_STRICT
DEFAULT_FPS = 24
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
        fps: int = DEFAULT_FPS,
    ):
        self.video = video_backend or vb.get_active_backend()
        self.tts = tts_engine or TTSEngine()
        self.sync_mode = sync_mode
        self.fps = fps

        if not FFMPEG_EXE.exists():
            raise FileNotFoundError(
                f"ffmpeg not found at {FFMPEG_EXE}. "
                f"Run install_ffmpeg.ps1 first."
            )
        if not FFPROBE_EXE.exists():
            raise FileNotFoundError(f"ffprobe not found at {FFPROBE_EXE}")

        log.info(
            f"ClipGenerator ready — sync={sync_mode}, fps={fps}, "
            f"video={self.video.backend_id}"
        )

    # ----------------- Public API -----------------

    def generate_clip(
        self,
        shot_id: str,
        narration: str,
        action_prompt: str,
        storyboard_image: Path,
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
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

        # Step 1 — TTS first (Strategy A)
        audio_path = self.tts.synthesize(
            narration,
            output_path=CLIPS_DIR / "_temp" / f"audio_{shot_id}.wav",
        )
        audio_duration = self._probe_duration(audio_path)
        log.info(f"Audio duration: {audio_duration:.2f}s")

        # Step 2 — Compute frame count for video to match audio
        if self.sync_mode == SYNC_MODE_STRICT:
            frame_count = self._frames_for_duration(audio_duration)
        else:  # loose
            frame_count = self._frames_for_duration(audio_duration + DEFAULT_BUFFER_SECS)
        # Let the backend round to its own latent constraint (Wan: 4N+1, LTX: 8N+1).
        # Falls back to the legacy 8N+1 rounding if the backend doesn't expose one.
        round_fn = getattr(self.video, "round_frame_count", None)
        if callable(round_fn):
            frame_count = round_fn(frame_count)
        else:
            frame_count = self._round_to_ltx2_frames(frame_count)
        log.info(f"Video target: {frame_count} frames @ {self.fps}fps "
                 f"(~{frame_count/self.fps:.2f}s)")

        # Step 3 — Video gen
        video_path = self.video.generate(
            prompt=action_prompt,
            input_image=storyboard_image,
            frame_count=frame_count,
            fps=self.fps,
            output_filename=f"video_{shot_id}.mp4",
            seed=seed,
        )
        # Move video to temp area (avoids cluttering 04_Outputs/videos/)
        temp_video = CLIPS_DIR / "_temp" / f"video_{shot_id}.mp4"
        temp_video.parent.mkdir(parents=True, exist_ok=True)
        if video_path.resolve() != temp_video.resolve():
            video_path.replace(temp_video)
            video_path = temp_video

        # Step 4 — Mux
        if output_filename is None:
            output_filename = f"clip_{shot_id}.mp4"
        final_path = CLIPS_DIR / output_filename
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)

        self._mux(video_path, audio_path, final_path)
        log.info(f"Clip ready: {final_path.name} "
                 f"({final_path.stat().st_size/(1024*1024):.2f} MB)")

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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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

    def _mux(self, video_path: Path, audio_path: Path, out_path: Path):
        """
        Combine silent video + audio into one MP4.
        - Audio is the source of truth for duration (-shortest)
        - Video is re-encoded if needed (LTX-2 yuv420p output is web-safe)
        - Audio is encoded to AAC for max compatibility
        """
        log.info(f"Muxing -> {out_path.name}")
        cmd = [
            str(FFMPEG_EXE),
            "-y",                          # overwrite without prompting
            "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",               # video from input 0
            "-map", "1:a:0",               # audio from input 1
            "-c:v", "copy",                # don't re-encode video
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",                   # cut to shorter stream (audio in our case)
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
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
