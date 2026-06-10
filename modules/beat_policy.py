"""
Claw Bot — Beat Policy

Single source of truth for per-beat generation knobs:
  - image cfg (calmer for breathing shots)
  - video cfg
  - max frame count (cap atmosphere/reaction clips so 30-sec story stays 30-sec)
  - 4-step Wan LoRA toggle (use fast path for low-motion beats)
  - extra negative prompt terms (suppress characters in atmosphere etc.)

Consumed by:
  - storyboard_generator → image cfg + negative_augment
  - clip_generator      → max frame count + video cfg + lora_4step
  - video_workflow      → reads via clip_generator
"""

from typing import Optional


# ==============================================================================
# PER-BEAT POLICY TABLE
# ==============================================================================
#
# Notes:
#   - frame_cap_sec: hard upper bound on per-clip duration regardless of TTS.
#     Atmosphere/reaction shots should not eat 10+ seconds of the 30-sec story.
#   - image_cfg_bias: ADDED to backend default cfg. Negative = calmer.
#   - video_cfg_override: REPLACES backend default cfg.
#   - lora_4step: True = use Wan 4-step lightx2v LoRA (~4x faster, less motion fidelity).
#   - negative_augment: extra terms appended to negative prompt.

BEAT_POLICY = {
    "atmosphere": {
        "frame_cap_sec": 3.0,
        "image_cfg_bias": -0.5,
        "video_cfg_override": 2.0,
        "lora_4step": True,
        "negative_augment": (
            "people, person, child, kid, adult, character, figure, silhouette, "
            "humanoid, animal, creature, hands, face, body"
        ),
    },
    "reaction": {
        "frame_cap_sec": 3.0,
        "image_cfg_bias": 0.0,
        "video_cfg_override": 2.5,
        "lora_4step": True,
        "negative_augment": "wide shot, full body, distant view, multiple people, crowd",
    },
    "observation": {
        "frame_cap_sec": 4.0,
        "image_cfg_bias": 0.0,
        "video_cfg_override": 2.8,
        "lora_4step": True,
        "negative_augment": "fast motion, action pose, running, jumping",
    },
    "moment-of-decision": {
        "frame_cap_sec": 4.0,
        "image_cfg_bias": 0.0,
        "video_cfg_override": 2.5,
        "lora_4step": True,
        "negative_augment": "fast motion, action pose, walking, running, multiple characters",
    },
    "hook": {
        "frame_cap_sec": 5.0,
        "image_cfg_bias": 0.0,
        "video_cfg_override": None,
        "lora_4step": True,
        "negative_augment": "",
    },
    "spark": {
        "frame_cap_sec": 5.0,
        "image_cfg_bias": 0.0,
        "video_cfg_override": None,
        "lora_4step": True,
        "negative_augment": "static pose, no motion",
    },
    "struggle": {
        "frame_cap_sec": 5.0,
        "image_cfg_bias": 0.0,
        "video_cfg_override": None,
        "lora_4step": True,
        "negative_augment": "static pose, no motion",
    },
    "choice": {
        "frame_cap_sec": 5.0,
        "image_cfg_bias": 0.0,
        "video_cfg_override": None,
        "lora_4step": True,
        "negative_augment": "",
    },
    "consequence": {
        "frame_cap_sec": 5.0,
        "image_cfg_bias": 0.0,
        "video_cfg_override": None,
        "lora_4step": True,
        "negative_augment": "",
    },
}

DEFAULT_POLICY = {
    "frame_cap_sec": 5.0,
    "image_cfg_bias": 0.0,
    "video_cfg_override": None,
    "lora_4step": True,
    "negative_augment": "",
}


def get_policy(beat: Optional[str]) -> dict:
    if not beat:
        return dict(DEFAULT_POLICY)
    return dict(BEAT_POLICY.get(beat.strip().lower(), DEFAULT_POLICY))


def frame_cap_for(beat: Optional[str], fps: int) -> int:
    """Hard upper bound on frame count for this beat."""
    cap_sec = get_policy(beat).get("frame_cap_sec", DEFAULT_POLICY["frame_cap_sec"])
    return max(int(round(cap_sec * fps)), fps)


def image_cfg_for(beat: Optional[str], backend_default_cfg: float) -> float:
    bias = get_policy(beat).get("image_cfg_bias", 0.0)
    return max(0.5, float(backend_default_cfg) + float(bias))


def video_cfg_for(beat: Optional[str], backend_default_cfg: float) -> float:
    override = get_policy(beat).get("video_cfg_override")
    if override is not None:
        return float(override)
    return float(backend_default_cfg)


def lora_4step_for(beat: Optional[str]) -> bool:
    return bool(get_policy(beat).get("lora_4step", False))


def negative_augment_for(beat: Optional[str]) -> str:
    return str(get_policy(beat).get("negative_augment", "")).strip()


def merge_negative(base_negative: str, beat: Optional[str]) -> str:
    extra = negative_augment_for(beat)
    if not extra:
        return base_negative
    if not base_negative:
        return extra
    return base_negative.rstrip(", .") + ", " + extra
