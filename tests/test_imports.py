"""Smoke test: every live module must import without crashing.

Catches the classic restart-killer — a typo'd import or missing dependency
that would only surface when the bot is relaunched at 9 AM for the daily
auto-story. Modules whose heavy deps (torch, audiocraft) may be absent are
allowed to skip, not fail.
"""

import importlib

import pytest

CORE_MODULES = [
    "modules.file_utils",
    "modules.job_lock",
    "modules.config_check",
    "modules.sync_bridge",
    "modules.runtime_settings",
    "modules.model_registry",
    "modules.pending_feedback",
    "modules.generation_meta",
    "modules.agent",
    "modules.agent_router",
    "modules.theme_bank",
    "modules.safety_filter",
    "modules.beat_policy",
    "modules.progress_bar",
    "modules.embed_styles",
    "modules.contact_sheet",
    "modules.card_generator",
    "modules.voice_casting",
    "modules.video_backends.comfyui_wan_s2v",
    "modules.story_writer",
    "modules.script_generator",
    "modules.prompt_assembler",
    "modules.prompt_polisher",
    "modules.prompt_approval",
    "modules.shot_tailor",
    "modules.feedback_thinker",
    "modules.storyboard_generator",
    "modules.storyboard_workflow",
    "modules.video_workflow",
    "modules.clip_generator",
    "modules.assembly",
    "modules.narration_segmenter",
    "modules.tts_elevenlabs",
    "modules.tts_voxcpm",
    "modules.tts_chatterbox",
    "modules.voice_bank",
    "modules.audio_first_pipeline",
    "modules.upscaler",
    "modules.gpu_utils",
    "modules.health_monitor",
    "modules.image_backend",
    "modules.video_backend",
    "modules.control_panel",
    "modules.approval_buttons",
    "modules.channel_cleanup",
    "modules.dashboard_nicegui",
    "modules.image_backends.comfyui_zimage_base",
    "modules.image_backends.comfyui_flux2",
    "modules.image_backends.comfyui_kontext_base",
    "modules.video_backends.comfyui_wan22",
    "modules.video_backends.comfyui_wan22_14B",
    "modules.video_backends.comfyui_ltx_video",
]

# Optional heavy deps (torch / audiocraft / kokoro) may be missing in a
# fresh checkout; import errors from THOSE specific packages are skips.
OPTIONAL_DEP_MODULES = {
    "modules.music_generator",
    "modules.tts_engine",
}


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", sorted(OPTIONAL_DEP_MODULES))
def test_optional_module_imports(name):
    try:
        importlib.import_module(name)
    except ImportError as e:
        pytest.skip(f"optional dependency missing: {e}")
