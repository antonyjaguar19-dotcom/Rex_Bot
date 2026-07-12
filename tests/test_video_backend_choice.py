"""Which model animates the mascot, and the bugs that made LTX look broken.

Wan won the bake-off. LTX-2 rendered a sharp first frame and then dissolved the
character into smeared paste by the end of the clip, at 21.8 min per clip against
Wan's ~4. But two of the three things that made LTX look bad were OUR bugs, and
they are fixed here — the model is simply not good enough, which is a different
statement from "our adapter was wrong".
"""
import json
import pytest

from modules import runtime_settings as rs


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "SETTINGS_PATH", tmp_path / "runtime_settings.json")


# ---------------------------------------------------------------- S2V toggle

def test_mascot_lipsync_is_off_by_default():
    """Wan S2V is a TALKING-HEAD model. It syncs a mouth well but has no idea what
    hands, props or legs are: it dissolved a paw mid-clip and bent a leg backwards.
    Default is I2V motion with the narration as a voice-over."""
    assert rs.get_facts_mascot_lipsync() is False


def test_mascot_lipsync_round_trips():
    rs.set_facts_mascot_lipsync(True)
    assert rs.get_facts_mascot_lipsync() is True
    rs.set_facts_mascot_lipsync(False)
    assert rs.get_facts_mascot_lipsync() is False


# ---------------------------------------------------------------- LTX adapter

def _ltx_backend(tmp_path):
    from modules.video_backends import comfyui_ltx_video as ltx
    wf = {
        "1": {"class_type": "ResizeImageMaskNode",
              "inputs": {"resize_type.width": 832, "resize_type.height": 480}},
        "2": {"class_type": "ImageScaleBy", "inputs": {"scale_by": 0.5}},
        "3": {"class_type": "LTXVPreprocess", "inputs": {"img_compression": 33}},
    }
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(wf), encoding="utf-8")
    import modules.video_backends.comfyui_ltx_video as mod
    mod.WORKFLOWS_DIR = tmp_path
    return ltx.Backend({
        "unet_name": "u.gguf", "clip_name1": "c1.gguf", "clip_name2": "c2.safetensors",
        "vae_name": "v.safetensors", "workflow_file": "wf.json",
        "default_width": 1280, "default_height": 720,
    })


def test_ltx_honours_the_aspect_ratio(tmp_path):
    """It used to ignore `aspect_ratio` completely: a 9:16 request came back as
    the config's landscape 1280x720."""
    from modules.video_backends import comfyui_ltx_video as ltx
    b = _ltx_backend(tmp_path)
    wf = b._build_workflow(prompt="p", negative_prompt="n", input_image_name="i.png",
                           frame_count=97, fps=24, seed=1, steps_override=20,
                           cfg_override=4.5, width=768, height=1344)
    resize = next(v for v in wf.values() if v["class_type"] == "ResizeImageMaskNode")
    assert resize["inputs"]["resize_type.width"] == 768
    assert resize["inputs"]["resize_type.height"] == 1344


def test_ltx_does_not_secretly_halve_the_frame(tmp_path):
    """THE bug behind 'LTX looks soft and dissolving': the saved workflow scaled
    the frame by 0.5 AFTER the resize node, so an injected 768x1344 was rendered at
    384x672 and nobody said a word."""
    b = _ltx_backend(tmp_path)
    wf = b._build_workflow(prompt="p", negative_prompt="n", input_image_name="i.png",
                           frame_count=97, fps=24, seed=1, steps_override=20,
                           cfg_override=4.5, width=768, height=1344)
    scale = next(v for v in wf.values() if v["class_type"] == "ImageScaleBy")
    assert scale["inputs"]["scale_by"] == 1.0, "the frame must reach the sampler whole"
