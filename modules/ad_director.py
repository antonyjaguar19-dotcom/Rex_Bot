"""
Claw Bot — Manual Mode "Ad Director"

Turn a one-line advert brief into the prompts each model actually wants — so the
operator never has to know that Z-Image likes a paragraph, SDXL likes tags, Wan
wants one action + one camera move, and LTX wants motion described over time.

Two jobs:
  • craft_ad(brief, image_backend, video_backend) -> {image_prompt, negative_prompt,
    motion_prompt} — the still prompt written in the IMAGE model's own style, plus a
    matching motion prompt in the VIDEO model's own style.
  • craft_action(brief, action) -> a clean edit instruction for an image-EDIT model
    (Qwen-Image-Edit) that turns the still into the "after the action" frame, used by
    the action-image-sequence -> video feature.

The LLM is the local Ollama model already used by prompt_assembler (routed to the
'prompt' role). No new dependency.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import requests

_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules import model_registry

log = logging.getLogger("claw_bot.ad_director")


# ==============================================================================
# PER-MODEL PROMPT STYLE — what each family of model actually wants
# ==============================================================================

# Image prompt style comes from the backend's own `prompt_style` in models.json
# ("paragraph" | "tags"); default paragraph.
_IMAGE_STYLE_GUIDE = {
    "paragraph": (
        "Write the image prompt as ONE flowing cinematic paragraph of 40-70 words. "
        "Natural sentences, no comma-tag lists. Lead with the SUBJECT/product, then "
        "setting, lighting, mood, lens/camera, and art direction. Concrete and visual."
    ),
    "tags": (
        "Write the image prompt as a COMMA-SEPARATED list of concise visual tags "
        "(20-35 tags). Order: subject/product, key attributes, setting, lighting, "
        "style, quality boosters. No full sentences."
    ),
}

# Video/motion style keyed by backend family (matched on the backend id).
# IMPORTANT: never name the model here — the LLM will echo "Wan"/"LTX" into the
# output as if it were the subject. Describe the STYLE only.
_VIDEO_STYLE_GUIDE = {
    "wan": (
        "Style: ONE main movement of the subject that is ALREADY in the scene, plus "
        "ONE camera move (push-in, pan, drift, or static). 25-45 words, plain concrete "
        "English. Do NOT restate static details."
    ),
    "ltx": (
        "Style: describe what happens OVER TIME in natural sentences (40-70 words) — the "
        "subject's motion as it unfolds, then a single camera move. Do NOT repeat static "
        "details already visible in the image; describe only motion and change."
    ),
    "generic": (
        "Style: ONE clear movement of the subject already in the scene + ONE camera "
        "move, 30-50 words, concrete English."
    ),
}

# Absolute rules for every motion prompt — the model-name/fps leak guard.
_MOTION_RULES = (
    "Animate ONLY what is actually in the scene. Keep the SAME subject — if the scene "
    "is a feather, the feather moves; never invent a person, character, or action the "
    "scene does not contain. Write no frame rate, no resolution, no model or software "
    "names, no camera-gear brands. Output ONE line of motion description only — no "
    "labels, no quotes, no preamble."
)


def image_prompt_style(image_backend_id: Optional[str]) -> str:
    if not image_backend_id or image_backend_id == "(active)":
        cfg = model_registry.get_active("image_backend") or {}
    else:
        cfg = model_registry.get_available("image_backend", image_backend_id) or {}
    return "tags" if cfg.get("prompt_style") == "tags" else "paragraph"


def video_style_family(video_backend_id: Optional[str]) -> str:
    bid = (video_backend_id or "").lower()
    if "ltx" in bid:
        return "ltx"
    if "wan" in bid:
        return "wan"
    if video_backend_id in (None, "", "(active)"):
        active = (model_registry.get_active("video_backend") or {}).get("_id", "")
        return video_style_family(active)
    return "generic"


# ==============================================================================
# LLM
# ==============================================================================

def _call_llm(system_prompt: str, user_prompt: str, *, temperature: float = 0.6,
              num_predict: int = 600) -> str:
    cfg = model_registry.get_for_role("prompt") or model_registry.get_active("llm_backend")
    model_name = (cfg.get("model_id") or cfg.get("model_name") or cfg.get("model")
                  or "qwen2.5:14b-instruct-q6_K")
    url = cfg.get("server_url", "http://127.0.0.1:11434") + "/api/generate"
    payload = {
        "model": model_name,
        "prompt": f"SYSTEM INSTRUCTIONS (STRICTLY FOLLOW):\n{system_prompt}\n\n"
                  f"USER BRIEF:\n{user_prompt}",
        "stream": False, "think": False,
        "options": {"temperature": temperature, "top_p": 0.9,
                    "num_ctx": 8192, "num_predict": num_predict},
    }
    log.info(f"AdDirector → {model_name} temp={temperature}")
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (tolerates fences/prose)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    t = t.replace("```json", "```").split("```")[1] if "```" in t else t
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in LLM reply: {text[:200]}")
    return json.loads(m.group(0))


def _extract_json_array(text: str) -> list:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    t = t.replace("```json", "```").split("```")[1] if "```" in t else t
    m = re.search(r"\[.*\]", t, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in reply: {text[:160]}")
    arr = json.loads(m.group(0))
    if not isinstance(arr, list):
        raise ValueError("not a list")
    return arr


def _clean_line(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return re.sub(r'^["\']|["\']$', "", s).strip()


# Model / software names and technical tokens that must never appear in a motion
# prompt (the LLM leaks them from the style guide otherwise).
_LEAK_WORDS = re.compile(
    r"\b(wan\s*2\.?2?|wan|ltx(?:-?2\.?3?)?|comfyui|stable\s*diffusion|sdxl|flux|"
    r"kontext|qwen)\b", re.IGNORECASE)
_TECH_TOKENS = re.compile(r"\b\d+\s*fps\b|\b\d{2,4}p\b|\b\d+x\d+\b", re.IGNORECASE)


def _scrub_motion(s: str) -> str:
    """Strip leaked model names / fps / resolution tokens from a motion prompt and
    tidy the punctuation they leave behind."""
    s = _LEAK_WORDS.sub("", s or "")
    s = _TECH_TOKENS.sub("", s)
    s = re.sub(r"\s*,\s*,", ",", s)
    s = re.sub(r"^[\s,;:.\-]+", "", s)
    s = re.sub(r"\s+([,.;])", r"\1", s)
    return _clean_line(s)


# ==============================================================================
# PUBLIC — craft an advert's prompts
# ==============================================================================

def craft_ad(brief: str, image_backend_id: Optional[str] = None,
             video_backend_id: Optional[str] = None,
             aspect_ratio: str = "16:9") -> dict:
    """Brief -> {image_prompt, negative_prompt, motion_prompt}, each in the chosen
    model's own style. Raises on empty brief or unparseable LLM reply."""
    brief = (brief or "").strip()
    if not brief:
        raise ValueError("Ad brief is empty — describe the advert first.")

    istyle = image_prompt_style(image_backend_id)
    vfam = video_style_family(video_backend_id)
    system = (
        "You are an advertising creative director and prompt engineer for AI image "
        "and video models. You turn a short product/advert brief into production "
        "prompts. The result must sell: appealing hero framing of the product/subject, "
        "clean commercial lighting, aspirational mood, no on-image text or logos "
        "unless the brief asks.\n\n"
        f"Target aspect ratio: {aspect_ratio}.\n"
        f"IMAGE PROMPT STYLE — {_IMAGE_STYLE_GUIDE[istyle]}\n"
        f"MOTION PROMPT STYLE — {_VIDEO_STYLE_GUIDE[vfam]} {_MOTION_RULES} "
        "The motion must animate the SAME subject as the image prompt.\n\n"
        "The NEGATIVE prompt lists what to avoid (artifacts, extra limbs, text, "
        "watermark, low quality, distortion) as short comma tags.\n\n"
        "Return ONLY a JSON object, no prose, with exactly these keys:\n"
        '{"image_prompt": "...", "negative_prompt": "...", "motion_prompt": "..."}'
    )
    raw = _call_llm(system, brief)
    try:
        data = _extract_json(raw)
    except Exception as e:
        # one retry with a firmer nudge
        raw = _call_llm(system + "\n\nReturn STRICT JSON only. No markdown.", brief,
                        temperature=0.4)
        data = _extract_json(raw)
    out = {
        "image_prompt": _clean_line(data.get("image_prompt", "")),
        "negative_prompt": _clean_line(data.get("negative_prompt", "")),
        "motion_prompt": _scrub_motion(data.get("motion_prompt", "")),
        "image_style": istyle,
        "video_family": vfam,
    }
    if not out["image_prompt"]:
        raise ValueError(f"LLM returned no image prompt (raw: {raw[:160]})")
    return out


def craft_motion(scene: str, video_backend_id: Optional[str] = None) -> str:
    """A motion prompt for a shot that already has a still. `scene` describes WHAT
    THE IMAGE SHOWS; the result animates that exact subject in the video model's
    style. Never invents a new subject, never leaks the model name or fps."""
    scene = (scene or "").strip()
    if not scene:
        raise ValueError("No scene to animate.")
    vfam = video_style_family(video_backend_id)
    system = (
        "You write a MOTION prompt for an AI video model. You are given the SCENE that "
        "the still image shows. Describe how the subject and elements in THAT scene "
        "move.\n" + _VIDEO_STYLE_GUIDE[vfam] + "\n" + _MOTION_RULES
    )
    user = f"SCENE (what the image shows):\n{scene}\n\nWrite the motion for this exact scene."
    raw = _call_llm(system, user, temperature=0.3, num_predict=200)
    return _scrub_motion(raw)


def craft_midframes(description: str = "", k: int = 1) -> list:
    """In-between edit instructions for a keyframe sequence that runs between TWO
    provided stills — image 1 = the FIRST frame, image 2 = the LAST frame (both
    handed to the edit model as references). Returns k one-line instructions, each
    telling the model to blend from the first toward the last at an evenly-spaced
    PROGRESS point. `description` is optional extra guidance on the motion."""
    description = (description or "").strip()
    k = max(1, min(int(k), 6))
    fractions = [f"{round((i + 1) / (k + 1) * 100)}%" for i in range(k)]
    system = (
        "You write edit instructions for an image-editing model (Qwen-Image-Edit). "
        "It is given TWO reference images: image 1 is the FIRST frame of a motion and "
        "image 2 is the LAST frame. Your job: for each PROGRESS point, write one "
        "imperative sentence that produces the IN-BETWEEN frame at that point — i.e. "
        "the subject/pose interpolated that far from image 1 toward image 2. Keep the "
        "same character, clothing, background, lighting, style and camera as the two "
        "references; only the pose/position advances. "
        + (f"Motion note: {description}. " if description else "")
        + "Return ONLY a JSON array of k strings, in order, one per progress point."
    )
    user = ("Image 1 = first frame, image 2 = last frame.\n"
            f"PROGRESS POINTS (fraction from first toward last): {', '.join(fractions)}")
    raw = _call_llm(system, user, temperature=0.4, num_predict=400)
    try:
        arr = _extract_json_array(raw)
    except Exception:
        arr = []
    out = [_clean_line(s) for s in arr if _clean_line(s)]
    # pad/truncate to exactly k with a deterministic fallback instruction
    while len(out) < k:
        f = fractions[len(out)]
        note = f" ({description})" if description else ""
        out.append(f"Interpolate the pose {f} of the way from the first image toward "
                   f"the second image{note}; keep the same character, scene, lighting "
                   f"and camera.")
    return out[:k]


def craft_action(action: str, subject_hint: str = "") -> str:
    """Turn a plain action ('she raises her arm and waves') into a clean EDIT
    instruction for an image-edit model (Qwen-Image-Edit): keep identity/scene,
    change only the pose/action to the END of the motion. Returns one line."""
    action = (action or "").strip()
    if not action:
        raise ValueError("Action is empty.")
    system = (
        "You write single-line edit instructions for an image-editing model "
        "(Qwen-Image-Edit). The user gives an ACTION. Rewrite it as an instruction "
        "that transforms the given image to show the FINAL pose/state at the END of "
        "that action, while KEEPING the same character, clothing, background, style "
        "and camera. Change only what the action changes. One imperative sentence, "
        "no quotes, no preamble."
        + (f" Subject: {subject_hint}." if subject_hint else "")
    )
    return _clean_line(_call_llm(system, action, temperature=0.4, num_predict=150))
