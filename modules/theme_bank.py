"""
Claw Bot — Theme Bank

A curated list of daily story themes. Mixed cultures & character types
(humans, animals, mixed) — no longer Indian-only.
The LLM picks appropriate setting, names, and cultural context based on
each theme's keywords.
"""

from datetime import datetime


# ==============================================================================
# THEME BANK — mixed cultures, mixed character types
# ==============================================================================

THEMES = [
    # ── Universal good habits (culture-neutral) ──
    "sharing toys with a sibling",
    "saying thank you without being reminded",
    "learning to wait your turn",
    "cleaning up after playing",
    "telling the truth when you made a mistake",
    "helping someone who dropped their things",
    "being kind to a new classmate",
    "asking for help when something is hard",
    "listening when someone else is talking",
    "saying sorry and meaning it",
    "finishing what you started",
    "being patient while waiting",

    # ── Animal protagonists ──
    "a shy little mouse learning to speak up",
    "a baby elephant afraid of rain",
    "a curious fox who won't stop asking questions",
    "a rabbit who wants to share but feels scared to",
    "a small bird learning to fly despite fear",
    "a cat who doesn't want to share its favorite sunny spot",
    "a puppy learning to stay calm when excited",
    "a baby owl who's afraid of the dark forest",
    "a turtle who wants to run fast like the hares",
    "a squirrel who hides all the nuts and regrets it",

    # ── Japanese/Eastern themes ──
    "a child learning to bow respectfully to elders",
    "planting a small garden and caring for it daily",
    "the beauty of cherry blossoms and letting go",
    "learning to say 'itadakimasu' before meals",

    # ── Indian themes ──
    "helping grandmother with her flowers for puja",
    "learning to offer water to a thirsty stranger",
    "sharing festival sweets with neighbors",
    "respecting elders by touching their feet",

    # ── Western themes ──
    "raking leaves together as a family chore",
    "learning to make the bed each morning",
    "waiting for Thanksgiving dinner to start",
    "a birthday where giving feels better than receiving",

    # ── Fantasy / whimsy (culture-ambiguous) ──
    "a tiny dragon learning not to breathe fire indoors",
    "a unicorn afraid of thunderstorms",
    "a cloud-child who cries rainbow tears",
    "a robot who finally learns to laugh",
    "a fairy who forgets how to fly when embarrassed",
    "a little star learning to shine its brightest",
    "a sock puppet finding its missing pair",
    "a friendly ghost learning not to scare people",

    # ── Nature / seasonal ──
    "the first leaf of autumn learning to let go",
    "a caterpillar impatient to become a butterfly",
    "a tiny seed waiting to grow into a tree",
    "first snowflake of the winter",
    "a puddle watching the sky after rain",

    # ── Mixed friendships (cross-culture / cross-species) ──
    "a cat and a bird becoming unlikely friends",
    "a child from one country visiting a friend from another",
    "a grandfather and grandchild from different generations bonding",
    "a new kid in school making their first friend",
    "a human and robot learning what friendship means",
]


# ==============================================================================
# ACCESSORS
# ==============================================================================

def get_theme_of_the_day(date: datetime | None = None) -> str:
    """
    Deterministic theme for a given date. The same date always returns
    the same theme, but themes cycle through the bank as days pass.
    """
    if date is None:
        date = datetime.now()
    day_of_year = date.timetuple().tm_yday
    return THEMES[day_of_year % len(THEMES)]


def get_random_theme() -> str:
    """Pick any theme at random."""
    import random
    return random.choice(THEMES)


def get_theme_count() -> int:
    return len(THEMES)


def list_themes(prefix: str | None = None, limit: int | None = None) -> list[str]:
    """List all themes, optionally filtering by prefix."""
    items = THEMES
    if prefix:
        p = prefix.lower()
        items = [t for t in THEMES if p in t.lower()]
    if limit:
        items = items[:limit]
    return list(items)


if __name__ == "__main__":
    print(f"Theme bank size: {get_theme_count()}")
    print(f"Today's theme: {get_theme_of_the_day()}")