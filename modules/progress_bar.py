"""
Claw Bot — Progress Bar Utilities

Renders Unicode progress bars + ETA for status_msg.edit() calls throughout the pipeline.

Usage:
    from modules.progress_bar import render_bar, render_full_status

    bar = render_bar(current=3, total=10, width=15)
    # ▓▓▓▓░░░░░░░░░░░  30%

    line = render_full_status(
        emoji="🎥", label="Generating", current=3, total=10,
        eta_sec=180, extra="Shot 3/10"
    )
    # 🎥 Generating · ▓▓▓▓░░░░░░░░░░░ 30% · Shot 3/10 · ETA ~3m
"""

import time
from typing import Optional

# Block-character bars look more polished than ASCII in Discord
FILL_CHAR = "▓"
EMPTY_CHAR = "░"

# Spinner frames for indeterminate progress
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def render_bar(current: int, total: int, width: int = 15) -> str:
    """Render a Unicode progress bar string with percentage.

    Examples:
        render_bar(3, 10)  -> "▓▓▓▓░░░░░░░░░░░ 30%"
        render_bar(10, 10) -> "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%"
        render_bar(0, 10)  -> "░░░░░░░░░░░░░░░ 0%"
    """
    if total <= 0:
        return f"{EMPTY_CHAR * width} 0%"

    current = max(0, min(current, total))
    pct = current / total
    filled = round(pct * width)
    empty = width - filled
    return f"{FILL_CHAR * filled}{EMPTY_CHAR * empty} {int(pct * 100)}%"


def render_indeterminate(tick: int, width: int = 15) -> str:
    """Animated bar for unknown duration (LLM gen, model load, etc).

    Returns a 'breathing' bar that moves a small block across the line.
    `tick` should be an int that increases on every call (e.g. seconds since start).
    """
    pos = tick % (width * 2)
    if pos >= width:
        pos = (width * 2) - pos - 1
    bar = [EMPTY_CHAR] * width
    bar[pos] = FILL_CHAR
    if pos > 0:
        bar[pos - 1] = FILL_CHAR
    if pos < width - 1:
        bar[pos + 1] = FILL_CHAR
    spinner = SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
    return f"{''.join(bar)} {spinner}"


def format_eta(seconds: float) -> str:
    """Convert seconds to a short human string."""
    if seconds <= 0:
        return "~done"
    if seconds < 60:
        return f"~{int(seconds)}s"
    if seconds < 3600:
        return f"~{int(seconds / 60)}m"
    hrs = int(seconds / 3600)
    mins = int((seconds % 3600) / 60)
    return f"~{hrs}h {mins}m"


def render_full_status(
    emoji: str,
    label: str,
    current: int,
    total: int,
    eta_sec: Optional[float] = None,
    extra: Optional[str] = None,
    width: int = 15,
) -> str:
    """Build a complete status line.

    Example output:
        🎥 Generating · ▓▓▓▓░░░░░░░ 30% · Shot 3/10 · ETA ~3m
    """
    bar = render_bar(current, total, width)
    parts = [f"{emoji} **{label}**", bar]
    if extra:
        parts.append(extra)
    if eta_sec is not None and eta_sec > 0:
        parts.append(f"ETA {format_eta(eta_sec)}")
    return " · ".join(parts)


class ProgressTracker:
    """Stateful tracker that auto-calculates ETA from rolling-window averages.

    Rolling-window beats simple cumulative average when shot durations vary —
    e.g. first shot pays model-load cost (~30s extra) but subsequent shots
    run in steady state. Linear average drags ETA up for the whole job; the
    rolling window converges to actual steady-state rate within a few samples.

    Usage:
        tracker = ProgressTracker(total=10, label="Generating", emoji="🎥")
        # ... before each unit of work:
        tracker.mark_step_start()
        # ... after each unit of work:
        tracker.mark_step_done()  # records duration, increments current
        line = tracker.render(extra="Shot 3 of 10")
    """

    WINDOW = 3  # rolling average of last N step durations

    def __init__(self, total: int, label: str = "Working", emoji: str = "⚙️"):
        self.total = total
        self.label = label
        self.emoji = emoji
        self.current = 0
        self.start_time = time.time()
        self._step_start: Optional[float] = None
        self._durations: list[float] = []

    def update(self, current: int):
        """Direct setter (legacy callers). Auto-records duration if step in flight."""
        if self._step_start is not None and current > self.current:
            self._durations.append(time.time() - self._step_start)
            self._durations = self._durations[-self.WINDOW:]
            self._step_start = None
        self.current = max(0, min(current, self.total))

    def mark_step_start(self):
        self._step_start = time.time()

    def mark_step_done(self):
        if self._step_start is not None:
            self._durations.append(time.time() - self._step_start)
            self._durations = self._durations[-self.WINDOW:]
            self._step_start = None
        self.current = min(self.current + 1, self.total)

    def _eta_sec(self) -> Optional[float]:
        remaining = self.total - self.current
        if remaining <= 0:
            return None
        if self._durations:
            avg = sum(self._durations) / len(self._durations)
            return avg * remaining
        # Fallback: cumulative average from start_time (early in job, no samples yet)
        if self.current > 0:
            return (time.time() - self.start_time) / self.current * remaining
        return None

    def render(self, extra: Optional[str] = None) -> str:
        return render_full_status(
            self.emoji, self.label, self.current, self.total,
            eta_sec=self._eta_sec(), extra=extra,
        )
