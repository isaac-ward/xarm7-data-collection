"""Pre-motion operator banner. Lifted from sisl/manipulation-mono so the warning an
operator sees is identical across both codebases."""

from __future__ import annotations

import time

ORANGE_BOLD = "\033[1;38;5;208m"
GREEN_BOLD = "\033[1;32m"
RESET = "\033[0m"


def emit_motion_warning(
    robot_name: str = "SWOOSH", countdown: int = 3, active_banner: bool = True
) -> None:
    """Loud bold-orange countdown before any real-arm motion."""
    for sec in range(countdown, 0, -1):
        print(
            f"{ORANGE_BOLD}{robot_name} IS ABOUT TO MOVE, DO NOT BE STANDING "
            f"WITHIN ITS REACH. STARTING MOVEMENT IN {sec}{RESET}",
            flush=True,
        )
        time.sleep(1)
    if active_banner:
        print(f"{GREEN_BOLD}{robot_name} ACTIVE{RESET}", flush=True)
