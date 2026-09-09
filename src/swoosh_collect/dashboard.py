"""Live `rich` panel for the collection session.

`rich.Live` is used rather than `textual` on purpose: Live has no event loop of its
own, so the 100 Hz servo loop keeps full control of timing and simply hands over a
renderable ~10 times a second. A framework that owned the loop would either throttle
the arm or need the control loop moved onto a worker thread, for no benefit.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _bar(value: float, width: int = 11, lo: float = -1.0, hi: float = 1.0) -> Text:
    """A centred bar for a stick axis, or a left-anchored one for a trigger."""
    span = hi - lo
    frac = 0.0 if span == 0 else (float(value) - lo) / span
    frac = max(0.0, min(1.0, frac))
    cells = ["─"] * width
    if lo < 0:
        cells[width // 2] = "┼"
    idx = min(width - 1, int(frac * (width - 1)))
    out = Text()
    for i, ch in enumerate(cells):
        out.append("█" if i == idx else ch,
                   style="bold cyan" if i == idx else "dim")
    return out


class Dashboard:
    def __init__(self, campaign_name: str, campaign_path: str) -> None:
        self.campaign_name = campaign_name
        self.campaign_path = campaign_path
        self.live: Live | None = None
        self._last = 0.0
        self.runs_completed = 0
        self.recording = False
        self.run_index: int | None = None
        self.run_started: float | None = None
        self.message = ""
        self.message_style = "dim"

    def __enter__(self) -> "Dashboard":
        self.live = Live(self._render({}, {}, []), refresh_per_second=8,
                         screen=False, transient=False)
        self.live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.live is not None:
            self.live.__exit__(*exc)

    def notify(self, text: str, style: str = "yellow") -> None:
        self.message = text
        self.message_style = style

    def maybe_update(self, pad: Any, arm_state: Any, cams: list[dict],
                     min_interval: float) -> None:
        now = time.monotonic()
        if now - self._last < min_interval or self.live is None:
            return
        self._last = now
        self.live.update(self._render(pad, arm_state, cams))

    def _render(self, pad: Any, st: Any, cams: list[dict]) -> Panel:
        head = Table.grid(padding=(0, 2))
        head.add_column(style="bold")
        head.add_column()
        head.add_row("campaign", f"[bold cyan]{self.campaign_name}[/]")
        head.add_row("runs done", f"[bold]{self.runs_completed}[/]  (A start -> B stop)")
        if self.recording and self.run_started is not None:
            head.add_row(
                "status",
                f"[bold white on red] REC [/] run {self.run_index:04d}   "
                f"{time.monotonic() - self.run_started:6.1f}s",
            )
        else:
            head.add_row("status", "[dim]idle -- press [bold]A[/bold] to start a run[/]")

        ctrl = Table(box=None, pad_edge=False, show_header=False)
        ctrl.add_column(width=9)
        ctrl.add_column(width=13)
        ctrl.add_column(justify="right", width=7)
        if pad:
            for label, key, lo, hi in (
                ("fwd/back", "move_x", -1.0, 1.0),
                ("left/right", "move_y", -1.0, 1.0),
                ("up/down", "height", -1.0, 1.0),
                ("yaw", "yaw", -1.0, 1.0),
                ("gripper", "gripper", 0.0, 1.0),
            ):
                v = float(getattr(pad, key, 0.0))
                ctrl.add_row(label, _bar(v, lo=lo, hi=hi), f"{v:+.2f}")
        else:
            ctrl.add_row("[red]no pad[/]", "", "")

        arm = Table(box=None, pad_edge=False, show_header=False)
        arm.add_column(width=9)
        arm.add_column()
        if st:
            wp = getattr(st, "pose_world_xyz", []) or []
            if len(wp) == 3:
                arm.add_row("world mm", f"x{wp[0]:7.1f}  y{wp[1]:7.1f}  z{wp[2]:7.1f}")
            g = getattr(st, "gripper_pos", float("nan"))
            arm.add_row("gripper", "n/a" if g != g else f"{g:.0f}")
            ec = int(getattr(st, "error_code", 0) or 0)
            arm.add_row(
                "error",
                "[green]none[/]" if ec == 0 else f"[bold red]code {ec}[/] -- run swoosh-sanity",
            )
        else:
            arm.add_row("[dim]waiting[/]", "")

        cam = Table(box=None, pad_edge=False, show_header=False)
        cam.add_column(width=17)
        cam.add_column(justify="right", width=8)
        cam.add_column()
        if cams:
            for c in cams:
                fps = c.get("fps", 0.0)
                style = "green" if fps >= 20 else ("yellow" if fps > 0 else "red")
                cam.add_row(
                    c["label"],
                    f"[{style}]{fps:5.1f} fps[/]",
                    f"[red]{c['error']}[/]" if c.get("error") else f"[dim]{c['frames']} frames[/]",
                )
        else:
            cam.add_row("[dim]no cameras[/]", "", "")

        body = Group(
            head, Text(""),
            Panel(ctrl, title="controller", border_style="dim", padding=(0, 1)),
            Panel(arm, title="arm", border_style="dim", padding=(0, 1)),
            Panel(cam, title="cameras", border_style="dim", padding=(0, 1)),
            Text(self.message, style=self.message_style),
            Text("A start   B stop   Y re-home   Start quit", style="dim"),
        )
        return Panel(body, title="swoosh collect", border_style="cyan")
