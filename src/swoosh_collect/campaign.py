"""Campaigns and runs on disk.

    campaigns/                     <- gitignored
      <campaign-name>/
        campaign.json              <- name, created-at, notes
        index.html                 <- regenerated review page (swoosh-summarize)
        run_0001_<timestamp>/
          run.json                 <- start/stop, duration, how it ended
          raw/                     <- authoritative per-stream logs
          video/                   <- one mp4 per camera
          summary.mp4              <- 2x2 camera grid, time-synced
          inputs.png  kinematics.png

A "run" is one A-to-B episode. Runs are numbered per campaign in creation order, so
run_0007 is the seventh episode of that campaign regardless of when it happened.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import get, load_config

_SAFE = re.compile(r"[^a-z0-9._-]+")


def slugify(name: str) -> str:
    s = _SAFE.sub("-", name.strip().lower()).strip("-.")
    if not s:
        raise ValueError(f"campaign name {name!r} has no usable characters")
    return s


def repo_root() -> Path:
    """The dir holding pyproject.toml. Inside the container this is /workspace."""
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / "pyproject.toml").is_file()), Path.cwd()
    )


def host_view(path: Path | str) -> str:
    """Rewrite a container path to the equivalent HOST path, for DISPLAY ONLY.

    compose bind-mounts the host repo at /workspace and passes the host-side dir as
    SWOOSH_HOST_ROOT. "/workspace/campaigns" is meaningless to an operator reading the
    dashboard on the host and cannot be pasted into a file manager, so anything shown
    to a human goes through here. Never use the result to open a file -- the process
    is inside the container and only the container path resolves.
    """
    import os

    s = str(path)
    host_root = os.environ.get("SWOOSH_HOST_ROOT", "").rstrip("/")
    if not host_root:
        return s
    container_root = str(repo_root()).rstrip("/")
    if s == container_root:
        return host_root
    if s.startswith(container_root + "/"):
        return host_root + s[len(container_root):]
    return s


def campaigns_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    # Relative to the repo root, so it lands in the gitignored campaigns/ no matter
    # where the command is run from.
    return repo_root() / str(get(cfg, "recording.campaigns_dir", "campaigns"))


@dataclass
class Run:
    path: Path
    index: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.meta.get("duration_s", 0.0) or 0.0)

    @property
    def stopped_by(self) -> str:
        return str(self.meta.get("stopped_by", "unknown"))

    @property
    def status(self) -> str:
        """recording -> processing -> ready.

        Written by the collector, so the UI can show a run the instant A is pressed
        rather than only once its files exist. A run from an older/crashed session
        has no status field; infer one from what is on disk so it still lists.
        """
        st = self.meta.get("status")
        if st in {"recording", "processing", "ready"}:
            return str(st)
        if not self.meta:
            return "recording"
        return "ready" if (self.path / "summary.mp4").is_file() else "processing"

    @property
    def timestamp(self) -> str:
        """Human timestamp from the folder name.

        Current: recording_2026_09_07_07_05_57
        Legacy:  run_0001_20260908_230800
        """
        p = self.path.name
        m = re.match(r"recording_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})$", p)
        if m:
            y, mo, d, h, mi, sec = m.groups()
            return f"{y}-{mo}-{d} {h}:{mi}:{sec}"
        parts = p.split("_")
        if len(parts) >= 4 and len(parts[2]) == 8 and len(parts[3]) == 6:
            d, t = parts[2], parts[3]
            return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
        return ""

    @property
    def complete(self) -> bool:
        """A run that reached a B press (or a clean stop), rather than being cut off."""
        return self.meta.get("stopped_by") in {"b_button", "quit", "signal"}


class Campaign:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.meta_path = path / "campaign.json"

    # -- creation / lookup ---------------------------------------------------
    @classmethod
    def create(cls, name: str, cfg: dict | None = None, notes: str = "") -> "Campaign":
        root = campaigns_root(cfg)
        root.mkdir(parents=True, exist_ok=True)
        path = root / slugify(name)
        if path.exists():
            raise FileExistsError(f"campaign already exists: {path}")
        path.mkdir(parents=True)
        c = cls(path)
        c.meta_path.write_text(
            json.dumps(
                {
                    "name": name,
                    "slug": path.name,
                    "created_unix": time.time(),
                    "created_iso": datetime.now().isoformat(timespec="seconds"),
                    "notes": notes,
                },
                indent=2,
            )
        )
        return c

    @classmethod
    def open(cls, name: str, cfg: dict | None = None) -> "Campaign":
        path = campaigns_root(cfg) / slugify(name)
        if not path.is_dir():
            raise FileNotFoundError(
                f"no campaign {name!r} at {path}. Create it with: "
                f"swoosh-campaign new {name!r}"
            )
        return cls(path)

    @classmethod
    def list_all(cls, cfg: dict | None = None) -> list["Campaign"]:
        root = campaigns_root(cfg)
        if not root.is_dir():
            return []
        return [cls(p) for p in sorted(root.iterdir()) if (p / "campaign.json").is_file()]

    # -- contents ------------------------------------------------------------
    @property
    def meta(self) -> dict[str, Any]:
        try:
            return json.loads(self.meta_path.read_text())
        except Exception:
            return {"name": self.path.name, "slug": self.path.name}

    @property
    def name(self) -> str:
        return str(self.meta.get("name", self.path.name))

    def runs(self) -> list[Run]:
        out: list[Run] = []
        # `recording_<ts>` is the current naming; `run_NNNN_<ts>` is legacy. Sorting by
        # name is chronological for both, because the timestamp is zero-padded.
        paths = sorted([p for p in self.path.glob("recording_*") if p.is_dir()]
                       + [p for p in self.path.glob("run_*") if p.is_dir()])
        for i, p in enumerate(paths, 1):
            m = re.match(r"run_(\d+)_", p.name)
            meta: dict[str, Any] = {}
            rj = p / "run.json"
            if rj.is_file():
                try:
                    meta = json.loads(rj.read_text())
                except Exception:
                    meta = {}
            out.append(Run(path=p, index=int(m.group(1)) if m else i, meta=meta))
        return out

    def new_run_dir(self) -> Path:
        """Allocate the next run folder and mark it `recording` straight away.

        run.json is written NOW, not at the end, so the dashboard can list the run the
        moment A is pressed instead of waiting for it to finish.
        """
        # Named by TIMESTAMP, not a counter: the name says when it happened without a
        # lookup, and two processes cannot collide on the next number.
        path = self.path / datetime.now().strftime("recording_%Y_%m_%d_%H_%M_%S")
        if path.exists():                      # same second, vanishingly rare
            path = Path(str(path) + "_b")
        (path / "raw").mkdir(parents=True)
        (path / "video").mkdir(parents=True)
        (path / "run.json").write_text(json.dumps({
            "status": "recording",
            "started_unix": time.time(),
            "started_iso": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
        return path


def set_run_status(run_dir: Path, status: str, **extra: Any) -> None:
    """Merge a status (and any extra fields) into a run's run.json."""
    p = run_dir / "run.json"
    meta: dict[str, Any] = {}
    if p.is_file():
        try:
            meta = json.loads(p.read_text())
        except Exception:
            meta = {}
    meta["status"] = status
    meta.update(extra)
    p.write_text(json.dumps(meta, indent=2))


# -- CLI ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create and inspect collection campaigns.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="create a new campaign folder under campaigns/")
    p_new.add_argument("name")
    p_new.add_argument("--notes", default="", help="free text stored in campaign.json")

    sub.add_parser("list", help="list all campaigns")

    p_show = sub.add_parser("show", help="list the runs in one campaign")
    p_show.add_argument("name")

    args = ap.parse_args()
    cfg = load_config()

    if args.cmd == "new":
        c = Campaign.create(args.name, cfg, notes=args.notes)
        print(f"created campaign {c.name!r}")
        print(f"  {c.path}")
        print(f"\nCollect into it with:\n  swoosh-collect --campaign {c.path.name}")
        return 0

    if args.cmd == "list":
        cs = Campaign.list_all(cfg)
        if not cs:
            print(f"no campaigns yet under {campaigns_root(cfg)}")
            print('create one with:  swoosh-campaign new "my-campaign"')
            return 0
        print(f"{'campaign':32s} {'runs':>5} {'complete':>9}  created")
        for c in cs:
            rs = c.runs()
            print(
                f"{c.path.name:32s} {len(rs):5d} {sum(r.complete for r in rs):9d}"
                f"  {c.meta.get('created_iso', '?')}"
            )
        return 0

    if args.cmd == "show":
        c = Campaign.open(args.name, cfg)
        rs = c.runs()
        print(f"campaign {c.name!r}  ({len(rs)} runs, {sum(r.complete for r in rs)} complete)")
        print(f"  {c.path}\n")
        if not rs:
            print("  no runs yet")
            return 0
        print(f"  {'run':6s} {'dur (s)':>8} {'status':11s} {'stopped_by':12s} when")
        for r in rs:
            print(f"  {r.index:04d}   {r.duration_s:8.1f} {r.status:11s} "
                  f"{r.stopped_by:12s} {r.timestamp}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
