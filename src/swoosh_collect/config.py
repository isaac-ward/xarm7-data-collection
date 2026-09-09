"""Load conf/collect.yaml. One config, no Hydra -- there is one entry point per job."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "conf" / "collect.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping at the top of {p}")
    data["_config_path"] = str(p)
    return data


def set_value(dotted: str, value: Any, path: str | Path | None = None) -> Path:
    """Rewrite ONE value in the YAML, in place, leaving every comment intact.

    Uses ruamel.yaml's round-trip mode, which is built for exactly this. The obvious
    `yaml.safe_dump` of the whole document is simpler and is what this used to do --
    but it silently deletes every comment, and this config is where most of the
    hard-won knowledge lives (why collision_sensitivity is 0, the evdev BTN trap, why
    the camera stagger cannot run in the control loop). Losing all of that to a camera
    rename is not an acceptable trade.

    A hand-rolled line-surgery version came before this one and got flow-style scalars
    and unquoted keys wrong; round-tripping with the library that already solves this
    is the right call.
    """
    from ruamel.yaml import YAML

    p = Path(path or DEFAULT_CONFIG_PATH)
    yml = YAML()                       # round-trip mode: preserves comments and order
    yml.preserve_quotes = True
    yml.width = 4096                   # don't re-wrap our long comment lines
    with p.open("r", encoding="utf-8") as fh:
        doc = yml.load(fh)

    parts = dotted.split(".")
    node = doc
    for key in parts[:-1]:
        if key not in node:
            raise KeyError(f"{dotted!r}: no such section {key!r} in {p}")
        node = node[key]
    node[parts[-1]] = value

    with p.open("w", encoding="utf-8") as fh:
        yml.dump(doc, fh)
    return p


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
