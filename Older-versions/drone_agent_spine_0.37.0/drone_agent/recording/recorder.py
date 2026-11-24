"""
The flight recorder. One JSONL file per flight, one monotonic clock.

This exists before the perception and planner layers on purpose: if the
logging is not synchronised and complete on the first flight day, every flight
gets flown twice. Every record carries t_mono from the same source
(time.monotonic() in this process), so once vision runs on the same machine
its frame timestamps align with setpoints without a clock-offset estimate.

Record types
------------
  meta        once, first line: config, git rev, host, session id
  tick        every executor tick — the bulk of the file
  skill       skill lifecycle: start / result
  event       named events (handover, touchdown, estop, land_begin)
  note        free text
  summary     once, last line: counters and outcome

Why the requested and applied setpoints are separate fields
-----------------------------------------------------------
Their difference is the safety-intervention metric for the whole thesis.
Loading this file into pandas should need no parsing work:

    import pandas as pd, json
    rows = [json.loads(l) for l in open("flight.jsonl")]
    ticks = pd.json_normalize([r for r in rows if r["type"] == "tick"])
    ticks[ticks["arb.intervened"]].groupby("arb.actions").size()
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git_rev() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
            timeout=2).decode().strip()
    except Exception:
        return None


class FlightRecorder:

    def __init__(self, path: Optional[str] = None, log_dir: str = "logs",
                 session_label: str = "flight") -> None:
        self.session_id = uuid.uuid4().hex[:12]
        if path is None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = str(Path(log_dir) / f"{stamp}-{session_label}-"
                                       f"{self.session_id}.jsonl")
        self.path = path
        self._fh = open(self.path, "a", buffering=1)   # line buffered
        self._n = 0
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------

    def meta(self, config: Dict[str, Any], label: str = "",
             extra: Optional[Dict[str, Any]] = None) -> None:
        from .. import __version__
        self._write("meta", {
            "session_id": self.session_id,
            "drone_agent_version": __version__,
            "label": label,
            "t_wall_start": time.time(),
            "t_mono_start": self._t0,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_rev": _git_rev(),
            "pid": os.getpid(),
            "config": config,
            **(extra or {}),
        })

    def tick(self, world: Any, skill_name: str, skill_status: str,
             requested: Optional[Dict[str, Any]],
             applied: Optional[Dict[str, Any]],
             arbiter_actions: List[str], blocked: bool,
             arbiter_counters: Optional[Dict[str, int]] = None,
             extra: Optional[Dict[str, Any]] = None) -> None:
        rec = world.to_log()
        rec.update({
            "skill": skill_name,
            "skill_status": skill_status,
            "setpoint_requested": requested,
            "setpoint_applied": applied,
            "arb": {
                "actions": arbiter_actions,
                "blocked": blocked,
                "intervened": bool(arbiter_actions) or blocked,
            },
        })
        if arbiter_counters is not None:
            rec["arb"]["counters"] = arbiter_counters
        if extra:
            rec.update(extra)
        self._write("tick", rec)

    def skill_start(self, name: str, params: Dict[str, Any],
                    precondition: Dict[str, Any]) -> None:
        self._write("skill", {"phase": "start", "skill": name,
                              "params": params, "precondition": precondition})

    def skill_result(self, result: Any) -> None:
        payload = {"phase": "result"}
        payload.update(result.to_log())
        self._write("skill", payload)

    def event(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._write("event", {"kind": kind, **(payload or {})})

    def note(self, text: str) -> None:
        self._write("note", {"text": text})

    def summary(self, payload: Dict[str, Any]) -> None:
        self._write("summary", payload)

    # ------------------------------------------------------------------

    def _write(self, rec_type: str, payload: Dict[str, Any]) -> None:
        self._n += 1
        rec: Dict[str, Any] = {}
        rec.update(payload)
        # The recorder's own stamps are written LAST so a payload key can
        # never silently overwrite them. t_mono is always write time and
        # always increases; a tick's sample time is t_sample_mono.
        rec["type"] = rec_type
        rec["seq"] = self._n
        rec["t_mono"] = round(time.monotonic(), 4)
        rec["t_wall"] = round(time.time(), 4)
        try:
            self._fh.write(json.dumps(rec, default=_fallback) + "\n")
        except Exception as e:                      # never kill a flight
            try:
                self._fh.write(json.dumps(
                    {"type": "note", "seq": self._n,
                     "text": f"record serialisation failed: {e}"}) + "\n")
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "FlightRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _fallback(o: Any) -> Any:
    if hasattr(o, "value"):          # Enum
        return o.value
    if hasattr(o, "to_log"):
        return o.to_log()
    return str(o)
