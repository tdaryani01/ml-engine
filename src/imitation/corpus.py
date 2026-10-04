"""The corpus file: JSON lines, one decision episode per line (the shape TM's ``GET /episodes`` returns)."""
from __future__ import annotations

import json
from pathlib import Path

from tm_brain_contracts import DecisionEpisode


class CorpusError(ValueError):
    pass


def load_corpus(path: Path | str) -> list[DecisionEpisode]:
    p = Path(str(path))
    if not p.is_file():
        raise CorpusError(f"corpus file not found: {p}")
    episodes: list[DecisionEpisode] = []
    seen: set[str] = set()
    with p.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ep = DecisionEpisode.from_public(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise CorpusError(f"corpus line {n} is not an episode: {exc}") from exc
            if not ep.id or not ep.instance_id:
                raise CorpusError(f"corpus line {n} has no id / instance_id (tape)")
            if ep.id in seen:  # the snapshot is one row per episode; a repeated id is a defect upstream, not data
                continue
            seen.add(ep.id)
            episodes.append(ep)
    if not episodes:
        raise CorpusError("the corpus is empty")
    return episodes
