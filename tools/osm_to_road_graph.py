"""Turn an OpenStreetMap extract into the road network the navigate family reads (an ``.npz``).

Input: the JSON an Overpass query returns with ``out geom`` for the drivable ways in an area, for example

    [out:json][timeout:90];
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street)$"]({south},{west},{north},{east});
    out geom;

Output: ``node_xy`` (metres, a local projection around the area's centre), directed ``edge_from``/``edge_to`` and ``edge_time`` (seconds,
from each road's length and its ``maxspeed`` or a default for its kind), ``node_lonlat`` for reference. One-way streets are one-way;
only the largest strongly connected part is kept, so every kept place can be reached from every other. Points where a road only bends are
merged, so a road runs from intersection to intersection and every step of a route is a real choice (or a real continuation).

    python tools/osm_to_road_graph.py extract.json road_graph.npz
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_KMH = {"motorway": 100, "motorway_link": 60, "trunk": 80, "trunk_link": 50, "primary": 60, "primary_link": 40, "secondary": 50,
                "secondary_link": 40, "tertiary": 40, "tertiary_link": 30, "unclassified": 35, "residential": 30, "living_street": 10}


def _kmh(tags: dict[str, Any]) -> float:
    raw = str(tags.get("maxspeed", "")).strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph)?$", raw)
    if m:
        return float(m.group(1)) * (1.609344 if m.group(2) else 1.0)
    return float(_DEFAULT_KMH.get(str(tags.get("highway", "")), 30))


def _largest_scc(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Boolean mask of the nodes in the largest strongly connected component (iterative Tarjan)."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(src.tolist(), dst.tolist()):
        adj[a].append(b)
    index, low, on, comp = [-1] * n, [0] * n, [False] * n, [-1] * n
    stack: list[int] = []
    counter = ncomp = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on[v] = True
            if i < len(adj[v]):
                work.append((v, i + 1))
                w = adj[v][i]
                if index[w] == -1:
                    work.append((w, 0))
                elif on[w]:
                    low[v] = min(low[v], index[w])
            else:
                if low[v] == index[v]:
                    while True:
                        w = stack.pop()
                        on[w] = False
                        comp[w] = ncomp
                        if w == v:
                            break
                    ncomp += 1
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[v])
    sizes = np.bincount(np.asarray(comp), minlength=ncomp)
    return np.asarray(comp) == int(np.argmax(sizes))


def _contract_pass_through(n: int, src: np.ndarray, dst: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge the points where a road merely bends: a node with one road in and one out (one-way), or the same two neighbours both
    ways (two-way), carries no choice, so the roads either side of it become one road with their times added.

    Returns the kept nodes (indices into the old ones) and the new edges (in the new indices)."""
    out: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    inc: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b, w in zip(src.tolist(), dst.tolist(), time.tolist()):
        if a != b and (b not in out[a] or w < out[a][b]):
            out[a][b] = w
            inc[b].add(a)
    alive = set(range(n))
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            o, i = set(out[v]), inc[v]
            if len(o) == 1 and len(i) == 1:
                (a,), (b,) = i, o
                if a != b:
                    w = out[a][v] + out[v][b]
                    if b not in out[a] or w < out[a][b]:
                        out[a][b] = w
                    inc[b].add(a)
                    _drop(v, out, inc)
                    alive.discard(v)
                    changed = True
            elif len(o) == 2 and o == i:
                a, b = sorted(o)
                for x, y in ((a, b), (b, a)):
                    w = out[x][v] + out[v][y]
                    if y not in out[x] or w < out[x][y]:
                        out[x][y] = w
                    inc[y].add(x)
                _drop(v, out, inc)
                alive.discard(v)
                changed = True
    keep = np.array(sorted(alive), dtype=np.int64)
    new = {int(old): k for k, old in enumerate(keep)}
    e_from, e_to, e_t = [], [], []
    for a in keep.tolist():
        for b, w in out[a].items():
            e_from.append(new[a])
            e_to.append(new[b])
            e_t.append(w)
    return keep, np.array(e_from), np.array(e_to), np.array(e_t)


def _drop(v: int, out: dict[int, dict[int, float]], inc: dict[int, set[int]]) -> None:
    for b in out[v]:
        inc[b].discard(v)
    for a in inc[v]:
        out[a].pop(v, None)
    out[v], inc[v] = {}, set()


def road_graph_from_overpass(data: dict[str, Any]) -> dict[str, np.ndarray]:
    ids: dict[int, int] = {}
    lonlat: list[tuple[float, float]] = []
    edges: list[tuple[int, int, float, float]] = []  # from, to, metres, km/h
    pending: list[tuple[list[int], dict[str, Any]]] = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "highway" not in (el.get("tags") or {}) or not el.get("geometry") or not el.get("nodes"):
            continue
        local = []
        for nid, pt in zip(el["nodes"], el["geometry"]):
            if nid not in ids:
                ids[nid] = len(lonlat)
                lonlat.append((float(pt["lon"]), float(pt["lat"])))
            local.append(ids[nid])
        pending.append((local, el["tags"]))
    if not pending:
        raise ValueError("the extract has no drivable ways (expected Overpass JSON from a query with `out geom`)")
    ll = np.asarray(lonlat)
    lon0, lat0 = ll[:, 0].mean(), ll[:, 1].mean()
    xy = np.stack([(ll[:, 0] - lon0) * np.cos(np.radians(lat0)) * 111_320.0, (ll[:, 1] - lat0) * 110_540.0], axis=1)
    for local, tags in pending:
        kmh = _kmh(tags)
        one = str(tags.get("oneway", "no")).lower()
        if tags.get("junction") == "roundabout" and one == "no":
            one = "yes"
        for a, b in zip(local[:-1], local[1:]):
            if a == b:
                continue
            metres = float(np.linalg.norm(xy[a] - xy[b]))
            if one in ("yes", "true", "1"):
                edges.append((a, b, metres, kmh))
            elif one == "-1":
                edges.append((b, a, metres, kmh))
            else:
                edges.append((a, b, metres, kmh))
                edges.append((b, a, metres, kmh))
    src, dst = np.array([e[0] for e in edges]), np.array([e[1] for e in edges])
    metres, kmh = np.array([e[2] for e in edges]), np.array([e[3] for e in edges])
    keep = _largest_scc(len(xy), src, dst)
    remap = np.cumsum(keep) - 1
    ok = keep[src] & keep[dst] & (metres > 0)
    node_idx = np.where(keep)[0]
    times = metres[ok] / (kmh[ok] / 3.6)
    kept, e_from, e_to, e_time = _contract_pass_through(len(node_idx), remap[src[ok]], remap[dst[ok]], times)
    return {"node_xy": xy[keep][kept], "node_lonlat": ll[keep][kept], "edge_from": e_from, "edge_to": e_to, "edge_time": e_time}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    graph = road_graph_from_overpass(json.loads(Path(argv[1]).read_text(encoding="utf-8")))
    np.savez_compressed(argv[2], **graph)
    print(f"{argv[2]}: {len(graph['node_xy'])} intersections, {len(graph['edge_from'])} directed roads")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
