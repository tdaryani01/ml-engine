"""The OpenStreetMap converter: one-way streets, speeds from tags, only the part every place can reach."""
from __future__ import annotations

import numpy as np

from tools.osm_to_road_graph import road_graph_from_overpass


def _way(nodes, coords, **tags):
    return {"type": "way", "nodes": nodes, "geometry": [{"lat": a, "lon": b} for a, b in coords], "tags": {"highway": "residential", **tags}}


def test_a_small_extract_becomes_a_directed_graph_with_times() -> None:
    d = 0.001  # about 110 m of latitude
    data = {"elements": [
        _way([1, 2, 3], [(0, 0), (d, 0), (2 * d, 0)], maxspeed="36"),  # two-way, 36 km/h = 10 m/s
        _way([3, 4], [(2 * d, 0), (2 * d, d)], oneway="yes"),  # one-way out
        _way([4, 1], [(2 * d, d), (0, 0)], oneway="yes"),  # and back round: a loop through 1, 3 and 4
        _way([3, 9], [(2 * d, 0), (3 * d, 0)], oneway="yes"),  # a dead-end spur: reachable but nothing leads back
        {"type": "node", "id": 5},
    ]}
    g = road_graph_from_overpass(data)
    assert len(g["node_xy"]) == 4  # nodes 1, 2, 3, 4: the spur's end is dropped (not strongly connected)
    pairs = set(zip(g["edge_from"].tolist(), g["edge_to"].tolist()))
    assert len(pairs) == len(g["edge_from"])
    # 1 <-> 2 and 2 <-> 3 are two-way (4 roads); 3 -> 4 and 4 -> 1 are one-way (2 roads)
    assert len(g["edge_from"]) == 6
    t = g["edge_time"][np.argmin(np.abs(g["edge_time"] - 110.54 / 10.0))]
    assert abs(t - 11.054) < 0.2  # ~110.5 m at 10 m/s


def test_a_maxspeed_in_mph_and_a_missing_one_use_the_right_units() -> None:
    from tools.osm_to_road_graph import _kmh

    assert abs(_kmh({"highway": "primary", "maxspeed": "30 mph"}) - 48.28) < 0.01
    assert _kmh({"highway": "motorway"}) == 100 and _kmh({"highway": "unknown"}) == 30


def test_an_extract_with_no_drivable_ways_is_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="no drivable ways"):
        road_graph_from_overpass({"elements": [{"type": "node", "id": 1}]})
