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
    # the spur's end (9) is dropped (not strongly connected); node 2 only bends the road 1 <-> 3 and node 4 only bends the one-way 3 -> 1,
    # so what is left is 1 and 3 with a two-way road between them (~221 m at 10 m/s each way); the one-way loop 3 -> 4 -> 1 is slower than
    # the direct road, and between two intersections only the quickest road is kept
    assert len(g["node_xy"]) == 2 and len(g["edge_from"]) == 2
    assert np.all(np.abs(g["edge_time"] - 22.1) < 0.6)


def test_a_maxspeed_in_mph_and_a_missing_one_use_the_right_units() -> None:
    from tools.osm_to_road_graph import _kmh

    assert abs(_kmh({"highway": "primary", "maxspeed": "30 mph"}) - 48.28) < 0.01
    assert _kmh({"highway": "motorway"}) == 100 and _kmh({"highway": "unknown"}) == 30


def test_an_extract_with_no_drivable_ways_is_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="no drivable ways"):
        road_graph_from_overpass({"elements": [{"type": "node", "id": 1}]})


def test_bends_on_each_arm_of_a_junction_are_merged_and_no_travel_time_is_lost() -> None:
    d = 0.001
    arms = {"east": [(0, 0), (d, 0), (2 * d, 0), (3 * d, 0)], "north": [(0, 0), (0, d), (0, 2 * d), (0, 3 * d)], "west": [(0, 0), (-d, 0), (-2 * d, 0), (-3 * d, 0)]}
    elements, nid = [], 1000
    for name, pts in arms.items():
        ids = [1] + [nid + i for i in range(1, len(pts))]  # every arm starts at the junction (node 1)
        nid += 100
        elements.append(_way(ids, pts, maxspeed="36"))
    g = road_graph_from_overpass({"elements": elements})
    assert len(g["node_xy"]) == 4 and len(g["edge_from"]) == 6  # the junction and the three arm ends; two roads per arm
    seg = d * 111_320.0 * np.cos(np.radians(0.0))
    expected = 3 * 3 * 2 * (d * 110_540.0 / 10.0)  # 3 arms x 3 segments x 2 directions, ~110.5 m at 10 m/s (east-west is a little shorter)
    assert abs(g["edge_time"].sum() - expected) / expected < 0.05 and seg > 0  # contracting adds the times of the merged roads
