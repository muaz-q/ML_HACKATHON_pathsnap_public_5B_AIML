"""Raw-data loading and per-trip geometry.

Each trip folder <id>/ holds four headerless whitespace files:
  <id>.track  lon lat t          (GPS observations, ~1 Hz)
  <id>.nodes  lon lat            (row index = node id)
  <id>.arcs   from_node to_node  (row index = arc id, directed)
  <id>.route  arc_id             (reference route, chronological; may be absent)

Every trip carries its OWN local road network, so node/arc ids are only
meaningful inside that trip.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from pyproj import Transformer


@dataclass
class Trip:
    trip_id: str
    track_ll: np.ndarray          # (T,2) lon, lat
    times: np.ndarray             # (T,)
    nodes_ll: np.ndarray          # (N,2) lon, lat
    arcs: np.ndarray              # (A,2) from, to
    route: np.ndarray | None      # (R,) arc ids or None
    # filled by project()
    track_xy: np.ndarray = field(default=None, repr=False)
    nodes_xy: np.ndarray = field(default=None, repr=False)
    transformer: Transformer = field(default=None, repr=False)

    @property
    def arc_a(self):
        return self.nodes_xy[self.arcs[:, 0]]

    @property
    def arc_b(self):
        return self.nodes_xy[self.arcs[:, 1]]


def _read(path, dtype=float):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    return arr


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = lambda ext: os.path.join(trip_dir, f"{tid}.{ext}")
    track = _read(p("track"))
    nodes = _read(p("nodes"))
    arcs = _read(p("arcs"), dtype=np.int64)
    route = None
    if with_route and os.path.exists(p("route")):
        route = np.loadtxt(p("route"), dtype=np.int64, ndmin=1)
    # Data cleaning: drop non-finite rows and enforce time order.
    ok = np.isfinite(track).all(axis=1)
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    # drop exact duplicate timestamps (keep first)
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    track = track[keep]
    trip = Trip(tid, track[:, :2].copy(), track[:, 2].copy(), nodes, arcs, route)
    project(trip)
    return trip


def project(trip: Trip) -> None:
    """Project lon/lat into a local Azimuthal Equidistant frame (metres)
    centred on the trip's GPS centroid. Distances from the centre are exact and
    distortion is <0.1 % within ~50 km, which covers every trip; this works
    anywhere on Earth (the dataset spans all continents) without picking UTM zones."""
    lon0, lat0 = trip.track_ll.mean(axis=0)
    tr = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m",
        always_xy=True,
    )
    trip.transformer = tr
    trip.track_xy = np.column_stack(tr.transform(trip.track_ll[:, 0], trip.track_ll[:, 1]))
    trip.nodes_xy = np.column_stack(tr.transform(trip.nodes_ll[:, 0], trip.nodes_ll[:, 1]))


def list_trips(split_dir: str) -> list[str]:
    return sorted(
        os.path.join(split_dir, d)
        for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )


def point_segment(P, A, B):
    """Vectorised point-to-segment distance.
    Returns (distance, t in [0,1] along AB, projected point)."""
    AB = B - A
    L2 = (AB * AB).sum(-1)
    L2 = np.where(L2 == 0, 1e-12, L2)
    t = np.clip(((P - A) * AB).sum(-1) / L2, 0.0, 1.0)
    proj = A + t[..., None] * AB
    d = np.sqrt(((P - proj) ** 2).sum(-1))
    return d, t, proj
