"""Turn the reference .route (a sequence of arcs, NOT one row per GPS point)
into a per-observation reference arc.

The .route file says *which* arcs were driven and in *what order*, but not
*when*. We recover the timing with a monotonic alignment (dynamic programming,
like DTW): GPS point t is assigned route position j(t) such that
j(0) <= j(1) <= ... (the vehicle never goes backwards along its own route)
and the summed point-to-arc distance is minimal.

  D[t, j] = cost[t, j] + min_{j' <= j} D[t-1, j']

Route positions may be skipped (very short arcs that no GPS fix fell on).
"""
import numpy as np

from .data import Trip, point_segment


def align_route(trip: Trip):
    """Return (per-fix reference arc, per-fix route position, fix-to-arc distance in m)."""
    route = trip.route
    T, R = len(trip.track_xy), len(route)
    A = trip.arc_a[route]  # (R,2)
    B = trip.arc_b[route]
    prev = None
    back = np.empty((T, R), dtype=np.int32)  # argmin index of prefix-min
    for t in range(T):
        P = np.broadcast_to(trip.track_xy[t], (R, 2))
        c, _, _ = point_segment(P, A, B)
        if prev is None:
            D = c
            back[t] = np.arange(R)
        else:
            # prefix minimum of prev with arg
            pm = np.minimum.accumulate(prev)
            # argmin of prefix
            idx = np.arange(R)
            is_new = np.r_[True, prev[1:] < pm[:-1]]
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            D = c + pm
            back[t] = arg
        prev = D
    # backtrack: back[t, j] is the best route position for t-1 given j at t
    pos = np.empty(T, dtype=np.int64)
    j = int(np.argmin(prev))
    for t in range(T - 1, -1, -1):
        pos[t] = j
        if t > 0:
            j = int(back[t, j])
    labels = route[pos]
    d_lab, _, _ = point_segment(trip.track_xy, trip.arc_a[labels], trip.arc_b[labels])
    return labels, pos, d_lab
