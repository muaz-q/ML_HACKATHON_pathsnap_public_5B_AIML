"""HMM + Viterbi route reconstruction.

Hidden state at time t : one candidate arc for GPS observation t (we keep the
                         top-K_HMM candidates by classifier probability), plus
                         the position of the GPS fix projected onto that arc.
Emission  log P(z_t | x_t = a) = w_e * log p_norm(a | t)
          where p_norm is the classifier probability renormalised over the
          candidates of observation t.
Transition log P(x_t = b | x_{t-1} = a) = -| d_route(a -> b) - d_gps | / beta
          (Newson & Krumm 2009): the distance driven along the road network
          between the two projected points should match the straight-line
          distance between the two GPS fixes. Shortest paths are computed with
          a bounded Dijkstra on the trip's directed road graph.
          No path within the search bound => "disconnected": constant
          log-penalty -jump_penalty, so decoding never dies.
GPS gaps   dt > gap_s seconds or d_gps > gap_m metres => the chain is broken:
           transitions become uniform (0) and decoding restarts from emissions.
Viterbi    delta_t(b) = e_t(b) + max_a [delta_{t-1}(a) + log A(a,b)]
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class HMMConfig:
    k_hmm: int = 6
    beta: float = 8.0
    w_emit: float = 1.0
    jump_penalty: float = 25.0
    gap_s: float = 60.0
    gap_m: float = 2000.0
    back_tol: float = 15.0


class RoadGraph:
    def __init__(self, arcs: np.ndarray, nodes_xy: np.ndarray):
        self.arcs = arcs
        self.n_nodes = len(nodes_xy)
        self.length = np.linalg.norm(nodes_xy[arcs[:, 1]] - nodes_xy[arcs[:, 0]], axis=1)
        order = np.argsort(arcs[:, 0], kind="stable")
        self.out_arc = order
        self.out_ptr = np.searchsorted(arcs[order, 0], np.arange(self.n_nodes + 1))
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def dijkstra(self, src: int, cutoff: float):
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1], hit[2]
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        arcs, L, oa, ptr = self.arcs, self.length, self.out_arc, self.out_ptr
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for k in range(ptr[u], ptr[u + 1]):
                a = oa[k]
                v = arcs[a, 1]
                nd = d + L[a]
                if nd < dist.get(v, np.inf) and nd <= cutoff:
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (cutoff, dist, pred)
        return dist, pred

    def path_arcs(self, a: int, b: int, cutoff: float = 3000.0):
        """Arcs strictly between arc a and arc b on the shortest path (or None)."""
        src, dst = self.arcs[a, 1], self.arcs[b, 0]
        if src == dst:
            return []
        dist, pred = self.dijkstra(src, cutoff)
        if dst not in dist:
            return None
        out, v = [], dst
        while v != src:
            e = pred[v]
            out.append(int(e))
            v = self.arcs[e, 0]
        return out[::-1]


def viterbi(cands: list[np.ndarray], probs: list[np.ndarray], offsets: list[np.ndarray],
            gps_xy: np.ndarray, times: np.ndarray, graph: RoadGraph, cfg: HMMConfig):
    T = len(cands)
    L = graph.length
    arcs = graph.arcs
    delta = cfg.w_emit * np.log(probs[0] + 1e-12)
    back = [np.zeros(len(cands[0]), dtype=np.int64)]
    n_break = n_jump = 0
    for t in range(1, T):
        ca, cb = cands[t - 1], cands[t]
        sa, sb = offsets[t - 1], offsets[t]
        emit = cfg.w_emit * np.log(probs[t] + 1e-12)
        gc = float(np.linalg.norm(gps_xy[t] - gps_xy[t - 1]))
        dt = float(times[t] - times[t - 1])
        if dt > cfg.gap_s or gc > cfg.gap_m:
            # chain break: uniform transition, restart from best previous state
            n_break += 1
            best = int(np.argmax(delta))
            back.append(np.full(len(cb), best, dtype=np.int64))
            delta = delta[best] + emit
            continue
        cutoff = max(300.0, 3.0 * gc + 150.0)
        beta = cfg.beta * max(1.0, np.sqrt(dt))
        logA = np.full((len(ca), len(cb)), -cfg.jump_penalty)
        for i, a in enumerate(ca):
            rem = L[a] - sa[i]
            dist, _ = graph.dijkstra(int(arcs[a, 1]), cutoff)
            for j, b in enumerate(cb):
                if a == b and sb[j] - sa[i] >= -cfg.back_tol:
                    d = abs(sb[j] - sa[i])
                else:
                    sp = dist.get(int(arcs[b, 0]))
                    if sp is None:
                        continue
                    d = rem + sp + sb[j]
                logA[i, j] = -abs(d - gc) / beta
        score = delta[:, None] + logA
        bp = np.argmax(score, axis=0)
        if np.all(logA <= -cfg.jump_penalty):
            n_jump += 1
        delta = score[bp, np.arange(len(cb))] + emit
        back.append(bp)
    # backtrack
    state = np.empty(T, dtype=np.int64)
    state[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        state[t - 1] = back[t][state[t]]
    path = np.array([cands[t][state[t]] for t in range(T)], dtype=np.int64)
    return path, {"n_breaks": n_break, "n_all_jump_steps": n_jump}


def stitch_route(point_arcs: np.ndarray, graph: RoadGraph) -> list[int]:
    """Collapse per-observation arcs into a continuous arc sequence, filling
    gaps between non-adjacent consecutive arcs with shortest-path arcs."""
    route: list[int] = []
    for a in point_arcs:
        a = int(a)
        if route and route[-1] == a:
            continue
        if route:
            prev = route[-1]
            if graph.arcs[prev, 1] != graph.arcs[a, 0]:
                fill = graph.path_arcs(prev, a)
                if fill:
                    route.extend(fill)
        route.append(a)
    return route
