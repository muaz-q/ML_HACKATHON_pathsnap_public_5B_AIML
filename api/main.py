"""PathSnap API (runs on Laptop 1).

  uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables
  PATHSNAP_EVAL_DIR   folder holding the evaluation trips (one sub-folder per trip
                      with .track/.nodes/.arcs/.route). Default: data/student_test
  PATHSNAP_MODEL      trained classifier (default artifacts/model.joblib)
  PATHSNAP_HMM        decoder config JSON (default artifacts/hmm_config.json)
  PATHSNAP_API_KEY    optional shared key. If set, /trips and /predict require the
                      header  X-API-Key: <key>  (the UI sends it). /health stays open.

Two-laptop setup: run start_backend.bat on Laptop 1; open the UI on Laptop 2 and
enter http://<laptop-1-ip>:8000 (and the key, if one is set).

The reference .route is read ONLY inside /predict to score the prediction after
it has been made; it is never passed to the model and never returned to the UI.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import hmac

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.hmm import HMMConfig  # noqa: E402
from pathsnap.pipeline import evaluate_prediction, predict_trip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PathSnap] %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "student_test"))
MODEL_PATH = os.environ.get("PATHSNAP_MODEL", os.path.join(ROOT, "artifacts", "model.joblib"))
HMM_PATH = os.environ.get("PATHSNAP_HMM", os.path.join(ROOT, "artifacts", "hmm_config.json"))

MODEL = joblib.load(MODEL_PATH)
HMM_CFG = HMMConfig(**json.load(open(HMM_PATH))) if os.path.exists(HMM_PATH) else HMMConfig()
TRIPS = {os.path.basename(d): d for d in list_trips(EVAL_DIR)}
log.info("model=%s  eval_dir=%s  trips=%s  hmm=%s", MODEL_PATH, EVAL_DIR, list(TRIPS), HMM_CFG)

app = FastAPI(title="PathSnap map-matching API", version="1.0")
API_KEY = os.environ.get("PATHSNAP_API_KEY", "").strip()
log.info("api key %s", "REQUIRED (X-API-Key header)" if API_KEY else "not set (open access)")


@app.middleware("http")
async def check_key(request: Request, call_next):
    """Reject /trips and /predict without the shared key (only when PATHSNAP_API_KEY is set)."""
    if API_KEY and request.method != "OPTIONS" and request.url.path in ("/trips", "/predict"):
        sent = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(sent, API_KEY):
            log.warning("REJECTED %s from %s (missing or wrong API key)", request.url.path,
                        request.client.host if request.client else "?")
            return JSONResponse({"detail": "missing or wrong API key"}, status_code=401)
    return await call_next(request)


# added last so it runs first: CORS headers are also present on 401 replies
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _ll(trip, node):
    lon, lat = trip.nodes_ll[node]
    return [round(float(lat), 6), round(float(lon), 6)]


def _route_polyline(trip, arcs):
    """Chain consecutive arcs into polylines (new polyline when not connected)."""
    lines, cur = [], []
    for a in arcs:
        u, v = trip.arcs[a]
        if cur and trip.arcs[prev, 1] == u:
            cur.append(_ll(trip, v))
        else:
            if cur:
                lines.append(cur)
            cur = [_ll(trip, u), _ll(trip, v)]
        prev = a
    if cur:
        lines.append(cur)
    return lines


def _network(trip, limit=8000):
    lo, hi = trip.track_xy.min(0) - 250, trip.track_xy.max(0) + 250
    A, B = trip.arc_a, trip.arc_b
    near = np.nonzero(np.all(np.minimum(A, B) <= hi, 1) & np.all(np.maximum(A, B) >= lo, 1))[0]
    seen, out = set(), []
    for a in near:
        u, v = map(int, trip.arcs[a])
        k = (min(u, v), max(u, v))
        if k in seen:
            continue
        seen.add(k)
        out.append([_ll(trip, u), _ll(trip, v)])
        if len(out) >= limit:
            break
    return out


@app.get("/health")
def health():
    return {"status": "ok", "trips": len(TRIPS), "model": os.path.basename(MODEL_PATH), "key_required": bool(API_KEY)}


@app.get("/trips")
def trips():
    out = []
    for tid, d in TRIPS.items():
        with open(os.path.join(d, f"{tid}.track")) as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": tid, "n_obs": n})
    return {"eval_dir": os.path.basename(os.path.normpath(EVAL_DIR)), "trips": out}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    if req.trip_id not in TRIPS:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.perf_counter()
    log.info("REQUEST /predict trip=%s from %s", req.trip_id, request.client.host if request.client else "?")
    trip = load_trip(TRIPS[req.trip_id], with_route=False)  # prediction never sees the reference
    pred = predict_trip(trip, MODEL, HMM_CFG)
    t_pred = time.perf_counter() - t0

    # server-side scoring with the reference route (read only now)
    ref_path = os.path.join(TRIPS[req.trip_id], f"{req.trip_id}.route")
    metrics = None
    if os.path.exists(ref_path):
        trip.route = np.loadtxt(ref_path, dtype=np.int64, ndmin=1)
        metrics = evaluate_prediction(trip, pred)
        trip.route = None

    step = max(1, len(trip.track_ll) // 2500)
    resp = {
        "trip_id": req.trip_id,
        "n_obs": int(len(trip.track_ll)),
        "gps": [[round(float(lat), 6), round(float(lon), 6)] for lon, lat in trip.track_ll[::step]],
        "predicted_route": _route_polyline(trip, pred["route"]),
        "classifier_only_route": _route_polyline(trip, pred["classifier_arcs"]),
        "predicted_arc_per_obs": pred["hmm_arcs"].tolist(),
        "network": _network(trip),
        "metrics": metrics,
        "timings": {**{k: round(v, 3) for k, v in pred["timings"].items()},
                    "prediction_total_s": round(t_pred, 3),
                    "request_total_s": round(time.perf_counter() - t0, 3)},
        "hmm_info": pred["hmm_info"],
        "hmm_config": HMM_CFG.__dict__,
    }
    if metrics:
        log.info("RESULT trip=%s obs=%d | classifier top1=%.4f | HMM top1=%.4f | continuity=%.4f | "
                 "route F1=%.4f | length err=%.4f | %.2fs", req.trip_id, metrics["n_obs"],
                 metrics["classifier_top1"], metrics["hmm_top1"], metrics["hmm_continuity"],
                 metrics["hmm_arc_f1"], metrics["hmm_length_error"], t_pred)
    return resp
