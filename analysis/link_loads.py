#!/usr/bin/env python3
"""External validation: do the roads MoveOD loads most also slow down most?

MoveOD synthesises commute trips and routes them on an OSM road graph, but
nothing in the pipeline checks *which* roads the synthetic trips end up on.
This script takes a finished run, re-routes its calibrated trips while counting
how many trips traverse each road segment, matches those segments to INRIX XD
segments, and correlates the synthetic AM load with the observed AM speed drop.

The comparison is non-circular for the Hamilton 2025-03-17 run because that run
was built with OSM default speeds ("Creating graph using OSM default speeds" in
its log), so no INRIX observation entered the routing.

Loads are bucketed by *traversal* hour by default: a trip that departs at 07:50
and is still driving at 08:10 contributes its later edges to the 08:00 bucket.
``--load-hours departure`` reproduces the router's own bucketing instead, which
files every edge of a trip under the hour it left.

Usage:
    python analysis/link_loads.py --output-root move_OD --state Tennessee \
        --county Hamilton --run-id 2025-03-17_2025-03-17 \
        --inrix data/inrix/Hamilton-County-INRIX.csv \
        --xd data/inrix/XD_Identification.csv
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

# Run either as `python analysis/link_loads.py` or `python -m analysis.link_loads`;
# only the latter puts the repo root (and so cli / generate) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from scipy import stats
from shapely.geometry import LineString

from analysis import station_counts
from analysis.figures_from_output import find_run_dir
from analysis.validate_external import code_sha
from cli import deserialize_graphs
from generate.generate_routing_df import edge_loads_to_frame, get_routed

logger = logging.getLogger(__name__)

# Columns _build_arrays_from_df(post_calibration=True) reads off the OD frame.
_TRIP_COLUMNS = [
    "origin_geoid", "destination_geoid", "departure_datetime",
    "origin_lat", "origin_lon", "destination_lat", "destination_lon",
]


# ── Run inputs ──────────────────────────────────────────────────────────────

def load_hourly_graphs(run_dir: Path) -> dict:
    """Load the run's hourly graphs exactly as cli.py does when resuming."""
    path = run_dir / "hourly_graphs.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing hourly graphs: {path}")
    with open(path, "r") as fh:
        graphs = deserialize_graphs(json.load(fh))
    logger.info("Loaded %d hourly graph keys from %s", len(graphs), path)
    return graphs


def load_calibrated_trips(run_dir: Path) -> tuple[pd.DataFrame, datetime.date]:
    """Read calibrated_move_od/<day>.csv and return it with the day it covers."""
    candidates = sorted((run_dir / "calibrated_move_od").glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No calibrated CSVs in {run_dir / 'calibrated_move_od'}")
    path = candidates[-1]
    trips = pd.read_csv(path, usecols=_TRIP_COLUMNS)
    day = pd.to_datetime(trips["departure_datetime"]).dt.date.mode().iat[0]
    logger.info("Read %d calibrated trips for %s from %s", len(trips), day, path)
    return trips, day


# ── INRIX inputs ────────────────────────────────────────────────────────────

def read_inrix_ratios(inrix_path: Path, day: datetime.date,
                      hours: Sequence[int]) -> pd.DataFrame:
    """Mean speed / reference_speed per XD segment, per requested clock hour.

    Rows without a usable reference speed carry no congestion information, so
    reference_speed <= 0 or NaN is dropped rather than treated as free flow.
    """
    columns = ["xd_id", "measurement_tstamp", "speed", "reference_speed"]
    df = pd.read_csv(inrix_path, engine="pyarrow", usecols=columns)
    logger.info("Read %d INRIX rows from %s", len(df), inrix_path)

    ts = pd.to_datetime(df["measurement_tstamp"])
    keep = ts.dt.normalize().eq(pd.Timestamp(day)) & ts.dt.hour.isin(list(hours))
    df = df[keep].assign(hour=ts[keep].dt.hour)
    df = df[df["reference_speed"].gt(0) & df["speed"].notna()]
    df = df.assign(ratio=df["speed"] / df["reference_speed"])

    out = (
        df.groupby(["xd_id", "hour"])["ratio"]
        .agg(speed_ratio="mean", n_minutes="size")
        .reset_index()
    )
    out["congestion"] = 1.0 - out["speed_ratio"]
    logger.info("INRIX: %d segment-hours on %s", len(out), day)
    return out


def read_xd_segments(xd_path: Path) -> gpd.GeoDataFrame:
    """XD identification table as start->end LineStrings with a true bearing.

    The file's own `bearing` column only holds N/S/E/W/O, far too coarse for a
    +/-35 degree test, so the geodesic bearing is recomputed from the endpoints.
    The first column carries a BOM, hence utf-8-sig.
    """
    df = pd.read_csv(xd_path, encoding="utf-8-sig")
    df = df[(df["start_latitude"] != df["end_latitude"])
            | (df["start_longitude"] != df["end_longitude"])]
    geometry = [
        LineString([(x1, y1), (x2, y2)])
        for x1, y1, x2, y2 in zip(df["start_longitude"], df["start_latitude"],
                                  df["end_longitude"], df["end_latitude"])
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf["bearing_deg"] = ox.bearing.calculate_bearing(
        df["start_latitude"].to_numpy(), df["start_longitude"].to_numpy(),
        df["end_latitude"].to_numpy(), df["end_longitude"].to_numpy(),
    )
    return gdf[["xd", "bearing_deg", "frc", "geometry"]]


# ── Matching ────────────────────────────────────────────────────────────────

def bearing_difference(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    """Smallest absolute angle in degrees between two compass bearings."""
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def add_edge_bearings(edges: gpd.GeoDataFrame, G) -> gpd.GeoDataFrame:
    """Compass bearing per loaded edge, from its OSM node coordinates."""
    nodes = G.nodes
    return edges.assign(bearing=ox.bearing.calculate_bearing(
        np.array([nodes[u]["y"] for u in edges["u"]]),
        np.array([nodes[u]["x"] for u in edges["u"]]),
        np.array([nodes[v]["y"] for v in edges["v"]]),
        np.array([nodes[v]["x"] for v in edges["v"]]),
    ))


def match_segments_to_edges(segments: gpd.GeoDataFrame, edges: gpd.GeoDataFrame,
                            buffer_m: float = 30.0, bearing_tol_deg: float = 35.0,
                            fallback_m: float = 60.0) -> pd.DataFrame:
    """Match XD segments to loaded OSM edges; one row per (segment, edge) pair.

    Both frames must already be projected to the same metric CRS. An edge
    matches when its midpoint falls inside the segment's buffer and its bearing
    is within +/- bearing_tol_deg of the segment's. Segments still unmatched
    fall back to the single nearest edge to their midpoint within fallback_m;
    that fallback ignores bearing, so it is labelled separately.
    """
    mids = edges.assign(geometry=edges.geometry.interpolate(0.5, normalized=True))
    buffers = segments[["xd", "bearing_deg", "geometry"]].copy().assign(
        geometry=segments.geometry.buffer(buffer_m)
    )

    hits = gpd.sjoin(mids, buffers, how="inner", predicate="within")
    if len(hits):
        hits = hits[bearing_difference(hits["bearing"], hits["bearing_deg"])
                    <= bearing_tol_deg]
    cols = ["xd", "u", "v", "load"] + (["highway"] if "highway" in edges else [])
    matched = hits[cols].copy()
    matched["match_kind"] = "buffer"

    missing = segments[~segments["xd"].isin(matched["xd"])]
    if len(missing) and fallback_m > 0:
        seg_mids = missing[["xd", "geometry"]].assign(
            geometry=missing.geometry.interpolate(0.5, normalized=True)
        )
        near = gpd.sjoin_nearest(
            seg_mids, edges[[c for c in cols if c != "xd"] + ["geometry"]],
            how="inner", max_distance=fallback_m,
        ).drop_duplicates(subset="xd")
        fallback = near[cols].copy()
        fallback["match_kind"] = "nearest"
        matched = pd.concat([matched, fallback], ignore_index=True)

    return matched.reset_index(drop=True)


def segment_loads(matches: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-edge matches to one synthetic load per XD segment.

    match_kind travels with the load: the nearest-edge fallback ignores bearing,
    so those segments are the weaker matches and have to stay separable.
    """
    loads = (
        matches.groupby("xd")["load"]
        .agg(load_mean="mean", load_max="max", n_edges="size")
        .reset_index()
    )
    kinds = matches.groupby("xd")["match_kind"].agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else None
    )
    return loads.merge(kinds.rename("match_kind").reset_index(), on="xd")


def segment_classes(matches: pd.DataFrame) -> pd.DataFrame:
    """Modal OSM highway class of the edges matched to each XD segment."""
    modes = matches.groupby("xd")["highway"].agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else None
    )
    return modes.rename("highway").reset_index()


# ── Statistics ──────────────────────────────────────────────────────────────

def spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, int]:
    """Spearman rho, its p-value and the number of usable pairs."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    n = int(ok.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    result = stats.spearmanr(xs[ok], ys[ok])
    return float(result.statistic), float(result.pvalue), n


def stratified_spearman(df: pd.DataFrame, load_col: str, congestion_col: str,
                        group_col: str) -> pd.DataFrame:
    """Spearman rho, p and n within each class of group_col."""
    rows = []
    for name, group in df.groupby(group_col, dropna=True):
        rho, p, n = spearman(group[load_col], group[congestion_col])
        rows.append({group_col: name, "rho": rho, "p": p, "n": n})
    return pd.DataFrame(rows, columns=[group_col, "rho", "p", "n"])


def within_class_rank_spearman(df: pd.DataFrame, load_col: str,
                               congestion_col: str,
                               group_col: str) -> tuple[float, float, int]:
    """Spearman on ranks taken *inside* each class, then pooled.

    Ranks are percentile ranks so classes of different sizes are comparable
    once pooled; a class with a single segment cannot be ranked and is dropped.
    This is the one number that answers "within a road class, does load track
    congestion?" across all classes at once.
    """
    usable = df.dropna(subset=[load_col, congestion_col, group_col])
    usable = usable[usable.groupby(group_col)[load_col].transform("size") > 1]
    if usable.empty:
        return float("nan"), float("nan"), 0
    ranked_load = usable.groupby(group_col)[load_col].rank(pct=True)
    ranked_cong = usable.groupby(group_col)[congestion_col].rank(pct=True)
    return spearman(ranked_load, ranked_cong)


def decile_means(load: Iterable[float], congestion: Iterable[float],
                 n_bins: int = 10) -> pd.DataFrame:
    """Mean congestion per synthetic-load decile.

    Loads are ranked before binning: most edges carry only a handful of trips,
    so binning on the raw values would collapse the low deciles into one.
    """
    df = pd.DataFrame({
        "load": np.asarray(load, dtype=float),
        "congestion": np.asarray(congestion, dtype=float),
    }).dropna()
    if df.empty:
        return pd.DataFrame(columns=["decile", "n", "load_mean", "congestion_mean"])

    df["decile"] = pd.qcut(df["load"].rank(method="first"), n_bins,
                           labels=False, duplicates="drop")
    return (
        df.groupby("decile")
        .agg(n=("load", "size"), load_mean=("load", "mean"),
             congestion_mean=("congestion", "mean"))
        .reset_index()
    )


# ── Pipeline ────────────────────────────────────────────────────────────────

def routed_edge_loads(run_dir: Path, load_hours: str = "traversal"
                      ) -> tuple[gpd.GeoDataFrame, datetime.date]:
    """Re-route the run's calibrated trips, counting trips per road edge.

    load_hours="traversal" files each edge under the hour the trip is actually
    on it; "departure" files a whole trip under the hour it left, which is the
    router's own bucketing and is kept so that variant can be reproduced.
    """
    hourly_graphs = load_hourly_graphs(run_dir)
    trips, day = load_calibrated_trips(run_dir)
    _, edge_loads = get_routed(
        od_df=trips, desired_date=day, hourly_graphs_arg=hourly_graphs,
        post_calibration=True, return_edge_loads=True,
        edge_load_hours=load_hours,
    )
    # Geometry, length and class are described from one graph; the hourly
    # graphs differ only in edge speeds, but say so if that ever stops holding.
    sizes = {G.number_of_edges() for G in hourly_graphs.values()}
    if len(sizes) > 1:
        logger.warning("Hourly graphs differ in edge count %s; edges are described "
                       "from the first graph only", sorted(sizes))
    G = next(iter(hourly_graphs.values()))
    frame = add_edge_bearings(edge_loads_to_frame(edge_loads, G), G)
    logger.info("Loaded %d edge-hours across %d hours", len(frame), len(edge_loads))
    return frame, day


def compare_hour(edges_all: gpd.GeoDataFrame, segments: gpd.GeoDataFrame,
                 hour_key: pd.Timestamp, inrix: pd.DataFrame,
                 hour: int) -> tuple[pd.DataFrame, float]:
    """Join one hour's synthetic loads to that hour's observed congestion."""
    edges = edges_all[edges_all["hour"].eq(hour_key)]
    if edges.empty:
        return pd.DataFrame(
            columns=["xd", "load_mean", "load_max", "highway", "frc", "congestion"]
        ), 0.0

    edges_proj = ox.project_gdf(edges)
    segs_proj = segments.to_crs(edges_proj.crs)
    matches = match_segments_to_edges(segs_proj, edges_proj)
    loads = segment_loads(matches).merge(segment_classes(matches), on="xd")
    loads = loads.merge(segments[["xd", "frc"]], on="xd", how="left")

    observed = inrix[inrix["hour"].eq(hour)][["xd_id", "congestion", "speed_ratio"]]
    if observed.empty:
        logger.warning("Hour %02d: the INRIX extract has no observations for this "
                       "hour on the INRIX day; nothing to correlate against", hour)
    joined = loads.merge(observed, left_on="xd", right_on="xd_id", how="inner")
    match_rate = len(joined) / len(observed) if len(observed) else 0.0
    logger.info("Hour %02d: %d loaded edges, %d/%d XD segments matched (%.1f%%)",
                hour, len(edges), len(joined), len(observed), 100 * match_rate)
    return joined, match_rate


def _metric_rows(label: str, hour: int, joined: pd.DataFrame,
                 match_rate: float, context: dict) -> list[dict]:
    rho_mean, p_mean, n = spearman(joined["load_mean"], joined["congestion"])
    rho_max, p_max, _ = spearman(joined["load_max"], joined["congestion"])
    values = {
        "spearman_rho_load_mean": rho_mean,
        "spearman_p_load_mean": p_mean,
        "spearman_rho_load_max": rho_max,
        "spearman_p_load_max": p_max,
        "mean_congestion": float(joined["congestion"].mean()) if n else float("nan"),
        "mean_load": float(joined["load_mean"].mean()) if n else float("nan"),
    }
    # The nearest-edge fallback ignores bearing, so the buffer-only figure is
    # the one built purely from direction-checked matches.
    buffered = joined[joined["match_kind"].eq("buffer")] if n else joined
    rho_buf, p_buf, n_buf = spearman(buffered["load_mean"], buffered["congestion"])
    values.update({
        "spearman_rho_load_mean_buffer_only": rho_buf,
        "spearman_p_load_mean_buffer_only": p_buf,
        "n_buffer": float(n_buf),
        "share_nearest": float(1 - n_buf / n) if n else float("nan"),
    })
    buffer_only = {"spearman_rho_load_mean_buffer_only", "spearman_p_load_mean_buffer_only"}
    rows = [
        dict(context, hour=hour, window=label, metric=metric, value=value,
             n_segments=n_buf if metric in buffer_only else n, match_rate=match_rate)
        for metric, value in values.items()
    ]
    return rows + _stratified_rows(label, hour, joined, match_rate, context)


def _stratified_rows(label: str, hour: int, joined: pd.DataFrame,
                     match_rate: float, context: dict) -> list[dict]:
    """Per-road-class rho, plus the pooled within-class-rank rho.

    The unstratified rho above mixes classes: MoveOD loads motorways hardest
    and motorways run closest to free flow, so pooling can flip its sign.
    """
    if joined.empty:
        return []
    rows = []
    for group_col, prefix in (("frc", "frc"), ("highway", "osm")):
        table = stratified_spearman(joined, "load_mean", "congestion", group_col)
        for row in table.itertuples():
            name = getattr(row, group_col)
            rows += [
                dict(context, hour=hour, window=label,
                     metric=f"spearman_rho_load_mean__{prefix}_{name}",
                     value=row.rho, n_segments=int(row.n), match_rate=match_rate),
                dict(context, hour=hour, window=label,
                     metric=f"spearman_p_load_mean__{prefix}_{name}",
                     value=row.p, n_segments=int(row.n), match_rate=match_rate),
            ]
        rho, p, n = within_class_rank_spearman(
            joined, "load_mean", "congestion", group_col
        )
        rows += [
            dict(context, hour=hour, window=label,
                 metric=f"spearman_rho_within_class_{prefix}", value=rho,
                 n_segments=n, match_rate=match_rate),
            dict(context, hour=hour, window=label,
                 metric=f"spearman_p_within_class_{prefix}", value=p,
                 n_segments=n, match_rate=match_rate),
        ]
    return rows


def _plot_deciles(deciles: dict, hours: dict, results: dict,
                  out_path: Path, title: str) -> None:
    """Decile curve on the left, per-FRC scatter small multiples on the right."""
    frcs = sorted({f for joined, _ in results.values()
                   for f in joined["frc"].dropna().unique()})
    fig, axes = plt.subplots(1, 1 + len(frcs), figsize=(4.2 + 2.6 * len(frcs), 4.0))
    axes = np.atleast_1d(axes)

    for label, frame in deciles.items():
        if frame.empty:
            continue
        axes[0].plot(frame["decile"] + 1, frame["congestion_mean"], marker="o",
                     label=f"{label} ({hours[label]:02d}:00-{hours[label] + 1:02d}:00)")
    axes[0].set_xlabel("Synthetic load decile (1 = least loaded)")
    axes[0].set_ylabel("Mean observed congestion  (1 - speed / reference)")
    axes[0].set_title("Pooled (confounded by road class)", fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    for ax, frc in zip(axes[1:], frcs):
        for label, (joined, _) in results.items():
            sub = joined[joined["frc"].eq(frc)]
            if sub.empty:
                continue
            ax.scatter(sub["load_mean"] + 1, sub["congestion"], s=5, alpha=0.35,
                       label=label)
            rho, _, n = spearman(sub["load_mean"], sub["congestion"])
            ax.plot([], [], " ", label=f"  {label}: rho={rho:.3f}, n={n}")
        ax.set_xscale("log")
        ax.set_xlabel("Synthetic load + 1")
        ax.set_title(f"FRC {frc}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_summary(path: Path, context: dict, results: dict, deciles: dict) -> None:
    lines = [
        f"# Link-load validation — {context['county']} County, {context['state']}",
        "",
        f"Run `{context['run_id']}`, synthetic day {context['day']}, "
        f"INRIX day **{context['inrix_day']}**, code `{context['code_sha']}`.",
        "",
        "Synthetic per-edge loads come from re-routing the calibrated trips; the "
        "run itself was built from OSM default speeds, so INRIX is an "
        f"independent observation here. Loads are bucketed by **{context['load_hours']} "
        "hour**.",
        "",
        "| window | hour | n segments | match rate | Spearman rho (mean load) | p "
        "| n buffer-matched | share nearest-fallback | rho buffer-only | p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, (joined, match_rate) in results.items():
        rho, p, n = spearman(joined["load_mean"], joined["congestion"])
        buffered = joined[joined["match_kind"].eq("buffer")] if n else joined
        rho_b, p_b, n_b = spearman(buffered["load_mean"], buffered["congestion"])
        lines.append(
            f"| {label} | {context['hours'][label]:02d} | {n} | "
            f"{match_rate:.1%} | {rho:.4f} | {p:.3g} | {n_b} | "
            f"{(1 - n_b / n) if n else float('nan'):.1%} | {rho_b:.4f} | {p_b:.3g} |"
        )
    lines += _stratified_tables(results)
    for label, frame in deciles.items():
        lines += ["", f"## {label} deciles", "",
                  "| decile | n | mean load | mean congestion |",
                  "| --- | --- | --- | --- |"]
        lines += [
            f"| {int(r.decile) + 1} | {int(r.n)} | {r.load_mean:.2f} | "
            f"{r.congestion_mean:.4f} |"
            for r in frame.itertuples()
        ]
    path.write_text("\n".join(lines) + "\n")


def _stratified_tables(results: dict) -> list[str]:
    """Per-class rho tables; the pooled rho above mixes road classes."""
    lines = []
    for group_col, heading in (("frc", "INRIX FRC (1 = motorway .. 5 = local)"),
                               ("highway", "OSM highway class of matched edges")):
        lines += ["", f"## Spearman rho within {heading}", "",
                  "| window | class | n | rho | p |", "| --- | --- | --- | --- | --- |"]
        for label, (joined, _) in results.items():
            if joined.empty:
                continue
            table = stratified_spearman(joined, "load_mean", "congestion", group_col)
            lines += [
                f"| {label} | {getattr(r, group_col)} | {int(r.n)} | "
                f"{r.rho:.4f} | {r.p:.3g} |" for r in table.itertuples()
            ]
            rho, p, n = within_class_rank_spearman(
                joined, "load_mean", "congestion", group_col
            )
            lines.append(f"| {label} | **pooled within-class ranks** | {n} | "
                         f"{rho:.4f} | {p:.3g} |")
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MoveOD synthetic link loads with observed INRIX speeds."
    )
    parser.add_argument("--output-root", default="move_OD", help="Root output directory")
    parser.add_argument("--state", default="Tennessee", help="State name")
    parser.add_argument("--county", default="Hamilton", help="County name")
    parser.add_argument("--run-id", default=None, help="Run folder name")
    parser.add_argument("--inrix", required=True, help="INRIX observation CSV")
    parser.add_argument("--xd", required=True, help="INRIX XD identification CSV")
    parser.add_argument("--am", type=int, default=7, help="AM hour (default 7 = 7-8am)")
    parser.add_argument("--pm", type=int, default=17, help="PM hour (default 17 = 5-6pm)")
    parser.add_argument("--inrix-day", default=None,
                        help="INRIX observation day YYYY-MM-DD (default: the run day)")
    parser.add_argument("--load-hours", choices=("traversal", "departure"),
                        default="traversal",
                        help="Bucket edge loads by the hour a trip is on the edge "
                             "(traversal, default) or the hour it departed")
    parser.add_argument("--tmas-sta", default=None,
                        help="TMAS station file; enables the count-station comparison")
    parser.add_argument("--tmas-vol", default=None, help="TMAS monthly volume file")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write the outputs (default <run_dir>/validation); "
                             "point variants elsewhere so they do not clobber each other")
    parser.add_argument("--county-code", type=int, default=65,
                        help="TMAS County_Code for the county (Hamilton TN = 65)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    run_dir = find_run_dir(Path(args.output_root), args.state, args.county, args.run_id)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_all, day = routed_edge_loads(run_dir, args.load_hours)
    edges_all.to_parquet(out_dir / "edge_loads.parquet")

    inrix_day = (datetime.date.fromisoformat(args.inrix_day)
                 if args.inrix_day else day)
    hours = {"AM": args.am, "PM": args.pm}
    inrix = read_inrix_ratios(Path(args.inrix), inrix_day, list(hours.values()))
    segments = read_xd_segments(Path(args.xd))

    context = {
        "state": args.state, "county": args.county, "run_id": run_dir.name,
        "day": str(day), "inrix_day": str(inrix_day), "load_hours": args.load_hours,
        "code_sha": code_sha(), "hours": hours,
    }
    results, deciles, rows = {}, {}, []
    for label, hour in hours.items():
        # The synthetic loads always come from the run day; only the observation
        # day can differ, which is why the two are recorded separately.
        hour_key = pd.Timestamp(day) + pd.Timedelta(hours=hour)
        joined, match_rate = compare_hour(edges_all, segments, hour_key, inrix, hour)
        results[label] = (joined, match_rate)
        deciles[label] = decile_means(joined["load_mean"], joined["congestion"])
        rows += _metric_rows(label, hour, joined, match_rate,
                             {k: v for k, v in context.items() if k != "hours"})

    pd.DataFrame(rows).to_csv(out_dir / "link_loads_metrics.csv", index=False)
    _plot_deciles(deciles, hours, results, out_dir / "inrix_congestion_vs_load.png",
                  f"{args.county} County: synthetic loads {day} "
                  f"vs INRIX {inrix_day} ({args.load_hours} hours)")
    _write_summary(out_dir / "link_loads_summary.md", context, results, deciles)

    if args.tmas_sta and args.tmas_vol:
        if args.load_hours != "traversal":
            logger.warning("--load-hours=%s with --tmas-*: a count station records "
                           "vehicles as they pass, so the station comparison is "
                           "only meaningful under traversal hours",
                           args.load_hours)
        station_counts.run(
            edges_all=edges_all, day=day, out_dir=out_dir,
            sta_path=Path(args.tmas_sta), vol_path=Path(args.tmas_vol),
            county_code=args.county_code,
            context={k: v for k, v in context.items() if k != "hours"},
        )
    logger.info("Wrote validation outputs to %s", out_dir)


if __name__ == "__main__":
    main()
