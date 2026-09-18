"""Compare MoveOD synthetic link loads with FHWA TMAS continuous-count stations.

INRIX tells us how fast a road was; a count station tells us how many vehicles
crossed it. The second is the more direct test of a trip-generation model, and
it comes with an expected answer: MoveOD synthesises commute trips only, so the
synthetic count should be a *fraction* of the observed count, largest in the
morning peak and smallest off-peak. The shape of the morning profile is the
part that should match.

Two TMG 2001 files are read:

* ``.STA`` -- one row per station / direction / lane, carrying the location.
* ``.VOL`` -- one row per station / direction / lane / day, with 24 hourly
  counts. ``Travel_Lane`` 0 means "all lanes in this direction"; when such a
  row exists it is authoritative, otherwise the per-lane rows are summed.

Loads must have been collected with ``edge_load_hours="traversal"``: a count
station records vehicles as they pass, not as they set off.
"""

import logging
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_HOUR_COLS = [f"Hour_{h:02d}" for h in range(24)]

# TMG 2001 Travel_Dir codes -> compass bearings. 9 and 0 are the "both
# directions combined" codes; the diagonal variants they also cover (NE-SW for
# 9, SE-NW for 0) cannot be told apart from the code alone, so the cardinal
# pair is used and the +/-45 degree tolerance absorbs the difference.
_DIRECTION_BEARINGS = {
    1: (0.0,), 2: (45.0,), 3: (90.0,), 4: (135.0,),
    5: (180.0,), 6: (225.0,), 7: (270.0,), 8: (315.0,),
    9: (0.0, 180.0),
    0: (90.0, 270.0),
}

AM_HOURS = (5, 6, 7, 8, 9, 10)
DAY_HOURS = tuple(range(24))


def direction_bearings(code: int) -> tuple[float, ...]:
    """Compass bearings a TMG Travel_Dir code covers."""
    return _DIRECTION_BEARINGS[int(code)]


# -- Readers -----------------------------------------------------------------

def read_stations(sta_path: Path, county_code: int) -> gpd.GeoDataFrame:
    """One row per station-direction in the county, with its location."""
    df = pd.read_csv(sta_path, sep="|", dtype=str)
    counties = pd.to_numeric(df["County_Code"], errors="coerce")
    df = df[counties.eq(county_code)].copy()
    df["station_id"] = df["Station_Id"]
    df["travel_dir"] = pd.to_numeric(df["Travel_Dir"], errors="coerce").astype(int)
    df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["route"] = df["Posted_Route_Sign_Number"].str.strip()
    df = df.dropna(subset=["lat", "lon"])
    keep = ["station_id", "travel_dir", "lat", "lon", "route", "F_System"]
    df = df[keep].drop_duplicates(subset=["station_id", "travel_dir"])
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    logger.info("TMAS: %d station-directions (%d stations) in county %s",
                len(gdf), gdf["station_id"].nunique(), county_code)
    return gdf


def read_volumes(vol_path: Path, day: int, station_ids: Iterable[str] | None = None,
                 year: int | None = None, month: int | None = None) -> pd.DataFrame:
    """Hourly counts per station-direction for one day.

    A lane-0 row holds every lane in that direction, so when one is present the
    individual lane rows would double-count and are ignored. Year and month are
    filtered too when given: a monthly file can carry more than one of either.
    """
    df = pd.read_csv(vol_path, sep="|", dtype=str)
    df = df[pd.to_numeric(df["Day_Record"], errors="coerce").eq(day)]
    if year is not None:
        df = df[pd.to_numeric(df["Year_Record"], errors="coerce").eq(year)]
    if month is not None:
        df = df[pd.to_numeric(df["Month_Record"], errors="coerce").eq(month)]
    if station_ids is not None:
        df = df[df["Station_Id"].isin(set(station_ids))]
    df = df.assign(
        travel_dir=pd.to_numeric(df["Travel_Dir"], errors="coerce").astype("Int64"),
        travel_lane=pd.to_numeric(df["Travel_Lane"], errors="coerce"),
        **{c: pd.to_numeric(df[c], errors="coerce") for c in _HOUR_COLS},
    )

    frames = []
    for (station, direction), group in df.groupby(["Station_Id", "travel_dir"]):
        lane0 = group[group["travel_lane"].eq(0)]
        used = lane0 if len(lane0) else group
        frames.append(pd.DataFrame({
            "station_id": station,
            "travel_dir": int(direction),
            "hour_of_day": range(24),
            "count": used[_HOUR_COLS].sum(axis=0).to_numpy(dtype=float),
            "lane_source": "lane_0" if len(lane0) else "sum_lanes",
            "n_lane_rows": len(used),
        }))
    if not frames:
        return pd.DataFrame(columns=["station_id", "travel_dir", "hour_of_day",
                                     "count", "lane_source", "n_lane_rows"])
    out = pd.concat(frames, ignore_index=True)
    logger.info("TMAS: %d station-directions report on day %d",
                out.groupby(["station_id", "travel_dir"]).ngroups, day)
    return out


# -- Matching ----------------------------------------------------------------

def _bearing_difference(a, b) -> np.ndarray:
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def _direction_targets(stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per (station-direction, carriageway bearing)."""
    rows = []
    for station in stations.itertuples():
        for group, bearing in enumerate(direction_bearings(station.travel_dir)):
            rows.append({"station_id": station.station_id,
                         "travel_dir": station.travel_dir,
                         "bearing_group": group, "target_bearing": bearing,
                         "geometry": station.geometry})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=stations.crs)


def _match_within(targets: gpd.GeoDataFrame, edges: gpd.GeoDataFrame,
                  radius_m: float, bearing_tol_deg: float) -> pd.DataFrame:
    buffers = targets.assign(geometry=targets.geometry.buffer(radius_m))
    hits = gpd.sjoin(edges, buffers, how="inner", predicate="intersects")
    if len(hits):
        hits = hits[_bearing_difference(hits["bearing"], hits["target_bearing"])
                    <= bearing_tol_deg]
    return hits[["station_id", "travel_dir", "bearing_group", "u", "v",
                 "highway"]].copy()


def match_stations_to_edges(stations: gpd.GeoDataFrame, edges: gpd.GeoDataFrame,
                            radius_m: float = 50.0, widen_m: float = 100.0,
                            bearing_tol_deg: float = 45.0) -> pd.DataFrame:
    """Loaded OSM edges representing each station-direction's cross-section.

    Both frames must already be projected to the same metric CRS. Unlike the XD
    matcher this tests the whole edge geometry rather than its midpoint: a long
    edge can run right past a station while its midpoint is far away. Stations
    with nothing inside radius_m are retried once at widen_m and labelled.
    """
    targets = _direction_targets(stations)
    matched = _match_within(targets, edges, radius_m, bearing_tol_deg)
    matched["match_kind"] = f"{int(radius_m)}m"

    found = set(zip(matched["station_id"], matched["travel_dir"]))
    still = [t not in found for t in zip(targets["station_id"], targets["travel_dir"])]
    missing = targets[still]
    if len(missing) and widen_m > radius_m:
        wider = _match_within(missing, edges, widen_m, bearing_tol_deg)
        wider["match_kind"] = f"{int(widen_m)}m"
        matched = pd.concat([matched, wider], ignore_index=True)

    return matched.drop_duplicates(
        subset=["station_id", "travel_dir", "bearing_group", "u", "v"]
    ).reset_index(drop=True)


# Best-to-worst OSM road classes. A count station on a freeway can sit within
# 50 m of a frontage road, and averaging the two halves the synthetic count.
_CLASS_ORDER = (
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
    "unclassified", "residential",
)
_CLASS_RANK = {name: i for i, name in enumerate(_CLASS_ORDER)}


def keep_highest_class(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within each carriageway keep only the edges of its best OSM class.

    Returns (kept, dropped) so the caller can say what it discarded. Classes
    outside _CLASS_ORDER rank last, so a named class always wins.
    """
    if matches.empty:
        return matches, matches
    rank = matches["highway"].map(_CLASS_RANK).fillna(len(_CLASS_ORDER))
    ranked = matches.assign(_rank=rank)
    best = ranked.groupby(["station_id", "travel_dir", "bearing_group"])["_rank"]
    keep = ranked["_rank"].eq(best.transform("min"))
    return (ranked[keep].drop(columns="_rank").reset_index(drop=True),
            ranked[~keep].drop(columns="_rank").reset_index(drop=True))


def station_hourly_loads(matches: pd.DataFrame,
                         edges_all: pd.DataFrame) -> pd.DataFrame:
    """Synthetic count per station-direction-hour.

    Averaged over the edges matched to one carriageway -- consecutive edges are
    the same cross-section, so summing them would count each trip once per edge
    -- then summed across the carriageways of a combined direction code. An
    edge with no row for an hour carried no trips that hour, so it counts zero
    rather than dropping out of the average.
    """
    if matches.empty:
        return pd.DataFrame(columns=["station_id", "travel_dir", "hour", "synthetic"])

    hours = pd.DataFrame({"hour": sorted(edges_all["hour"].unique())})
    grid = matches.merge(hours, how="cross").merge(
        edges_all[["u", "v", "hour", "load"]], on=["u", "v", "hour"], how="left"
    )
    grid["load"] = grid["load"].fillna(0.0)
    per_carriageway = grid.groupby(
        ["station_id", "travel_dir", "bearing_group", "hour"]
    )["load"].mean()
    return (
        per_carriageway.groupby(level=["station_id", "travel_dir", "hour"]).sum()
        .rename("synthetic").reset_index()
    )


def synthetic_by_clock_hour(synthetic: pd.DataFrame) -> pd.DataFrame:
    """Collapse hour buckets to clock hours of the day.

    Traversal-hour loads run past midnight into a 25th bucket, whose clock hour
    is 0 -- the same as the run day's first bucket. Summing here keeps one row
    per station-direction-hour; without it the observed counts join twice and
    the synthetic load is split across the two rows.
    """
    if synthetic.empty:
        return synthetic.assign(hour_of_day=pd.Series(dtype=int))
    hours = pd.DatetimeIndex(synthetic["hour"]).hour
    return (
        synthetic.assign(hour_of_day=hours)
        .groupby(["station_id", "travel_dir", "hour_of_day"], as_index=False)
        ["synthetic"].sum()
    )


# -- Metrics -----------------------------------------------------------------

def geh(model: Iterable[float], count: Iterable[float]) -> np.ndarray:
    """GEH statistic; < 5 is the usual "acceptable" threshold for hourly flows."""
    m = np.asarray(model, dtype=float)
    c = np.asarray(count, dtype=float)
    total = m + c
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.sqrt(2.0 * (m - c) ** 2 / total)
    return np.where(total > 0, value, np.nan)


def _pearson(x, y) -> tuple[float, int]:
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 3 or np.ptp(xs[ok]) == 0 or np.ptp(ys[ok]) == 0:
        return float("nan"), int(ok.sum())
    return float(stats.pearsonr(xs[ok], ys[ok]).statistic), int(ok.sum())


def profile_shape_correlation(df: pd.DataFrame,
                              hours: Sequence[int]) -> tuple[pd.DataFrame, float]:
    """Pearson r between each station-direction's synthetic and observed shape.

    Each profile is normalised to sum 1 over `hours` first, so this asks only
    whether the synthetic peak falls at the same time, not whether it is the
    same size.
    """
    sub = df[df["hour_of_day"].isin(list(hours))].copy()
    for column in ("synthetic", "observed"):
        totals = sub.groupby(["station_id", "travel_dir"])[column].transform("sum")
        sub[f"{column}_share"] = np.where(totals > 0, sub[column] / totals, np.nan)

    rows = []
    for (station, direction), group in sub.groupby(["station_id", "travel_dir"]):
        r, n = _pearson(group["synthetic_share"], group["observed_share"])
        rows.append({"station_id": station, "travel_dir": direction,
                     "shape_r": r, "n_hours": n})
    pooled, _ = _pearson(sub["synthetic_share"], sub["observed_share"])
    return pd.DataFrame(rows), pooled


def _window_metrics(df: pd.DataFrame, hours: Sequence[int], label: str) -> dict:
    sub = df[df["hour_of_day"].isin(list(hours))]
    if sub.empty:
        return {}
    values = geh(sub["synthetic"], sub["observed"])
    scored = values[np.isfinite(values)]
    r_log, n = _pearson(np.log1p(sub["synthetic"]), np.log1p(sub["observed"]))
    observed_total = float(sub["observed"].sum())
    return {
        f"geh_lt_5_share_{label}": float((scored < 5).mean()) if scored.size else float("nan"),
        f"geh_lt_10_share_{label}": float((scored < 10).mean()) if scored.size else float("nan"),
        f"geh_median_{label}": float(np.median(scored)) if scored.size else float("nan"),
        f"n_scored_station_hours_{label}": float(scored.size),
        f"pearson_r_log1p_{label}": r_log,
        f"r2_log1p_{label}": r_log ** 2 if np.isfinite(r_log) else float("nan"),
        f"ratio_overall_{label}": (float(sub["synthetic"].sum()) / observed_total
                                   if observed_total else float("nan")),
        f"n_station_hours_{label}": float(n),
    }


def compute_metrics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, float]:
    """Headline metrics, the per-station shape table and the pooled shape r."""
    metrics = {}
    metrics.update(_window_metrics(df, AM_HOURS, "am"))
    metrics.update(_window_metrics(df, DAY_HOURS, "day"))

    hourly = df.groupby("hour_of_day")[["synthetic", "observed"]].sum()
    for hour, row in hourly.iterrows():
        metrics[f"ratio_hour_{int(hour):02d}"] = (
            float(row["synthetic"] / row["observed"]) if row["observed"] else float("nan")
        )

    shapes, pooled = profile_shape_correlation(df, AM_HOURS)
    metrics["profile_shape_pearson_pooled_am"] = pooled
    for row in shapes.itertuples():
        metrics[f"profile_shape_pearson__{row.station_id}_dir{row.travel_dir}"] = row.shape_r
    return metrics, shapes, pooled


# -- Outputs -----------------------------------------------------------------

def _plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Observed vs synthetic hourly counts, 05-10, one panel per station-direction."""
    keys = sorted(df.groupby(["station_id", "travel_dir"]).groups)
    cols = min(4, max(1, len(keys)))
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows),
                             squeeze=False)
    width = 0.4
    for ax, (station, direction) in zip(axes.ravel(), keys):
        sub = df[df["station_id"].eq(station) & df["travel_dir"].eq(direction)]
        sub = sub[sub["hour_of_day"].isin(AM_HOURS)].sort_values("hour_of_day")
        x = np.arange(len(sub))
        ax.bar(x - width / 2, sub["observed"], width, label="TMAS observed")
        ax.bar(x + width / 2, sub["synthetic"], width, label="MoveOD synthetic")
        ax.set_xticks(x, [f"{h:02d}" for h in sub["hour_of_day"]], fontsize=7)
        route = sub["route"].iat[0] if "route" in sub and len(sub) else ""
        ratio = sub["synthetic"].sum() / max(sub["observed"].sum(), 1)
        ax.set_title(f"{station} dir {direction} {route}\nratio {ratio:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _append_summary(path: Path, metrics: dict, shapes: pd.DataFrame,
                    unmatched: pd.DataFrame, df: pd.DataFrame) -> None:
    hourly = df.groupby("hour_of_day")[["synthetic", "observed"]].sum()
    lines = [
        "", "## TMAS continuous-count stations", "",
        "Synthetic loads are bucketed by *traversal* hour here: a count station "
        "records vehicles as they pass. MoveOD synthesises commute trips only, "
        "so the synthetic/observed ratio is expected to be well below 1; the "
        "profile shape is the part that should match.", "",
        "| metric | value |", "| --- | --- |",
    ]
    lines += [f"| {name} | {value:.4g} |" for name, value in metrics.items()
              if not name.startswith("profile_shape_pearson__")
              and not name.startswith("ratio_hour_")]
    lines += ["", "### Synthetic / observed by hour", "",
              "| hour | observed | synthetic | ratio |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {int(h):02d} | {r.observed:,.0f} | {r.synthetic:,.1f} | "
        f"{(r.synthetic / r.observed if r.observed else float('nan')):.3f} |"
        for h, r in hourly.iterrows()
    ]
    lines += ["", "### AM profile shape (05-10, each profile normalised to sum 1)", "",
              "| station | dir | Pearson r |", "| --- | --- | --- |"]
    lines += [f"| {r.station_id} | {r.travel_dir} | {r.shape_r:.4f} |"
              for r in shapes.itertuples()]
    if len(unmatched):
        lines += ["", "Unmatched station-directions (no loaded edge within 100 m "
                  "at the right bearing): "
                  + ", ".join(f"{r.station_id} dir {r.travel_dir} ({r.route})"
                              for r in unmatched.itertuples())]
    with open(path, "a") as fh:
        fh.write("\n".join(lines) + "\n")


def reporting_sites(stations: gpd.GeoDataFrame,
                    volumes: pd.DataFrame) -> gpd.GeoDataFrame:
    """The station-directions that actually reported, located on the ground.

    A station-direction can report volume without a matching .STA direction row
    (Hamilton 000540 is signed N/S there but reports E/W), so the point comes
    from the station and the bearing from the volume record.
    """
    points = stations.drop_duplicates(subset="station_id").set_index("station_id")
    reporting = volumes[["station_id", "travel_dir"]].drop_duplicates()
    reporting = reporting[reporting["station_id"].isin(points.index)]
    return gpd.GeoDataFrame(
        reporting.assign(
            route=points.loc[reporting["station_id"], "route"].to_numpy(),
            geometry=points.loc[reporting["station_id"], "geometry"].to_numpy(),
        ),
        geometry="geometry", crs=stations.crs,
    )


def run(edges_all: gpd.GeoDataFrame, day, out_dir: Path, sta_path: Path,
        vol_path: Path, county_code: int, context: dict) -> pd.DataFrame:
    """Match TMAS stations to loaded edges and write the count comparison."""
    stations = read_stations(sta_path, county_code)
    stamp = pd.Timestamp(day)
    volumes = read_volumes(vol_path, day=stamp.day, year=stamp.year,
                           month=stamp.month,
                           station_ids=stations["station_id"].unique())
    if volumes.empty:
        logger.warning("TMAS: no volume rows for %s; skipping station comparison", day)
        return pd.DataFrame()

    sites = reporting_sites(stations, volumes)
    edges_proj = ox.project_gdf(edges_all.drop_duplicates(subset=["u", "v"]))
    matches = match_stations_to_edges(sites.to_crs(edges_proj.crs), edges_proj)
    matches, dropped = keep_highest_class(matches)
    for (station, direction), group in dropped.groupby(["station_id", "travel_dir"]):
        logger.info("TMAS %s dir %s: dropped %s (lower class than %s)", station,
                    direction, sorted(set(group["highway"])),
                    sorted(set(matches[matches["station_id"].eq(station)]["highway"])))
    synthetic = synthetic_by_clock_hour(station_hourly_loads(matches, edges_all))

    matched_sites = set(zip(matches["station_id"], matches["travel_dir"]))
    unmatched = sites[[t not in matched_sites
                       for t in zip(sites["station_id"], sites["travel_dir"])]]
    logger.info("TMAS: %d/%d station-directions matched to loaded edges",
                len(sites) - len(unmatched), len(sites))

    kinds = matches.groupby(["station_id", "travel_dir"], as_index=False).agg(
        match_kind=("match_kind", "min"), n_edges=("u", "size"))
    df = volumes.rename(columns={"count": "observed"}).merge(
        synthetic, on=["station_id", "travel_dir", "hour_of_day"], how="inner",
    ).merge(sites[["station_id", "travel_dir", "route"]],
            on=["station_id", "travel_dir"], how="left").merge(
        kinds, on=["station_id", "travel_dir"], how="left")
    if df.duplicated(["station_id", "travel_dir", "hour_of_day"]).any():
        raise RuntimeError("TMAS merge produced more than one row per station-direction-hour")
    if df.empty:
        logger.warning("TMAS: nothing matched; skipping station comparison")
        return df

    metrics, shapes, _ = compute_metrics(df)
    sites_matched = kinds["match_kind"].value_counts()
    metrics["n_station_dirs_matched_50m"] = float(sites_matched.get("50m", 0))
    metrics["n_station_dirs_matched_100m"] = float(sites_matched.get("100m", 0))
    metrics["n_station_dirs_unmatched"] = float(len(unmatched))
    rows = [dict(context, tmas_day=str(day), metric=name, value=value,
                 n_station_dirs=df.groupby(["station_id", "travel_dir"]).ngroups,
                 n_unmatched=len(unmatched))
            for name, value in metrics.items()]
    pd.DataFrame(rows).to_csv(out_dir / "station_counts_metrics.csv", index=False)
    df.to_csv(out_dir / "station_counts_hourly.csv", index=False)
    _plot(df, out_dir / "station_counts.png",
          f"{context.get('county', '')} TMAS stations {day}: observed vs synthetic, 05-10")
    _append_summary(out_dir / "link_loads_summary.md", metrics, shapes, unmatched, df)
    return df
