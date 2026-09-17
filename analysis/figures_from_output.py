import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_run_dir(output_root: Path, state: str, county: str, run_id: str | None) -> Path:
    base = output_root / state / county
    if run_id:
        candidate = base / run_id
        if not candidate.exists():
            raise FileNotFoundError(f"Run folder not found: {candidate}")
        return candidate

    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run folders found under {base}")
    return runs[-1]


def find_calibrated_csv(run_dir: Path) -> Path:
    calib_dir = run_dir / "calibrated_move_od"
    if not calib_dir.exists():
        raise FileNotFoundError(f"Missing calibrated_move_od folder: {calib_dir}")

    candidates = sorted(calib_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No calibrated CSVs found in {calib_dir}")
    return candidates[-1]


def find_initial_frame(run_dir: Path) -> Path | None:
    """Locate the pre-calibration ("Initial") assignment.

    The live pipeline writes the initial routed trips to
    intermediate/<day>/post_mssr_routing_df.parquet -- that is the dataset the
    ILP calibrates, so it is what Figure 2's "Initial" series should show. Falls
    back to the pre-speed-shift routing frame, then to the older
    lodes_combs/lodes_<day>.csv layout.
    """
    for pattern in ("intermediate/*/post_mssr_routing_df.parquet",
                    "intermediate/*/routing_df.parquet"):
        candidates = sorted(run_dir.glob(pattern))
        if candidates:
            return candidates[-1]

    lodes_dir = run_dir / "lodes_combs"
    if lodes_dir.exists():
        candidates = sorted(lodes_dir.glob("lodes_*.csv"))
        if candidates:
            return candidates[-1]
    return None


def read_table(path: Path) -> pd.DataFrame:
    """Read a .parquet or .csv table based on its suffix."""
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def find_census_table(census_dir: Path, stem: str) -> Path | None:
    """Census tables are written as .parquet; older runs wrote .csv."""
    for suffix in (".parquet", ".csv"):
        candidate = census_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def plot_od_map(df: pd.DataFrame, out_path: Path, sample_size: int = 5000) -> None:
    plot_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    origin_lat_col = pick_first_col(df, ["origin_lat", "origin_loc_lat", "home_loc_lat"])
    origin_lon_col = pick_first_col(df, ["origin_lon", "origin_loc_lon", "home_loc_lon"])
    dest_lat_col = pick_first_col(df, ["dest_lat", "dest_loc_lat", "work_loc_lat"])
    dest_lon_col = pick_first_col(df, ["dest_lon", "dest_loc_lon", "work_loc_lon"])
    if not all([origin_lat_col, origin_lon_col, dest_lat_col, dest_lon_col]):
        raise KeyError("Origin/destination lat/lon columns not found in calibrated data.")

    plt.figure(figsize=(7, 7))
    plt.scatter(plot_df[origin_lon_col], plot_df[origin_lat_col], s=2, alpha=0.4, label="Origins")
    plt.scatter(plot_df[dest_lon_col], plot_df[dest_lat_col], s=2, alpha=0.4, label="Destinations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Supplementary Figure A: OD Locations (sample)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def get_departure_seconds(frame: pd.DataFrame) -> pd.Series:
    sec_col = pick_first_col(
        frame, ["departure_time", "departure_time_secs", "departure_time_sec", "departure_seconds"]
    )
    if sec_col:
        return pd.to_numeric(frame[sec_col], errors="coerce").dropna()
    time_col = pick_first_col(frame, ["departure_time", "departure_time_str", "go_time", "go_time_str"])
    if time_col:
        ts = pd.to_datetime(frame[time_col], errors="coerce")
        return (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).dropna()
    raise KeyError("No departure time column found in calibrated data.")


def plot_departure_hist(df: pd.DataFrame, out_path: Path) -> None:
    seconds = get_departure_seconds(df)
    hours = (seconds / 3600.0) % 24

    plt.figure(figsize=(7, 4))
    plt.hist(hours, bins=24, color="#4C78A8", alpha=0.85)
    plt.xlabel("Departure Hour (0–24)")
    plt.ylabel("Trips")
    plt.title("Supplementary Figure B: Departure Time Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_departure_calibration(df: pd.DataFrame, dept_df: pd.DataFrame | None, out_path: Path) -> None:
    dep_bins = [
        0,
        5 * 3600,
        int(5.5 * 3600),
        6 * 3600,
        int(6.5 * 3600),
        7 * 3600,
        int(7.5 * 3600),
        8 * 3600,
        int(8.5 * 3600),
        9 * 3600,
        10 * 3600,
        11 * 3600,
        12 * 3600,
        16 * 3600,
        24 * 3600,
    ]
    dep_labels = [
        "12am to 5am",
        "5am to 5:30am",
        "5:30am to 6am",
        "6am to 6:30am",
        "6:30am to 7am",
        "7am to 7:30am",
        "7:30am to 8am",
        "8am to 8:30am",
        "8:30am to 9am",
        "9am to 10am",
        "10am to 11am",
        "11am to 12pm",
        "12pm to 4pm",
        "4pm to 12am",
    ]

    def census_departure_pct_from_wide(frame: pd.DataFrame) -> pd.Series | None:
        wide_map = {
            "12am_to_4:59am_estimate": "12am to 5am",
            "5am_to_5:29am_estimate": "5am to 5:30am",
            "5:30am_to_5:59am_estimate": "5:30am to 6am",
            "6am_to_6:29am_estimate": "6am to 6:30am",
            "6:30am_to_6:59am_estimate": "6:30am to 7am",
            "7am_to_7:29am_estimate": "7am to 7:30am",
            "7:30am_to_7:59am_estimate": "7:30am to 8am",
            "8am_to_8:29am_estimate": "8am to 8:30am",
            "8:30am_to_8:59am_estimate": "8:30am to 9am",
            "9am_to_9:59am_estimate": "9am to 10am",
            "10am_to_10:59am_estimate": "10am to 11am",
            "11am_to_11:59am_estimate": "11am to 12pm",
            "12pm_to_3:59pm_estimate": "12pm to 4pm",
            "4pm_to_11:59pm_estimate": "4pm to 12am",
        }
        cols = [c for c in wide_map if c in frame.columns]
        if not cols:
            return None
        totals = frame[cols].apply(pd.to_numeric, errors="coerce").sum()
        total = totals.sum()
        if total == 0:
            return None
        pct = (totals / total * 100).rename(index=wide_map)
        return pct.reindex(dep_labels).fillna(0)

    gen_seconds = get_departure_seconds(df)
    gen_bins = pd.cut(gen_seconds, bins=dep_bins, labels=dep_labels, right=False, include_lowest=True)
    gen_pct = gen_bins.value_counts(normalize=True).reindex(dep_labels).fillna(0) * 100

    census_pct = None
    if dept_df is not None:
        census_pct = census_departure_pct_from_wide(dept_df)
        if census_pct is None:
            label_col = pick_first_col(dept_df, ["time_bin", "time", "bin", "label", "departure_time_bin"])
            value_col = pick_first_col(dept_df, ["percent", "share", "pct", "percentage", "value"])
            if label_col is None or value_col is None:
                if dept_df.shape[1] >= 2:
                    label_col, value_col = dept_df.columns[:2]
            if label_col and value_col:
                tmp = dept_df[[label_col, value_col]].copy()
                tmp[label_col] = tmp[label_col].astype(str).str.lower().str.replace(" ", "", regex=False)
                label_map = {
                    "12amto5am": "12am to 5am",
                    "5amto5:30am": "5am to 5:30am",
                    "5:30amto6am": "5:30am to 6am",
                    "6amto6:30am": "6am to 6:30am",
                    "6:30amto7am": "6:30am to 7am",
                    "7amto7:30am": "7am to 7:30am",
                    "7:30amto8am": "7:30am to 8am",
                    "8amto8:30am": "8am to 8:30am",
                    "8:30amto9am": "8:30am to 9am",
                    "9amto10am": "9am to 10am",
                    "10amto11am": "10am to 11am",
                    "11amto12pm": "11am to 12pm",
                    "12pmto4pm": "12pm to 4pm",
                    "4pmto12am": "4pm to 12am",
                }
                tmp["label"] = tmp[label_col].map(label_map)
                tmp = tmp.dropna(subset=["label"])
                census_pct = tmp.groupby("label")[value_col].mean().reindex(dep_labels).fillna(0).astype(float)

    x = np.arange(len(dep_labels))
    width = 0.35
    plt.figure(figsize=(9, 4.5))
    plt.bar(
        x - width / 2,
        gen_pct.values,
        width,
        label="Generated",
        color="#4C78A8",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    if census_pct is not None:
        plt.bar(
            x + width / 2,
            census_pct.values,
            width,
            label="Census",
            color="#E45756",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
    plt.xticks(x, dep_labels, rotation=45, ha="right")
    plt.ylabel("Trip Distribution (%)")
    plt.xlabel("Departure Time Bin")
    plt.title("Figure 1: Departure Time Calibration")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_travel_time_distributions(
    df_calib: pd.DataFrame, df_init: pd.DataFrame | None, tt_df: pd.DataFrame | None, out_path: Path
) -> None:
    tt_bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90, 10**9]
    tt_labels = ["<5", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-59", "60-89", "90+"]

    def get_travel_time_minutes(frame: pd.DataFrame | None) -> pd.Series | None:
        if frame is None:
            return None
        col = pick_first_col(
            frame, ["travel_time_min", "travel_time_minutes", "travel_time", "travel_time_to_work", "time_to_work"]
        )
        if col is None:
            return None
        return pd.to_numeric(frame[col], errors="coerce").dropna()

    def census_travel_pct_from_wide(frame: pd.DataFrame) -> pd.Series | None:
        wide_map = {
            "under_5_minutes_estimate": "<5",
            "5_to_9_minutes_estimate": "5-9",
            "10_to_14_minutes_estimate": "10-14",
            "15_to_19_minutes_estimate": "15-19",
            "20_to_24_minutes_estimate": "20-24",
            "25_to_29_minutes_estimate": "25-29",
            "30_to_34_minutes_estimate": "30-34",
            "35_to_39_minutes_estimate": "35-39",
            "40_to_44_minutes_estimate": "40-44",
            "45_to_59_minutes_estimate": "45-59",
            "60_to_89_minutes_estimate": "60-89",
            "90_minutes_and_over_estimate": "90+",
        }
        cols = [c for c in wide_map if c in frame.columns]
        if not cols:
            return None
        totals = frame[cols].apply(pd.to_numeric, errors="coerce").sum()
        total = totals.sum()
        if total == 0:
            return None
        pct = (totals / total * 100).rename(index=wide_map)
        return pct.reindex(tt_labels).fillna(0)

    def to_pct(series: pd.Series | None) -> pd.Series | None:
        if series is None or series.empty:
            return None
        bins = pd.cut(series, bins=tt_bins, labels=tt_labels, right=False, include_lowest=True)
        return bins.value_counts(normalize=True).reindex(tt_labels).fillna(0) * 100

    calib_pct = to_pct(get_travel_time_minutes(df_calib))
    init_pct = to_pct(get_travel_time_minutes(df_init))

    acs_pct = None
    if tt_df is not None:
        acs_pct = census_travel_pct_from_wide(tt_df)
        if acs_pct is None:
            label_col = pick_first_col(tt_df, ["time_bin", "bin", "label", "travel_time_bin"])
            value_col = pick_first_col(tt_df, ["percent", "share", "pct", "percentage", "value"])
            if label_col is None or value_col is None:
                if tt_df.shape[1] >= 2:
                    label_col, value_col = tt_df.columns[:2]
            if label_col and value_col:
                tmp = tt_df[[label_col, value_col]].copy()
                tmp[label_col] = tmp[label_col].astype(str).str.replace(" ", "", regex=False).str.lower()
                label_map = {
                    "<5": "<5",
                    "5-9": "5-9",
                    "10-14": "10-14",
                    "15-19": "15-19",
                    "20-24": "20-24",
                    "25-29": "25-29",
                    "30-34": "30-34",
                    "35-39": "35-39",
                    "40-44": "40-44",
                    "45-59": "45-59",
                    "60-89": "60-89",
                    "90+": "90+",
                    "90ormore": "90+",
                    "60to89": "60-89",
                    "45to59": "45-59",
                    "40to44": "40-44",
                    "35to39": "35-39",
                    "30to34": "30-34",
                    "25to29": "25-29",
                    "20to24": "20-24",
                    "15to19": "15-19",
                    "10to14": "10-14",
                    "5to9": "5-9",
                }
                tmp["label"] = tmp[label_col].map(label_map)
                tmp = tmp.dropna(subset=["label"])
                acs_pct = tmp.groupby("label")[value_col].mean().reindex(tt_labels).fillna(0).astype(float)

    x = np.arange(len(tt_labels))
    width = 0.25
    plt.figure(figsize=(9, 4.5))
    if init_pct is not None:
        plt.bar(
            x - width,
            init_pct.values,
            width,
            label="Initial",
            color="#4C78A8",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
    if calib_pct is not None:
        plt.bar(
            x,
            calib_pct.values,
            width,
            label="Calibrated",
            color="#E45756",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
    if acs_pct is not None:
        plt.bar(
            x + width,
            acs_pct.values,
            width,
            label="ACS",
            color="#F4A261",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
    plt.xticks(x, tt_labels, rotation=45, ha="right")
    plt.ylabel("Trip distribution (%)")
    plt.xlabel("Travel Time bins (minutes)")
    plt.title("Figure 2: Travel Time Distributions")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figures 1 and 2 from Move-OD outputs.")
    parser.add_argument("--output-root", default="move_OD", help="Root output directory")
    parser.add_argument("--state", default="Tennessee", help="State name")
    parser.add_argument("--county", default="Hamilton", help="County name")
    parser.add_argument("--run-id", default=None, help="Run folder name (e.g., 2025-03-10_2025-03-10)")
    parser.add_argument("--out-dir", default="plots/figures", help="Output directory for figure images")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    run_dir = find_run_dir(output_root, args.state, args.county, args.run_id)
    calib_csv = find_calibrated_csv(run_dir)
    init_path = find_initial_frame(run_dir)
    df = pd.read_csv(calib_csv)
    df_initial = read_table(init_path) if init_path else None

    if init_path is None:
        print("No initial assignment found; Figure 2 will omit the Initial series.")

    census_dir = run_dir / "census_data"
    dept_path = find_census_table(census_dir, "census_depart_times")
    tt_path = find_census_table(census_dir, "travel_time_to_work")
    dept_df = read_table(dept_path) if dept_path else None
    tt_df = read_table(tt_path) if tt_path else None

    if dept_df is None:
        print("No census_depart_times table found; Figure 1 will show generated only.")
    if tt_df is None:
        print("No travel_time_to_work table found; Figure 2 will show generated only.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_path = out_dir / "fig1_departure_time_calibration.png"
    fig2_path = out_dir / "fig2_travel_time_distribution.png"

    plot_departure_calibration(df, dept_df, fig1_path)
    plot_travel_time_distributions(df, df_initial, tt_df, fig2_path)

    print(f"Saved Figure 1 to: {fig1_path}")
    print(f"Saved Figure 2 to: {fig2_path}")


if __name__ == "__main__":
    main()
