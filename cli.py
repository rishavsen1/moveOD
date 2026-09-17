#!/usr/bin/env python3
"""
MOVE-OD Command Line Interface

A CLI alternative to the Streamlit app (app.py) for running OD generation
without risk of disconnection. Safe to run inside screen/tmux sessions.

Usage examples:
    # Interactive mode (prompts for all inputs):
    python cli.py

    # Fully specified via command-line arguments:
    python cli.py --state Tennessee --county Hamilton \
        --start-date 2025-03-10 --end-date 2025-03-10 \
        --lodes-year 2022 --tiger-year 2024 --ms-buildings

    # Resume a previous run (skips already completed steps):
    python cli.py --state Tennessee --county Hamilton \
        --start-date 2025-03-10 --end-date 2025-03-10

    # With INRIX data:
    python cli.py --state Tennessee --county Hamilton \
        --start-date 2025-03-10 --end-date 2025-03-10 \
        --inrix-path ./data/inrix/data.csv \
        --inrix-conversion-path ./data/inrix/conversion.csv

    # With Safegraph data:
    python cli.py --state Tennessee --county Hamilton \
        --start-date 2025-03-10 --end-date 2025-03-10 \
        --safegraph --safegraph-paths ./data/sg/2025/
"""

import warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

import argparse
import os
import sys
import datetime
import json
import pickle
import base64
import multiprocessing as mp
import gc

import pandas as pd
import numpy as np
import geopandas as gpd

from generate.lodes_read import LodesGen
from generate.safegraph import Safegraph
from generate.locations_OSM_SG import LocationsOSMSG
from generate.read_ms_buildings import MSBuildings
from generate.lodes_combs import LodesComb
from generate.process_inrix import process_inrix
from generate.safegraph_combs import SgCombs
from generate.union_lodes_sg import union
from generate.generate_routing_df import (
    get_routed,
    perform_mean_speed_shift,
    mean_speed_shift_is_uniform,
    rescale_routing_df,
)
from generate.utils import calculate_speed_shift
from generate.calibrate_ilp import calibrate_with_ilp
from generate.utils import (
    get_states_and_counties,
    get_datetime_ranges,
    download_shapefile,
    download_lodes,
    download_ms_buildings,
    intpt_func,
)
from generate.logger import Logger


# ---------------------------------------------------------------------------
# Serialization helpers (same as app.py)
# ---------------------------------------------------------------------------
def serialize_graphs(graphs_dict):
    """Serialise the hourly graphs, storing each distinct graph only once.

    Many timestamps share the same graph object (peak / off-peak), so pickling
    per timestamp would write the same bytes dozens of times.
    """
    slot_of = {}
    graphs = {}
    keys = {}
    for key, graph in graphs_dict.items():
        gid = id(graph)
        if gid not in slot_of:
            slot_of[gid] = str(len(slot_of))
            graphs[slot_of[gid]] = base64.b64encode(pickle.dumps(graph)).decode("utf-8")
        keys[str(key)] = slot_of[gid]
    return {"format": "shared-v1", "graphs": graphs, "keys": keys}


def deserialize_graphs(serialized_dict):
    """Inverse of serialize_graphs; also reads the older one-blob-per-key format.

    Keys are restored as Timestamps: lookups use dep_time.floor(TIME_INTERVAL),
    so leaving them as strings makes every lookup miss.
    """
    if serialized_dict.get("format") == "shared-v1":
        loaded = {
            slot: pickle.loads(base64.b64decode(blob.encode("utf-8")))
            for slot, blob in serialized_dict["graphs"].items()
        }
        return {pd.Timestamp(key): loaded[slot] for key, slot in serialized_dict["keys"].items()}

    # Legacy format: {timestamp_str: base64_pickle}
    return {
        pd.Timestamp(key): pickle.loads(base64.b64decode(blob.encode("utf-8")))
        for key, blob in serialized_dict.items()
    }


# ---------------------------------------------------------------------------
# Data-reading helpers (mirrored from app.py, minus Streamlit calls)
# ---------------------------------------------------------------------------
def read_county_geoid_df(county_geoid_path, output_path, logger):
    if os.path.exists(f"{output_path}/county_geoid.geojson"):
        county_geoid_df = gpd.read_file(f"{output_path}/county_geoid.geojson")
        county_geoid_df["intpt"] = county_geoid_df[["INTPTLAT", "INTPTLON"]].apply(
            lambda p: intpt_func(p), axis=1
        )
        county_geoid_df["location"] = county_geoid_df.intpt.apply(lambda p: [p.y, p.x])
        success = True
    else:
        county_geoid_df = gpd.read_file(county_geoid_path)
        if not all(
            col in county_geoid_df.columns
            for col in ["GEOID", "COUNTYFP", "INTPTLAT", "INTPTLON"]
        ):
            county_geoid_df = county_geoid_df.rename(
                {
                    "GEOID20": "GEOID",
                    "COUNTYFP20": "COUNTYFP",
                    "INTPTLAT20": "INTPTLAT",
                    "INTPTLON20": "INTPTLON",
                },
                axis=1,
            )
        county_geoid_df = county_geoid_df[
            ["GEOID", "COUNTYFP", "INTPTLAT", "INTPTLON", "geometry"]
        ]
        county_geoid_df.to_file(
            f"{output_path}/county_geoid.geojson", driver="GeoJSON"
        )
        success = county_geoid_df.shape[0] > 0

    return county_geoid_df, success


def read_lodes_df(county_fips, county_lodes_paths, county_geoid_df, output_path, logger, od_option):
    success = False
    if os.path.exists(f"{output_path}/county_lodes.parquet"):
        county_lodes_df = pd.read_parquet(f"{output_path}/county_lodes.parquet")
        unique_countyfps = list(
            set(county_lodes_df["COUNTYFP_x"].astype(str).unique()).union(
                set(county_lodes_df["COUNTYFP_y"].astype(str).unique())
            )
        )
        county_lodes_df = county_lodes_df.drop(
            ["GEOID_x", "GEOID_y", "COUNTYFP_x", "COUNTYFP_y"], axis=1
        )
        success = True
    else:
        lodes_read = LodesGen(
            county_fips, county_lodes_paths, county_geoid_df, output_path, logger, od_option
        )
        county_lodes_df, unique_countyfps, success = lodes_read.generate()

    return county_lodes_df, unique_countyfps, success


def read_safegraph_data(
    county_fips, city, county_geoid_path, safe_df, start_date, end_date, logger, output_path
):
    success = False
    if os.path.exists(f"{output_path}/sg_poi_geoids.csv") and os.path.exists(
        f"{output_path}/sg_visits_by_day.csv"
    ):
        sg_poi_df = pd.read_csv(f"{output_path}/sg_poi_geoids.csv")
        sg_df = pd.read_csv(f"{output_path}/sg_visits_by_day.csv")
        success = True
    else:
        safegraph = Safegraph(
            county_fips,
            city,
            county_geoid_path,
            safe_df,
            output_path,
            start_date,
            end_date,
            logger,
        )
        sg_poi_df, success1 = safegraph.get_sg_poi()
        sg_df, success2 = safegraph.get_day_of_week()
        success = success1 and success2

    return sg_poi_df, sg_df, success


def read_ms_buildings_data(county_fips, county_geoid_df, ms_path, logger, output_path):
    success = False
    ms_buildings_path = f"{output_path}/county_buildings_MS.geojson"
    if os.path.exists(ms_buildings_path):
        ms_buildings_df = gpd.read_file(ms_buildings_path)
        ms_buildings_df["geo_centers"] = ms_buildings_df.geometry.centroid
        ms_buildings_df["location"] = ms_buildings_df.geo_centers.apply(
            lambda p: [p.y, p.x]
        )
        ms_buildings_df = ms_buildings_df[
            ["geometry", "GEOID", "geo_centers", "location"]
        ]
        success = True
    else:
        ms_builds = MSBuildings(
            county_fips, county_geoid_df, ms_path, output_path, logger
        )
        ms_buildings_df, success = ms_builds.buildings()

    return ms_buildings_df, success


def read_origin_dest_locations(
    county_fips, county, county_geoid_df, sg_enabled, output_path, logger, od_option
):
    if os.path.exists(
        f"{output_path}/county_residential_buildings.geojson"
    ) and os.path.exists(f"{output_path}/county_work_locations.geojson"):
        res_builds = gpd.read_file(
            f"{output_path}/county_residential_buildings.geojson"
        )
        res_builds["geo_centers"] = res_builds.geometry.centroid
        res_builds["location"] = res_builds.geometry.centroid.apply(
            lambda p: [p.y, p.x]
        )

        combined_work_locations = gpd.read_file(
            f"{output_path}/county_work_locations.geojson"
        )
        combined_work_locations["geo_centers"] = (
            combined_work_locations.geometry.centroid
        )
        combined_work_locations["location"] = (
            combined_work_locations.geometry.centroid.apply(lambda p: [p.y, p.x])
        )
        success = True
    else:
        locations = LocationsOSMSG(
            county_fips,
            county,
            county_geoid_df,
            sg_enabled,
            output_path,
            logger,
            od_option,
        )
        res_builds, combined_work_locations, success = locations.find_locations_OSM()

    return res_builds, combined_work_locations, success


def read_inrix_data(state, county, start_date, inrix_path, inrix_conversion_path, logger):
    inrix_df = None
    conversion_df = None
    if inrix_path and os.path.exists(inrix_path):
        inrix_df = pd.read_csv(inrix_path)
        conversion_df = pd.read_csv(inrix_conversion_path) if inrix_conversion_path else None

        inrix_df["measurement_tstamp"] = pd.to_datetime(inrix_df["measurement_tstamp"])
        inrix_df = inrix_df[inrix_df["measurement_tstamp"].dt.date == start_date]
    else:
        if inrix_path:
            logger.info(f"INRIX data file not found at: {inrix_path}")
        logger.info("Creating graph using OSM default speeds")

    G, hourly_graphs = process_inrix(state, county, inrix_df, conversion_df, start_date)
    return G, hourly_graphs


# ---------------------------------------------------------------------------
# Interactive prompts (used when CLI args are not supplied)
# ---------------------------------------------------------------------------
def interactive_select(prompt, options, default=None):
    """Let the user pick from a numbered list."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " [default]" if opt == default else ""
        print(f"  {i}. {opt}{marker}")

    while True:
        raw = input("Enter number (or press Enter for default): ").strip()
        if raw == "" and default is not None:
            return default
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print("Invalid choice, try again.")


def interactive_input(prompt, default=None):
    """Simple text prompt with optional default."""
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else (str(default) if default is not None else "")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="MOVE-OD: Origin-Destination generation (CLI mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--state", type=str, help="State name (e.g. Tennessee)")
    parser.add_argument("--county", type=str, help="County name (e.g. Hamilton)")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (default: 2025-03-10)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (default: same as start-date)",
    )
    parser.add_argument(
        "--lodes-year",
        type=str,
        default="2022",
        help="LODES data year (default: 2022)",
    )
    parser.add_argument(
        "--tiger-year",
        type=str,
        default="2024",
        help="TIGER shapefile year (default: 2024)",
    )
    parser.add_argument(
        "--ms-buildings",
        action="store_true",
        default=True,
        help="Use Global Buildings Footprint data (default: True)",
    )
    parser.add_argument(
        "--no-ms-buildings",
        action="store_true",
        help="Disable Global Buildings Footprint data",
    )
    parser.add_argument(
        "--safegraph",
        action="store_true",
        default=False,
        help="Enable Safegraph data",
    )
    parser.add_argument(
        "--safegraph-paths",
        nargs="*",
        default=[],
        help="Safegraph parquet file paths (one per year in date range)",
    )
    parser.add_argument("--inrix-path", type=str, default="", help="INRIX data CSV path")
    parser.add_argument(
        "--inrix-conversion-path", type=str, default="", help="INRIX conversion CSV path"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Force interactive mode even if some args are provided",
    )
    return parser


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    state,
    county,
    start_date,
    end_date,
    lodes_year,
    tiger_shapefile_year,
    ms_enabled,
    sg_enabled,
    safe_df,
    inrix_path,
    inrix_conversion_path,
):
    """Core pipeline — equivalent to pressing BEGIN in the Streamlit app."""

    states, state_fips, counties_in_state, county_fips_map = get_states_and_counties()

    # Validate state / county
    if state not in states:
        print(f"[ERROR] Unknown state: {state}")
        print(f"Available states: {', '.join(sorted(states.keys()))}")
        sys.exit(1)
    if county not in counties_in_state[state]:
        print(f"[ERROR] Unknown county '{county}' in {state}")
        print(f"Available counties: {', '.join(sorted(counties_in_state[state]))}")
        sys.exit(1)

    state_id = states[state]
    county_fips = str(county_fips_map[state, county][0])[-3:]
    state_fips_id = state_fips[state]

    od_option = "Origin and Destination in same County"

    output_path = f"./move_OD/{state}/{county}/{start_date}_{end_date}"
    os.makedirs(output_path, exist_ok=True)
    print(f"Output path: {output_path}")

    logger = Logger(f"{output_path}/{county}_{state}_{start_date}_{end_date}")

    datetime_ranges = get_datetime_ranges(start_date, end_date, timedelta=15)

    # ------------------------------------------------------------------
    # 1. Download shapefiles
    # ------------------------------------------------------------------
    print("\n[1/9] Downloading shapefiles...")
    county_geoid_path = (
        f"./data/states/{state}/tl_{tiger_shapefile_year}_{state_fips_id}_bg.zip"
    )
    if not os.path.exists(county_geoid_path):
        url = f"https://www2.census.gov/geo/tiger/TIGER{tiger_shapefile_year}/BG/tl_{tiger_shapefile_year}_{state_fips_id}_bg.zip"
        download_shapefile(logger, url=url, compressed_path=county_geoid_path)
    else:
        logger.info(f"Shapefile already exists: {county_geoid_path}")

    # ------------------------------------------------------------------
    # 2. Download MS buildings if enabled
    # ------------------------------------------------------------------
    ms_path = None
    if ms_enabled:
        print("[2/9] Downloading Global Buildings Footprint...")
        state_stripped = state.replace(" ", "")
        ms_path = f"./data/states/{state}/{state_stripped}.geojson"
        if not os.path.exists(ms_path):
            download_ms_buildings(logger, state, state_stripped)
        else:
            logger.info(f"MS Buildings file already exists: {ms_path}")
    else:
        print("[2/9] MS Buildings disabled, skipping.")

    # ------------------------------------------------------------------
    # 3. Download LODES files
    # ------------------------------------------------------------------
    print("[3/9] Downloading LODES files...")
    county_lodes_paths = [
        f"./data/states/{state}/{state_id.lower()}_od_main_JT00_{lodes_year}.csv"
    ]
    if od_option != "Origin and Destination in same County":
        county_lodes_paths.append(
            f"./data/states/{state}/{state_id.lower()}_od_aux_JT00_{lodes_year}.csv"
        )

    flag = any(not os.path.exists(p) for p in county_lodes_paths)
    if flag:
        download_lodes(logger, state, state_id.lower(), lodes_code=0, year=lodes_year)
    else:
        logger.info("LODES files already present.")

    os.makedirs(f"{output_path}/lodes_combs", exist_ok=True)
    if sg_enabled:
        os.makedirs(f"{output_path}/safegraph_combs", exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Read GEOID data
    # ------------------------------------------------------------------
    print("[4/9] Reading GEOID data...")
    county_geoid_df, success = read_county_geoid_df(
        county_geoid_path, output_path, logger
    )
    if success:
        logger.info("GEOID data stored")
    else:
        logger.error("GEOID data not processed — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Read LODES data
    # ------------------------------------------------------------------
    print("[5/9] Reading LODES data...")
    county_lodes_df, unique_countyfps, success = read_lodes_df(
        county_fips, county_lodes_paths, county_geoid_df, output_path, logger, od_option
    )
    if success:
        logger.info("LODES data filtered")
    else:
        logger.error("LODES data not processed — aborting.")
        sys.exit(1)

    county_geoid_df = county_geoid_df[
        county_geoid_df["COUNTYFP"].astype(str).isin([str(fp) for fp in unique_countyfps])
    ]
    logger.info(f"County has {county_geoid_df.shape[0]} census block groups")

    # ------------------------------------------------------------------
    # 6. Safegraph (optional)
    # ------------------------------------------------------------------
    if sg_enabled:
        print("[6/9] Reading Safegraph data...")
        sg_poi_df, sg_df, success = read_safegraph_data(
            county_fips,
            "",
            county_geoid_path,
            safe_df,
            start_date,
            end_date,
            logger,
            output_path,
        )
        if success:
            logger.info("Safegraph data filtered")
        else:
            logger.error("Safegraph data not processed — aborting.")
            sys.exit(1)
    else:
        print("[6/9] Safegraph disabled, skipping.")

    # ------------------------------------------------------------------
    # 7. MS Buildings data
    # ------------------------------------------------------------------
    if ms_enabled:
        print("[7/9] Reading MS Buildings data...")
        ms_buildings_df, success = read_ms_buildings_data(
            county_fips, county_geoid_df, ms_path, logger, output_path
        )
        if success:
            logger.info("MS Buildings data filtered")
        else:
            logger.error("MS Buildings data not processed — aborting.")
            sys.exit(1)
    else:
        print("[7/9] MS Buildings disabled, skipping.")
        ms_buildings_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # 8. Origin / Destination locations
    # ------------------------------------------------------------------
    print("[8/9] Generating origins and destinations...")
    res_locations, combined_work_locations, success = read_origin_dest_locations(
        county_fips, county, county_geoid_df, sg_enabled, output_path, logger, od_option
    )
    if success:
        logger.info("Origins and Destinations generated")
    else:
        logger.error("Origins and Destinations not processed — aborting.")
        sys.exit(1)

    logger.info(f"Lodes entries: {county_lodes_df.shape[0]}")
    logger.info(f"Census block groups: {county_geoid_df.shape[0]}")
    logger.info(f"OSM Residential buildings: {res_locations.shape[0]}")
    logger.info(f"OSM Commercial buildings: {combined_work_locations.shape[0]}")
    if ms_enabled:
        logger.info(f"Microsoft Building Footprints buildings: {ms_buildings_df.shape[0]}")

    # ------------------------------------------------------------------
    # 9. Build road graphs, generate ODs, calibrate
    # ------------------------------------------------------------------
    print("[9/9] Building road graphs & generating calibrated ODs...")

    output_graphs_path = f"{output_path}/hourly_graphs.json"

    def _ensure_hourly_graphs_loaded(current_graphs):
        if current_graphs is not None:
            return current_graphs
        with open(output_graphs_path, "r") as f:
            loaded = deserialize_graphs(json.load(f))
        logger.info(f"Hourly graphs loaded from {output_graphs_path}")
        return loaded

    if not os.path.exists(output_graphs_path):
        G, hourly_graphs = read_inrix_data(
            state, county, start_date, inrix_path, inrix_conversion_path, logger
        )
        with open(output_graphs_path, "w") as f:
            json.dump(serialize_graphs(hourly_graphs), f)
        logger.info(f"Hourly graphs saved to {output_graphs_path}")
    else:
        hourly_graphs = _ensure_hourly_graphs_loaded(None)
        G = list(hourly_graphs.values())[0]

    # LODES combinations
    lodes_combs = LodesComb(
        county_geoid_df,
        output_path,
        ms_enabled,
        datetime_ranges,
        logger,
    )
    lodes_output_dfs, days, travel_time_to_work_df, census_depart_times_df = (
        lodes_combs.main(
            county_geoid_df,
            res_locations,
            combined_work_locations,
            ms_buildings_df,
            county_lodes_df,
            state_fips_id,
            county_fips,
            G,
            hourly_graphs,
            block_groups="*",
        )
    )

    if len(lodes_output_dfs) < 1:
        logger.error("LODES combination generated 0 output dataframes — aborting.")
        sys.exit(1)

    gc.collect()

    success = True

    if sg_enabled:
        days_range = pd.date_range(start_date, end_date, freq="d").to_list()
        for day in days_range:
            union(output_path, day, sg_enabled)

    # Calibration loop
    calibrated_output_path = f"{output_path}/calibrated_move_od/"
    os.makedirs(calibrated_output_path, exist_ok=True)

    for day_idx, (day, lodes_output_df) in enumerate(zip(days, lodes_output_dfs)):
        logger.info(f"Processing day {day_idx + 1}/{len(days)}: {day}")

        os.makedirs(f"{output_path}/intermediate/{day}", exist_ok=True)
        routing_df_output_path = f"{output_path}/intermediate/{day}/routing_df.parquet"
        post_mssr_routing_df_output_path = (
            f"{output_path}/intermediate/{day}/post_mssr_routing_df.parquet"
        )
        adjusted_graphs_path = (
            f"{output_path}/intermediate/{day}/hourly_graphs_adjusted.json"
        )

        # Initial routing
        if not os.path.exists(routing_df_output_path):
            logger.info("Generating routing df")
            hourly_graphs = _ensure_hourly_graphs_loaded(hourly_graphs)
            routing_df = get_routed(
                od_df=lodes_output_df,
                desired_date=start_date,
                hourly_graphs_arg=hourly_graphs,
            )
            routing_df.to_parquet(routing_df_output_path)
        else:
            logger.info("Reading stored routing df")
            routing_df = pd.read_parquet(routing_df_output_path)

        # Post-MSSR routing
        if not os.path.exists(post_mssr_routing_df_output_path):
            logger.info("Generating post mssr routing df")
            hourly_graphs = _ensure_hourly_graphs_loaded(hourly_graphs)

            if mean_speed_shift_is_uniform(hourly_graphs):
                # Every edge is scaled by the same factor, so the shortest paths
                # are unchanged: rescale the routed frame instead of routing the
                # whole county a second time.
                psi = calculate_speed_shift(routing_df, travel_time_to_work_df)
                logger.info(
                    f"Uniform mean speed shift (psi={psi:.4f}); rescaling routed "
                    "trips instead of re-routing"
                )
                post_mssr_routing_df = rescale_routing_df(routing_df, psi)
            else:
                # INRIX edges keep their measured speeds, so the shift is not
                # uniform and the routes really can change: re-route.
                if not os.path.exists(adjusted_graphs_path):
                    hourly_graphs_adjusted = perform_mean_speed_shift(
                        routing_df=routing_df,
                        travel_time_to_work_by_geoid=travel_time_to_work_df,
                        hourly_graphs=hourly_graphs,
                    )
                    with open(adjusted_graphs_path, "w") as f:
                        json.dump(serialize_graphs(hourly_graphs_adjusted), f)
                    logger.info(f"Adjusted graphs saved to {adjusted_graphs_path}")
                else:
                    with open(adjusted_graphs_path, "r") as f:
                        hourly_graphs_adjusted = deserialize_graphs(json.load(f))
                    logger.info(f"Adjusted graphs loaded from {adjusted_graphs_path}")

                post_mssr_routing_df = get_routed(
                    od_df=lodes_output_df,
                    desired_date=start_date,
                    hourly_graphs_arg=hourly_graphs_adjusted,
                )
                del hourly_graphs_adjusted
                gc.collect()

            post_mssr_routing_df.to_parquet(post_mssr_routing_df_output_path)
        else:
            logger.info("Reading stored post mssr routing df")
            post_mssr_routing_df = pd.read_parquet(post_mssr_routing_df_output_path)

        if "routing_df" in locals():
            del routing_df
            gc.collect()

        try:
            lodes_mb = lodes_output_df.memory_usage(deep=True).sum() / 1e6
            post_mssr_mb = post_mssr_routing_df.memory_usage(deep=True).sum() / 1e6
            logger.info(f"Pre-calibration memory: lodes_output_df={lodes_mb:.0f} MB, post_mssr_routing_df={post_mssr_mb:.0f} MB")
        except Exception:
            logger.info("Pre-calibration memory: unable to compute DataFrame memory usage")

        # Calibration
        logger.info("Starting calibration...")
        calibrated_df = calibrate_with_ilp(
            lodes_output_df,
            post_mssr_routing_df,
            res_locations,
            combined_work_locations,
            ms_buildings_df,
            census_depart_times_df,
            travel_time_to_work_df,
        )
        
        # Free post-MSSR routing data and adjusted graphs after calibration
        del post_mssr_routing_df
        if "hourly_graphs_adjusted" in locals():
            del hourly_graphs_adjusted
        gc.collect()
        logger.info("Freed post-calibration memory")

        # NOTE: a final post-calibration routing pass used to run here, but its
        # result was discarded without being read -- roughly a third of all
        # routing work for no output. The calibrated frame already carries
        # travel_time_min and departure times. Re-add it only with a consumer.
        if "hourly_graphs" in locals() and hourly_graphs is not None:
            del hourly_graphs
        hourly_graphs = None
        gc.collect()
        logger.info("Freed routing memory")

        calibrated_df_out = f"{calibrated_output_path}/{day}.csv"
        calibrated_df.to_csv(calibrated_df_out)
        logger.info(f"Calibrated OD saved to {calibrated_df_out}")

    print(f"\n{'='*60}")
    print(f"  DONE — Calibrated ODs written to: {calibrated_output_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if __name__ == "__main__":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

    parser = build_parser()
    args = parser.parse_args()

    # Determine whether we need interactive mode
    need_interactive = args.interactive or (args.state is None or args.county is None)

    states_data, state_fips, counties_in_state, county_fips_map = get_states_and_counties()

    if need_interactive:
        print("=" * 60)
        print("  MOVE-OD  —  Command Line Interface")
        print("=" * 60)

        # --- State ---
        state = args.state
        if not state:
            sorted_states = sorted(states_data.keys())
            state = interactive_select("Select a state:", sorted_states, default="Tennessee")

        # --- County ---
        county = args.county
        if not county:
            sorted_counties = sorted(counties_in_state[state])
            county = interactive_select(
                f"Select a county in {state}:", sorted_counties
            )

        # --- Dates ---
        default_start = args.start_date or "2025-03-10"
        start_str = interactive_input("Start date (YYYY-MM-DD)", default=default_start)
        default_end = args.end_date or start_str
        end_str = interactive_input("End date (YYYY-MM-DD)", default=default_end)

        # --- LODES year ---
        lodes_year = interactive_input("LODES data year", default=args.lodes_year)

        # --- TIGER year ---
        tiger_year = interactive_input("TIGER shapefile year", default=args.tiger_year)

        # --- MS Buildings ---
        ms_input = interactive_input("Use Global Buildings Footprint? (y/n)", default="y")
        ms_enabled = ms_input.lower().startswith("y")

        # --- Safegraph ---
        sg_input = interactive_input("Use Safegraph data? (y/n)", default="n")
        sg_enabled = sg_input.lower().startswith("y")

        safe_df = []
        if sg_enabled:
            start_date_obj = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
            end_date_obj = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
            years = range(start_date_obj.year, end_date_obj.year + 1)
            for year in years:
                default_sg = f"./data/states/{state}/safegraph.parquet/year={year}/region={state}/city=/"
                path = interactive_input(f"Safegraph path for {year}", default=default_sg)
                safe_df.append(path)

        # --- INRIX ---
        inrix_path = interactive_input("INRIX data CSV path (leave blank to skip)", default="")
        inrix_conversion_path = ""
        if inrix_path:
            inrix_conversion_path = interactive_input("INRIX conversion CSV path", default="")

    else:
        state = args.state
        county = args.county
        start_str = args.start_date or "2025-03-10"
        end_str = args.end_date or start_str
        lodes_year = args.lodes_year
        tiger_year = args.tiger_year
        ms_enabled = args.ms_buildings and not args.no_ms_buildings
        sg_enabled = args.safegraph
        safe_df = list(args.safegraph_paths) if args.safegraph_paths else []
        inrix_path = args.inrix_path
        inrix_conversion_path = args.inrix_conversion_path

    # Parse dates
    start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

    # Summary
    print(f"\n{'─'*60}")
    print(f"  State:          {state}")
    print(f"  County:         {county}")
    print(f"  Date range:     {start_date} → {end_date}")
    print(f"  LODES year:     {lodes_year}")
    print(f"  TIGER year:     {tiger_year}")
    print(f"  MS Buildings:   {'yes' if ms_enabled else 'no'}")
    print(f"  Safegraph:      {'yes' if sg_enabled else 'no'}")
    print(f"  INRIX path:     {inrix_path or '(none)'}")
    print(f"{'─'*60}\n")

    run_pipeline(
        state=state,
        county=county,
        start_date=start_date,
        end_date=end_date,
        lodes_year=lodes_year,
        tiger_shapefile_year=tiger_year,
        ms_enabled=ms_enabled,
        sg_enabled=sg_enabled,
        safe_df=safe_df,
        inrix_path=inrix_path,
        inrix_conversion_path=inrix_conversion_path,
    )


if __name__ == "__main__":
    main()
