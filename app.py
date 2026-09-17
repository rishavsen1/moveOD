import warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

import streamlit as st
import os
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
import zipfile
import io
import json
import pickle
import base64

from generate.lodes_read import LodesGen
from generate.safegraph import Safegraph
from generate.locations_OSM_SG import LocationsOSMSG
from generate.read_ms_buildings import MSBuildings
from generate.lodes_combs import LodesComb
from generate.process_inrix import process_inrix
from generate.safegraph_combs import SgCombs
from generate.union_lodes_sg import union
from generate.generate_routing_df import get_routed, perform_mean_speed_shift
from generate.calibrate_ilp import calibrate_with_ilp
from generate.utils import *

from generate.logger import Logger

import multiprocessing as mp


def serialize_graphs(graphs_dict):
    """Convert NetworkX graphs to JSON-serializable format using pickle+base64."""
    serialized = {}
    for key, graph in graphs_dict.items():
        pickled = pickle.dumps(graph)
        serialized[str(key)] = base64.b64encode(pickled).decode('utf-8')
    return serialized


def deserialize_graphs(serialized_dict):
    """Convert JSON data back to NetworkX graphs from pickle+base64 encoding."""
    deserialized = {}
    for key, encoded_str in serialized_dict.items():
        pickled = base64.b64decode(encoded_str.encode('utf-8'))
        # serialize_graphs stringifies the Timestamp keys; convert them back, or
        # every graph lookup (which is keyed by a floored Timestamp) silently misses.
        deserialized[pd.Timestamp(key)] = pickle.loads(pickled)
    return deserialized


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

# # Ensure the script runs from the base folder of the repository
# base_path = os.path.dirname(os.path.abspath(__file__))
# os.chdir(base_path)
import logging
import warnings

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")


def zip_and_download(output_path):
    # Zip all files in output_path
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_path)
                zipf.write(file_path, arcname=arcname)
    zip_buffer.seek(0)

    # Add download button to Streamlit
    st.download_button(
        label="Download All Calibrated OD Files (ZIP)",
        data=zip_buffer,
        file_name="calibrated_move_od.zip",
        mime="application/zip",
    )


def read_county_geoid_df(county_geoid_path, output_path, logger):
    if os.path.exists(f"{output_path}/county_geoid.geojson"):
        county_geoid_df = gpd.read_file(f"{output_path}/county_geoid.geojson")
        county_geoid_df["intpt"] = county_geoid_df[["INTPTLAT", "INTPTLON"]].apply(lambda p: intpt_func(p), axis=1)
        county_geoid_df["location"] = county_geoid_df.intpt.apply(lambda p: [p.y, p.x])
        success = True
    else:
        county_geoid_df = gpd.read_file(county_geoid_path)
        if not all(col in county_geoid_df.columns for col in ["GEOID", "COUNTYFP", "INTPTLAT", "INTPTLON"]):
            county_geoid_df = county_geoid_df.rename(
                {"GEOID20": "GEOID", "COUNTYFP20": "COUNTYFP", "INTPTLAT20": "INTPTLAT", "INTPTLON20": "INTPTLON"},
                axis=1,
            )

        county_geoid_df = county_geoid_df[["GEOID", "COUNTYFP", "INTPTLAT", "INTPTLON", "geometry"]]

        county_geoid_df.to_file(f"{output_path}/county_geoid.geojson", driver="GeoJSON")

        success = False
        if county_geoid_df.shape[0] > 0:
            success = True

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
        county_lodes_df = county_lodes_df.drop(["GEOID_x", "GEOID_y", "COUNTYFP_x", "COUNTYFP_y"], axis=1)
        success = True
    else:
        lodes_read = LodesGen(county_fips, county_lodes_paths, county_geoid_df, output_path, logger, od_option)
        county_lodes_df, unique_countyfps, success = lodes_read.generate()

    return county_lodes_df, unique_countyfps, success


def read_safegraph_data(county_fips, city, county_geoid_path, safe_df, start_date, end_date, logger, output_path):
    success = False
    if os.path.exists(f"{output_path}/sg_poi_geoids.csv") and os.path.exists(f"{output_path}/sg_visits_by_day.csv"):
        sg_poi_df = pd.read_csv(f"{output_path}/sg_poi_geoids.csv")
        sg_df = pd.read_csv(f"{output_path}/sg_visits_by_day.csv")
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


def read_ms_buildings_data(county_fips, county_geoid_df, logger, output_path):
    success = False
    ms_buldings_path = f"{output_path}/county_buildings_MS.geojson"
    if os.path.exists(ms_buldings_path):
        ms_buildings_df = gpd.read_file(ms_buldings_path)

        ms_buildings_df["geo_centers"] = ms_buildings_df.geometry.centroid
        ms_buildings_df["location"] = ms_buildings_df.geo_centers.apply(lambda p: [p.y, p.x])
        ms_buildings_df = ms_buildings_df[["geometry", "GEOID", "geo_centers", "location"]]

        success = True
    else:
        ms_builds = MSBuildings(county_fips, county_geoid_df, ms_path, output_path, logger)
        ms_buildings_df, success = ms_builds.buildings()

    return ms_buildings_df, success


def read_origin_dest_locations(county_fips, county, county_geoid_df, sg_enabled, output_path, logger, od_option):
    if os.path.exists(f"{output_path}/county_residential_buildings.geojson") and os.path.exists(
        f"{output_path}/county_work_locations.geojson"
    ):
        res_builds = gpd.read_file(f"{output_path}/county_residential_buildings.geojson")
        res_builds["geo_centers"] = res_builds.geometry.centroid
        res_builds["location"] = res_builds.geometry.centroid.apply(lambda p: [p.y, p.x])

        combined_work_locations = gpd.read_file(f"{output_path}/county_work_locations.geojson")
        combined_work_locations["geo_centers"] = combined_work_locations.geometry.centroid
        combined_work_locations["location"] = combined_work_locations.geometry.centroid.apply(lambda p: [p.y, p.x])

        success = True

    else:
        locations = LocationsOSMSG(county_fips, county, county_geoid_df, sg_enabled, output_path, logger, od_option)
        res_builds, combined_work_locations, success = locations.find_locations_OSM()
    return res_builds, combined_work_locations, success


def read_inrix_data(start_date, inrix_path, inrix_conversion_path):
    inrix_df = None
    conversion_df = None
    if not os.path.exists(inrix_path):
        logger.info(f"INRIX data file not found at: {inrix_path}")
        if not os.path.exists(inrix_conversion_path):
            logger.info(f"INRIX conversion file not found at: {inrix_conversion_path}")
    else:
        st.success(f"Creating graph using OSM default speeds")
        inrix_df = pd.read_csv(inrix_path)
        conversion_df = pd.read_csv(inrix_conversion_path)

        desired_date = start_date
        inrix_df["measurement_tstamp"] = pd.to_datetime(inrix_df["measurement_tstamp"])
        inrix_df = inrix_df[inrix_df["measurement_tstamp"].dt.date == desired_date]

    G, hourly_graphs = process_inrix(state, county, inrix_df, conversion_df, start_date)

    return G, hourly_graphs


st.set_page_config(layout="wide")

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "processing_data" not in st.session_state:
    st.session_state.processing_data = {}


def set_dates():
    if len(st.session_state.dates) == 2:
        st.session_state.start_date, st.session_state.end_date = st.session_state.dates
    else:
        # Optionally handle cases where the date range is not valid
        st.error("Please select a valid date range.")


if "start_date" not in st.session_state or "end_date" not in st.session_state or "state_name" not in st.session_state:
    current_date = datetime.datetime.now()
    st.session_state.start_date = current_date.replace(year=2025, month=3, day=10).date()
    st.session_state.end_date = current_date.replace(year=2025, month=3, day=10).date()
    st.session_state.state_name = None

os.makedirs("log_files", exist_ok=True)

st.header("MOVE-OD")

# st.write(
#     "You can find FIPS codes here: [Federal Information Processing System (FIPS) Codes for States and Counties](https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt)"
# )
col1, col2, col3 = st.columns(3)


states, state_fips, counties_in_state, county_fips = get_states_and_counties()
with col1:
    state = st.selectbox("State", options=list(states.keys()), index=42)
with col2:
    county = st.selectbox("County", options=sorted(counties_in_state[state]), index=0)
with col3:
    st.session_state.dates = st.date_input(
        "Enter date range",
        value=(st.session_state.start_date, st.session_state.end_date),
    )
#     county_fips = st.text_input("County's FIPS code", value=(str(county_fips[state, county][0])[-3:]))
city = ""
state_id = states[state]
county_fips = str(county_fips[state, county][0])[-3:]
# col1, col2, col3 = st.columns(3)

# with col1:

set_dates()
start_date = st.session_state.start_date
end_date = st.session_state.end_date

# times = col3.number_input("Choose number of slots to generate for:", value=1, min_value=0, max_value=10)
# start_times = []
# end_times = []

# for time in range(int(times)):
#     col1, col2 = st.columns(2)

#     with col1:
#         start_times.append(st.time_input(f"Start time for slot {time+1}", datetime.time(6, 00)))

#     with col2:
#         end_times.append(st.time_input(f"Enter end time for slot {time+1}", datetime.time(11, 00)))

# for s, e in zip(start_times, end_times):
#     if s >= e:
#         st.error("Start time should be greater than end time")
#         break

year_range = []
if start_date.year == end_date.year:
    year_range.append(str(start_date.year))
else:
    for year in range(start_date.year, end_date.year + 1):
        year_range.append(str(year))

sg_enabled = False
lodes_enabled = False

# choice = col2.multiselect("Choose type of data to generate for:", ["LODES", "Safegraph"], default=["LODES"])
choice = ["LODES"]

col1, col2, col3, col4 = st.columns(4)

# translation = col2.text_input("INRIX translation file", value=f"{inrix_path}/USA_Tennessee.zip")
# segments = col1.text_input(
#     "INRIX translation segments file", value=f"{inrix_path}/USA_TN_OSM_20231201_segments_shapefile.zip"
# )
inrix_folder_path = "./data/inrix"
lodes_year = col1.text_input("Enter LODES data year (Latest year is 2022):", value=("2022"))

tiger_shapefile_year = col2.text_input("Enter TIGER shapefile year (Latest year is 2024):", value=("2024"))
inrix_path = col3.text_input("INRIX data path", value=f"")
inrix_conversion_path = col4.text_input("INRIX conversion path", value=f"")

output_path = f"./move_OD/{state}/{county}/{start_date}_{end_date}"
st.write(f"Output file path: {output_path}")


safe_df = []

col1, col2 = st.columns(2)
ccol1, ccol2 = st.columns(2)
if "Safegraph" in choice:
    sg_enabled = True
    for idx, year in enumerate(year_range):
        safe_df.append(
            st.text_input(
                f"Enter Safegraph parquet file path for {year}",
                value=f"./data/states/{state}/safegraph.parquet/year={year}/region={state}/city={city}/",
            )
        )
else:
    # sg_enabled = col2.checkbox("Use Safegraph data to get additional POI(workplace) locations?")
    sg_enabled = False
    if sg_enabled:
        for idx, year in enumerate(year_range):
            safe_df.append(
                ccol2.text_input(
                    f"Enter Safegraph parquet file path for {year}",
                    value=f"./data/states/{state}/safegraph.parquet/year={year}/region={state}/city={city}/",
                )
            )

ms_enabled = col1.checkbox("Use Global Buildings Footprint data", value=True)

# od_option = st.radio(
#     "Choose an option for OD generation:",
#     (
#         "Origin and Destination in same County",
#         # "Only Origin in County",
#         # "Only Destination in County",
#     ),
# )

od_option = "Origin and Destination in same County"

# st.write("You selected:", od_option)

begin = st.button("BEGIN")

if st.session_state.processing_complete:
    if st.button("🔄 Reset and Start New Process"):
        st.session_state.processing_complete = False
        st.session_state.processing_data = {}
        st.rerun()

calibrated_output_path = ""

if begin:
    st.session_state.processing_complete = False

    with st.spinner("Downloading and gathering files"):
        invalid_path = False
        os.makedirs(output_path, exist_ok=True)

        if "LODES" in choice:
            lodes_enabled = True

        county_lodes_paths = [f"./data/states/{state}/{state_id.lower()}_od_main_JT00_{lodes_year}.csv"]
        if od_option != "Origin and Destination in same County":
            county_lodes_paths.append(f"./data/states/{state}/{state_id.lower()}_od_aux_JT00_{lodes_year}.csv")

        if not invalid_path:

            cpus = os.cpu_count() - 1
            run_lodes_sg_parallel = False
            lodes_cpu_max = 1
            days_count = (end_date - start_date).days

            if cpus > days_count:
                run_lodes_sg_parallel = True
                lodes_cpu_max = days_count
                sg_cpu_max = cpus - days_count

            datetime_ranges = get_datetime_ranges(start_date, end_date, timedelta=15)

            logger = Logger(f"{output_path}/{county}_{state}_{start_date}_{end_date}")

            state_fips_id = state_fips[state]
            # TODO: Change to block
            with st.spinner("Downloading Shapefiles"):
                county_geoid_path = f"./data/states/{state}/tl_{tiger_shapefile_year}_{state_fips_id}_bg.zip"

                if not os.path.exists(county_geoid_path):
                    compressed_path = f"./data/states/{state}/tl_{tiger_shapefile_year}_{state_fips_id}_bg.zip"
                    url = f"https://www2.census.gov/geo/tiger/TIGER{tiger_shapefile_year}/BG/tl_{tiger_shapefile_year}_{state_fips_id}_bg.zip"
                    download_shapefile(
                        logger,
                        url=url,
                        compressed_path=compressed_path,
                    )

            if ms_enabled:
                state_stripped = state.replace(" ", "")
                ms_path = f"./data/states/{state}/{state_stripped}.geojson"
                if not os.path.exists(ms_path):
                    with st.spinner("Downloading Global Buildings Footprint"):
                        download_ms_buildings(logger, state, state_stripped)

            if "LODES" in choice:
                flag = False
                for county_lodes_path in county_lodes_paths:
                    if not os.path.exists(county_lodes_path):
                        flag = True

                if flag:
                    with st.spinner("Downloading LODES files"):
                        download_lodes(
                            logger,
                            state,
                            states[state].lower(),
                            lodes_code=0,
                            year=lodes_year,
                        )

                if not os.path.exists(output_path + "/lodes_combs"):
                    os.makedirs(output_path + "/lodes_combs", exist_ok=True)

            if "Safegraph" in choice:
                if not os.path.exists(output_path + "/safegraph_combs"):
                    os.mkdir(output_path + "/safegraph_combs")

    with st.spinner("Generating ODs"):

        # Read GEOID data
        county_geoid_df, success = read_county_geoid_df(county_geoid_path, output_path, logger)
        if success:
            st.success("GEOID data stored")
        else:
            st.error("GEOID data not processesed")

        # Read LODES data
        county_lodes_df, unique_countyfps, success = read_lodes_df(
            county_fips, county_lodes_paths, county_geoid_df, output_path, logger, od_option
        )
        if success:
            st.success("LODES data filtered")
        else:
            st.error("LODES data not processesed")

        # Modifying GEOID data to keep the relevant counties
        county_geoid_df = county_geoid_df[
            county_geoid_df["COUNTYFP"].astype(str).isin([str(fp) for fp in unique_countyfps])
        ]
        logger.info(f"County has {county_geoid_df.shape[0]} census block groups")

        # Getting safegraph data if enabled
        if sg_enabled:
            sg_poi_df, sg_df, success = read_safegraph_data(
                county_fips, city, county_geoid_path, safe_df, start_date, end_date, logger, output_path
            )

            if success:
                st.success("Safegraph data filtered")
            else:
                st.error("Safegraph data not processesed")

        if ms_enabled:
            ms_buildings_df, success = read_ms_buildings_data(county_fips, county_geoid_df, logger, output_path)
            if success:
                st.success("MS Buildings data filtered")
            else:
                st.error("MS Buildings data not processesed")
        else:
            ms_buildings_df = pd.DataFrame()

        res_locations, combined_work_locations, success = read_origin_dest_locations(
            county_fips, county, county_geoid_df, sg_enabled, output_path, logger, od_option
        )
        if success:
            st.success("Origins and Destinations generated")
        else:
            st.error("Origins and Destinations not processesed")

        # county_geoid, res_locations, combined_work_locations, ms_build, county_lodes, sg = read_data(
        #     output_path=output_path,
        #     lodes=lodes_enabled,
        #     sg_enabled=sg_enabled,
        #     ms_enabled=ms_enabled,
        # )

        logger.info(f"Lodes entries: {county_lodes_df.shape[0]}")
        logger.info(f"Census block groups: {county_geoid_df.shape[0]}")
        logger.info(f"OSM Residential buildings: {res_locations.shape[0]}")
        logger.info(f"OSM Commercial buildings: {combined_work_locations.shape[0]}")

        if ms_enabled:
            logger.info(f"Microsoft Building Footprints buildings: {ms_buildings_df.shape[0]}")

        output_graphs_path = f"{output_path}/hourly_graphs.json"
        if not os.path.exists(output_graphs_path):
            G, hourly_graphs = read_inrix_data(start_date, inrix_path, inrix_conversion_path)
            with open(output_graphs_path, 'w') as f:
                json.dump(serialize_graphs(hourly_graphs), f)
            logger.info(f"Hourly graphs saved to {output_graphs_path}")
        else:
            with open(output_graphs_path, 'r') as f:
                hourly_graphs = deserialize_graphs(json.load(f))
            G = list(hourly_graphs.values())[0]
            logger.info(f"Hourly graphs loaded from {output_graphs_path}")

        success = False
        if "LODES" in choice:
            lodes_combs = LodesComb(
                county_geoid_df,
                output_path,
                ms_enabled,
                datetime_ranges,
                logger,
            )
            lodes_output_dfs, days, travel_time_to_work_df, census_depart_times_df = lodes_combs.main(
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
            if len(lodes_output_dfs) >= 1:
                success = True

        if "Safegraph" in choice and "LODES" in choice:
            days = pd.date_range(start_date, end_date, freq="d").to_list()
            for day in days:
                union(output_path, day, sg_enabled)
            success = True

        if success:
            
            calibrated_output_path = f"{output_path}/calibrated_move_od/"
            os.makedirs(calibrated_output_path, exist_ok=True)

            for day, lodes_output_df in zip(days, lodes_output_dfs):
                os.makedirs(f"{output_path}/intermediate/{day}", exist_ok=True)
                routing_df_output_path = f"{output_path}/intermediate/{day}/routing_df.parquet"
                post_mssr_routing_df_output_path = f"{output_path}/intermediate/{day}/post_mssr_routing_df.parquet"
                calibrated_df_output_path = f"{output_path}/intermediate/{day}/calibrated_df.parquet"
                adjusted_graphs_path = f"{output_path}/intermediate/{day}/hourly_graphs_adjusted.json"

                
                if not os.path.exists(routing_df_output_path):  
                    logger.info("Generating routing df")
                    # getting routed trips and travel times
                    routing_df = get_routed(
                        od_df=lodes_output_df, desired_date=start_date, hourly_graphs_arg=hourly_graphs
                    )

                    routing_df.to_parquet(routing_df_output_path)
                else:
                    logger.info("Reading stored routing df")
                    routing_df = pd.read_parquet(routing_df_output_path)

                if not os.path.exists(post_mssr_routing_df_output_path):
                    logger.info("Generating post mssr routing df")
                    routing_df = pd.read_parquet(routing_df_output_path)

                    # performing mean speed shift and generating new graphs
                    if not os.path.exists(adjusted_graphs_path):
                        hourly_graphs_adjusted = perform_mean_speed_shift(
                            routing_df=routing_df, travel_time_to_work_by_geoid=travel_time_to_work_df,
                            hourly_graphs=hourly_graphs,
                        )
                        with open(adjusted_graphs_path, 'w') as f:
                            json.dump(serialize_graphs(hourly_graphs_adjusted), f)
                        logger.info(f"Adjusted graphs saved to {adjusted_graphs_path}")
                    else:
                        with open(adjusted_graphs_path, 'r') as f:
                            hourly_graphs_adjusted = deserialize_graphs(json.load(f))
                        logger.info(f"Adjusted graphs loaded from {adjusted_graphs_path}")

                    # getting routed trips and travel times post mssr
                    post_mssr_routing_df = get_routed(
                        od_df=lodes_output_df, desired_date=start_date, hourly_graphs_arg=hourly_graphs_adjusted
                    )
                    post_mssr_routing_df.to_parquet(post_mssr_routing_df_output_path)
                
                else:
                    logger.info("Reading stored post mssr routing df")
                    post_mssr_routing_df = pd.read_parquet(post_mssr_routing_df_output_path)
                    if os.path.exists(adjusted_graphs_path):
                        with open(adjusted_graphs_path, 'r') as f:
                            hourly_graphs_adjusted = deserialize_graphs(json.load(f))
                        logger.info(f"Adjusted graphs loaded from {adjusted_graphs_path}")

                # calibrated trips
                calibrated_df = calibrate_with_ilp(
                    lodes_output_df,
                    post_mssr_routing_df,
                    res_locations,
                    combined_work_locations,
                    ms_buildings_df,
                    census_depart_times_df,
                    travel_time_to_work_df,
                )

                # getting travel times for calibrated df
                routing_df = get_routed(
                    od_df=calibrated_df,
                    desired_date=start_date,
                    hourly_graphs=hourly_graphs,
                    post_calibration=True,
                )

                calibrated_df_output_path = f"{calibrated_output_path}/{day}.csv"
                routing_df.to_csv(calibrated_df_output_path)
                st.session_state.calibrated_output_path = calibrated_output_path

            st.success("Calibrated ODs generated")

        if success:
            # Store all necessary data in session state
            st.session_state.processing_complete = True
            st.session_state.processing_data = {
                "res_locations": res_locations,
                "combined_work_locations": combined_work_locations,
                "county_geoid_df": county_geoid_df,
                "logger": logger,
            }

        logger.info("All days generated")

if st.session_state.processing_complete:

    if hasattr(st.session_state, "calibrated_output_path"):
        zip_and_download(st.session_state.calibrated_output_path)
    else:
        st.error("Calibrated output path not found")

    st.success("Processing complete! Displaying origin-destination map...")

    # Retrieve data from session state
    data = st.session_state.processing_data
    res_locations = data["res_locations"]
    combined_work_locations = data["combined_work_locations"]
    county_geoid_df = data["county_geoid_df"]
    logger = data["logger"]

    # Add user choice for what to display
    st.subheader("Origin-Destination Locations Map")

    # Map display options
    col1, col2, col3 = st.columns(3)
    with col1:
        map_choice = st.selectbox(
            "Choose what to display:",
            ["Both Origins and Destinations", "Origins Only", "Destinations Only", "OD Heatmap"],
            index=0,
        )
    with col2:
        sample_size = st.slider("Sample size (for performance)", min_value=100, max_value=5000, value=1000, step=100)
    with col3:
        if map_choice == "OD Heatmap":
            heatmap_type = st.selectbox("Heatmap type:", ["Point Density", "Hexagonal Binning"], index=0)

    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        origins_count = len(res_locations) if not res_locations.empty else 0
        st.metric("Total Origins", origins_count)
    with col2:
        destinations_count = len(combined_work_locations) if not combined_work_locations.empty else 0
        st.metric("Total Destinations", destinations_count)
    with col3:
        total_locations = origins_count + destinations_count
        st.metric("Total Locations", total_locations)

    # Folium map with color differentiation and heatmap
    try:
        import folium
        from streamlit_folium import st_folium
        from folium.plugins import HeatMap, MarkerCluster
        import numpy as np

        # Prepare data based on user choice
        origins_sample = pd.DataFrame()
        destinations_sample = pd.DataFrame()

        if not res_locations.empty and map_choice in ["Both Origins and Destinations", "Origins Only", "OD Heatmap"]:
            origins_df = res_locations.copy()
            origins_df["lat"] = origins_df["location"].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) >= 2 else None
            )
            origins_df["lon"] = origins_df["location"].apply(
                lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else None
            )
            origins_sample = origins_df.dropna(subset=["lat", "lon"]).sample(
                n=min(sample_size, len(origins_df)), random_state=42
            )

        if not combined_work_locations.empty and map_choice in [
            "Both Origins and Destinations",
            "Destinations Only",
            "OD Heatmap",
        ]:
            destinations_df = combined_work_locations.copy()
            destinations_df["lat"] = destinations_df["location"].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) >= 2 else None
            )
            destinations_df["lon"] = destinations_df["location"].apply(
                lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else None
            )
            destinations_sample = destinations_df.dropna(subset=["lat", "lon"]).sample(
                n=min(sample_size, len(destinations_df)), random_state=42
            )

        # Calculate center for folium map
        all_points = []
        if not origins_sample.empty:
            all_points.extend(origins_sample[["lat", "lon"]].values.tolist())
        if not destinations_sample.empty:
            all_points.extend(destinations_sample[["lat", "lon"]].values.tolist())

        if all_points:
            center_lat = np.mean([p[0] for p in all_points])
            center_lon = np.mean([p[1] for p in all_points])

            # Create folium map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

            if map_choice == "OD Heatmap":
                # Create heatmap/hexplot
                if heatmap_type == "Point Density":
                    # Combined heatmap of all locations
                    heat_data = all_points
                    if heat_data:
                        HeatMap(
                            heat_data,
                            radius=15,
                            blur=10,
                            max_zoom=1,
                            gradient={0.4: "blue", 0.6: "cyan", 0.7: "lime", 0.8: "yellow", 1.0: "red"},
                        ).add_to(m)

                        # Add separate heatmaps for origins and destinations with different colors
                        if not origins_sample.empty:
                            origins_heat = origins_sample[["lat", "lon"]].values.tolist()
                            HeatMap(
                                origins_heat,
                                radius=12,
                                blur=8,
                                max_zoom=1,
                                gradient={0.4: "darkgreen", 0.6: "green", 0.8: "lightgreen", 1.0: "lime"},
                                name="Origins Density",
                            ).add_to(m)

                        if not destinations_sample.empty:
                            dest_heat = destinations_sample[["lat", "lon"]].values.tolist()
                            HeatMap(
                                dest_heat,
                                radius=12,
                                blur=8,
                                max_zoom=1,
                                gradient={0.4: "darkred", 0.6: "red", 0.8: "orange", 1.0: "yellow"},
                                name="Destinations Density",
                            ).add_to(m)

                elif heatmap_type == "Hexagonal Binning":
                    # Use MarkerCluster as a substitute for hexagonal clustering
                    if not origins_sample.empty:
                        origins_cluster = MarkerCluster(name="Origins Cluster")
                        for _, row in origins_sample.iterrows():
                            folium.Marker(
                                location=[row["lat"], row["lon"]],
                                icon=folium.Icon(color="green"),
                                popup="Origin (Residential)",
                            ).add_to(origins_cluster)
                        origins_cluster.add_to(m)

                    if not destinations_sample.empty:
                        dest_cluster = MarkerCluster(name="Destinations Cluster")
                        for _, row in destinations_sample.iterrows():
                            folium.Marker(
                                location=[row["lat"], row["lon"]],
                                icon=folium.Icon(color="red"),
                                popup="Destination (Workplace)",
                            ).add_to(dest_cluster)
                        dest_cluster.add_to(m)

                # Add layer control
                folium.LayerControl().add_to(m)

                st.write("**Heatmap**: Density visualization of origins and destinations")

            else:
                # Regular point markers
                if map_choice in ["Both Origins and Destinations", "Origins Only"] and not origins_sample.empty:
                    for _, row in origins_sample.iterrows():
                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]],
                            radius=3,
                            color="green",
                            fillColor="green",
                            fillOpacity=0.7,
                            popup="Origin (Residential)",
                            weight=1,
                        ).add_to(m)

                if (
                    map_choice in ["Both Origins and Destinations", "Destinations Only"]
                    and not destinations_sample.empty
                ):
                    for _, row in destinations_sample.iterrows():
                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]],
                            radius=3,
                            color="red",
                            fillColor="red",
                            fillOpacity=0.7,
                            popup="Destination (Workplace)",
                            weight=1,
                        ).add_to(m)

                # Legend
                if map_choice == "Both Origins and Destinations":
                    st.write("🟢 Green: Origins (Residential) | 🔴 Red: Destinations (Workplace)")
                elif map_choice == "Origins Only":
                    st.write("🟢 Green: Origins (Residential)")
                else:
                    st.write("🔴 Red: Destinations (Workplace)")

            # Display the folium map
            # Make the map span the width of the page dynamically
            st_folium(m, use_container_width=True, height=600)

            # Additional information
            if map_choice == "OD Heatmap":
                st.info(f"Showing {heatmap_type.lower()} for {len(all_points)} total locations")
            else:
                displayed_origins = len(origins_sample) if not origins_sample.empty else 0
                displayed_destinations = len(destinations_sample) if not destinations_sample.empty else 0
                st.info(f"Displaying {displayed_origins} origins and {displayed_destinations} destinations")

        else:
            st.warning("No valid location data found for mapping")

    except ImportError as e:
        st.error(f"Missing required packages. Please install: `pip install folium streamlit-folium`")
        st.error(f"Error: {e}")

    # Optional: Add county boundary information
    if not county_geoid_df.empty:
        with st.expander("County Information"):
            try:
                bounds = county_geoid_df.total_bounds  # [minx, miny, maxx, maxy]
                st.write(f"**County bounds**: {bounds}")
                st.write(f"**Census block groups**: {len(county_geoid_df)}")
                st.write(f"**GENERATION COMPLETE**")
            except Exception as e:
                logger.warning(f"Could not display county boundary: {e}")
