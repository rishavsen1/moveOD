import json
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import re
from shapely.geometry import Point, LineString
import math
import datetime
import ast
import requests
import gzip
import os
import shutil
from tqdm import tqdm
import zipfile
import subprocess
from io import StringIO
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import hashlib


from generate.config import CENSUS_API_KEY

KM_TO_MILES = 0.621371


# Function to create a unique hash for the current inrix_path
def generate_hash_for_path(path):
    return hashlib.md5(path.encode()).hexdigest()


def save_graph(graph, path):
    nx.write_graphml(graph, path)
    print(f"Graph saved to {path}")


def load_graph(path):
    graph = nx.read_graphml(path)
    print(f"Graph loaded from {path}")
    return graph


def intpt_func(row):
    return Point(row["INTPTLON"], row["INTPTLAT"])


def func_home_pt(row):
    return Point(row.home_loc_lon, row.home_loc_lat)


def func_work_pt(row):
    return Point(row.work_loc_lon, row.work_loc_lat)


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def write_to_file(output_path, file_path, file_name, df):
    df.to_csv(f"{output_path}/{file_path}/{file_name}", index=False)


def sample_rows(grouped_df, sample_size):
    """Efficiently sample rows based on the 'total_jobs' distribution."""
    # Calculate cumulative sum of jobs and total jobs
    total_jobs = grouped_df["total_jobs"].sum()
    probabilities = grouped_df["total_jobs"] / total_jobs

    # Sample the data
    chosen_indices = np.random.choice(grouped_df.index, size=sample_size, replace=True, p=probabilities)
    sampled_df = grouped_df.loc[chosen_indices].copy()
    sampled_df["sampled_jobs"] = 1  # Each row contributes one job in the sample

    # Aggregate sampled data
    sampled_df = sampled_df.groupby(level=[0, 1]).sum().reset_index()
    return sampled_df


def marginal_dist(df, origin_col, dest_col, sample_size):
    """Create a subsampled DataFrame based on marginal distributions."""
    # Group by origin and destination, aggregating total_jobs
    grouped_df = df.groupby([origin_col, dest_col])["total_jobs"].sum().reset_index()
    grouped_df.set_index([origin_col, dest_col], inplace=True)

    # Handle case where sample_size exceeds the total available jobs
    if sample_size > grouped_df["total_jobs"].sum():
        sample_size = int(grouped_df["total_jobs"].sum())

    # Sample the rows according to the total_jobs distribution
    sampled_df = sample_rows(grouped_df, sample_size)

    return sampled_df


def combine_date_time(date, time):
    return datetime.datetime.combine(date, time)


def get_datetime_ranges(start_date, end_date, timedelta):
    # Creating the list of lists of datetime objects
    datetime_ranges = []
    current_date = start_date
    while current_date <= end_date:
        # for start_time, end_time in zip(start_times, end_times):
        #     start_datetime = combine_date_time(current_date, start_time)
        #     end_datetime = combine_date_time(current_date, end_time)

        #     # Adjust if end_datetime is before start_datetime (crossing midnight)
        #     if end_datetime < start_datetime:
        #         end_datetime += datetime.timedelta(days=1)

        # datetime_range = [
        #     start_datetime + datetime.timedelta(seconds=x * timedelta)
        #     for x in range(0, int((end_datetime - start_datetime).total_seconds() / timedelta))
        # ]
        datetime_range = current_date
        datetime_ranges.append(datetime_range)
        current_date = current_date + datetime.timedelta(days=1)

    return datetime_ranges


def fetch_and_parse_fips():
    # URL of the FIPS code file
    url = "https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt"

    # Fetch the data
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
    else:
        raise Exception(f"Failed to retrieve data: Status code {response.status_code}")

    data = response.text.split("\n")[13:]  # Adjust the number based on the actual header line count
    data = "\n".join(data)
    # Define column specifications according to the fixed-width format of the file
    col_specification = [
        (0, 2),  # State FIPS (this seems to be slightly off based on your output)
        (4, 90),  # Rest of the line, which seems to include state name and possibly county dataz
    ]

    # Read the fixed-width file
    fips_df = pd.read_fwf(StringIO(data), colspecs=col_specification, header=None)
    fips_df.columns = ["State FIPS", "Data"]

    return fips_df


def get_fips_codes(state_abbreviation, county_name):
    # Fetch and parse the FIPS data
    fips_df = fetch_and_parse_fips()

    # Query the DataFrame for state FIPS code
    state_fips = fips_df[fips_df["State Abbreviation"] == state_abbreviation]["State FIPS"].unique()

    if not state_fips:
        return None, None

    # Query the DataFrame for county FIPS code
    county_fips = fips_df[
        (fips_df["State Abbreviation"] == state_abbreviation)
        & (fips_df["County Name"].str.startswith(county_name, na=False))
    ]["County FIPS"].values

    if county_fips.size > 0:
        return state_fips[0], county_fips[0]
    else:
        return state_fips[0], None


def read_data(output_path, lodes=False, sg_enabled=False, ms_enabled=False):
    print("Reading data")
    # loading geometry data
    county_geoid = pd.read_csv(f"{output_path}/county_geoid.csv")
    county_geoid = gpd.GeoDataFrame(county_geoid, geometry=gpd.GeoSeries.from_wkt(county_geoid.geometry))
    county_geoid.GEOID = county_geoid.GEOID.astype(str)

    # loading residential buildings
    res_build = pd.read_csv(
        f"{output_path}/county_residential_buildings.csv",
    )
    res_build = gpd.GeoDataFrame(res_build, geometry=gpd.GeoSeries.from_wkt(res_build.geometry))
    res_build["location"] = res_build.geometry.apply(lambda p: [p.y, p.x])
    res_build.GEOID = res_build.GEOID.astype(str)

    # loading work buildings
    com_build = pd.read_csv(
        f"{output_path}/county_work_locations.csv",
    )

    com_build = gpd.GeoDataFrame(com_build, geometry=gpd.GeoSeries.from_wkt(com_build.geometry))
    com_build["location"] = com_build.geometry.apply(lambda p: [p.y, p.x])
    com_build = com_build.reset_index()
    com_build.GEOID = com_build.GEOID.astype(str)

    # loading all buildings (MS dataset)
    if ms_enabled:
        ms_build = pd.read_csv(f"{output_path}/county_buildings_MS.csv")
        ms_build = gpd.GeoDataFrame(ms_build, geometry=gpd.GeoSeries.from_wkt(ms_build.geo_centers))
        ms_build.GEOID = ms_build.GEOID.astype(str)
        ms_build["location"] = ms_build.geometry.apply(lambda p: [p.y, p.x])
    else:
        ms_build = pd.DataFrame()

    sg = pd.DataFrame()
    county_lodes = pd.DataFrame()

    if sg_enabled:
        sg = pd.read_csv(f"{output_path}/sg_visits_by_day.csv")
        sg["home_geoid"] = sg["home_geoid"].astype(str)
        sg["poi_geoid"] = sg["poi_geoid"].astype(str)
        # marginal_dist(sg, "home_geoid", "poi_geoid", sample_size)

    if lodes:
        county_lodes = pd.read_csv(
            f"{output_path}/county_lodes.csv",
            dtype={"TRACTCE20_home": "string", "TRACTCE20_work": "string"},
        )
        county_lodes["h_geocode"] = county_lodes["h_geocode"].astype(str)
        county_lodes["w_geocode"] = county_lodes["w_geocode"].astype(str)
        # marginal_dist(county_lodes, "h_geocode", "w_geocode", sample_size)

    return county_geoid, res_build, com_build, ms_build, county_lodes, sg


# def state_abbreviation_to_full(name):
#     state_name = None
#     if name:
#         name_to_abbreviation = {value: key for key, value in states.items()}
#         print(name_to_abbreviation)
#         state_name = name_to_abbreviation.get(name, "Invalid state abbreviation")
#     return state_name


def get_states_and_counties():
    df = pd.read_csv("./data/uscounties.csv", dtype={"county_fips": "str"})
    states = df.groupby("state_name").first()["state_id"].to_dict()
    counties_in_state = df.groupby("state_name")["county"].apply(list).to_dict()
    state_fips = df.groupby("state_name").first()["county_fips"].apply(lambda x: str(x)[:2]).to_dict()
    county_fips = df.groupby(["state_name", "county"])["county_fips"].apply(list).to_dict()
    return states, state_fips, counties_in_state, county_fips


def download_shapefile(logger, url, compressed_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(compressed_path), exist_ok=True)
        with open(compressed_path, "wb") as f:
            f.write(response.content)
        logger.info(f"File downloaded and saved as: {compressed_path}")

    else:
        logger.error(f"Failed to download the shapefile: Status code {response.status_code}")


def download_lodes(logger, state, state_abbr, lodes_code, year):
    type = "gzip"

    url = f"https://lehd.ces.census.gov/data/lodes/LODES8/{state_abbr}/od/{state_abbr}_od_main_JT0{lodes_code}_{year}.csv.gz"
    compressed_path = f"./data/states/{state}/{state_abbr}_od_main_JT0{lodes_code}_{year}.csv.gz"
    decompressed_path = f"./data/states/{state}/{state_abbr}_od_main_JT0{lodes_code}_{year}.csv"
    download_and_decompress(type, logger, url, compressed_path, decompressed_path)

    url = f"https://lehd.ces.census.gov/data/lodes/LODES8/{state_abbr}/od/{state_abbr}_od_aux_JT0{lodes_code}_{year}.csv.gz"
    compressed_path = f"./data/states/{state}/{state_abbr}_od_aux_JT0{lodes_code}_{year}.csv.gz"
    decompressed_path = f"./data/states/{state}/{state_abbr}_od_aux_JT0{lodes_code}_{year}.csv"
    download_and_decompress(type, logger, url, compressed_path, decompressed_path)


def download_ms_buildings(logger, state, state_stripped):
    type = "zip"
    url = f"https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/{state_stripped}.geojson.zip"
    compressed_path = f"./data/states/{state}/{state_stripped}.geojson.zip"
    decompressed_path = f"./data/states/{state}/"  # Path to the directory to extract files
    download_and_decompress(type, logger, url, compressed_path, decompressed_path)


def download_and_decompress(type, logger, url, compressed_path, decompressed_path):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024

        os.makedirs(os.path.dirname(compressed_path), exist_ok=True)
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        # Fix: Write the response content properly chunk by chunk
        with open(compressed_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)  # Write the actual data chunk, not the entire content
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            logger.error(f"Something went wrong!")
        logger.info(f"File downloaded and saved as: {compressed_path}")

        if type == "gzip":
            try:
                with gzip.open(compressed_path, "rb") as f_in:
                    with open(decompressed_path.rstrip("/"), "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                logger.info(f"File decompressed and saved as: {decompressed_path.rstrip('/')}")
            except gzip.BadGzipFile as e:
                logger.error(f"Error decompressing file: {e}. The file might not be a valid gzip file.")
                return False

        elif type == "zip":
            with zipfile.ZipFile(compressed_path, "r") as zip_ref:
                zip_ref.extractall(decompressed_path)
            logger.info(f"Files decompressed and saved in directory: {decompressed_path}")

        os.remove(compressed_path)
        logger.info(f"Removed the compressed file: {compressed_path}")
        return True

    else:
        logger.error(f"Failed to download the LODES file: Status code {response.status_code}")
        return False


def run_script(script_path, logger):
    with subprocess.Popen([script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        # Read the output line by line as it is produced
        for line in proc.stdout:
            print(line, end="")  # Print to console in real-time
            logger.info(line.strip())  # Log each line

        # Check for errors and log them
        for line in proc.stderr:
            logger.error(line.strip())

    # Wait for the process to finish and get the exit code
    proc.wait()
    return proc.returncode


def traveltimeDrive(row):
    # latStart, lonStart, latDest, lonDest, time, mode_type = (
    #     row["origin_loc_lat"],
    #     row["origin_loc_lon"],
    #     row["dest_loc_lat"],
    #     row["dest_loc_lon"],
    #     str(row["departure_time_str"]),
    #     row["mode_type"],
    # )
    latStart, lonStart, latDest, lonDest, mode_type = (
        row["origin_loc_lat"],
        row["origin_loc_lon"],
        row["dest_loc_lat"],
        row["dest_loc_lon"],
        # str(row["departure_time_str"]),
        row["mode_type"],
    )
    time = "00:00:00"
    modes = {"drive": "CAR", "transit": "WALK,TRANSIT", "walk": "WALK"}
    mode = modes[mode_type]
    request_url = f"http://localhost:8080/otp/routers/default/plan?fromPlace={latStart},{lonStart}&toPlace={latDest},{lonDest}&mode={mode}&date=01-26-2024&time={time}&maxWalkDistance=100000"

    res_dict = {
        "origin_loc_lat": latStart,
        "origin_loc_lon": lonStart,
        "dest_loc_lat": latDest,
        "dest_loc_lon": lonDest,
        "error": "",  # include an error field in your result dictionary
    }

    try:
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            try:
                if "plan" in data and "itineraries" in data["plan"]:
                    distance = sum(leg["distance"] for leg in data["plan"]["itineraries"][0]["legs"])
                    res_dict.update(
                        {"transit_time": int(data["plan"]["itineraries"][0]["duration"]), "distance_m": distance}
                    )
            except (KeyError, TypeError) as e:
                res_dict["error"] = "Error parsing response data: {}".format(e)
        else:
            res_dict["error"] = f"HTTP error {response.status_code}"

    except requests.exceptions.RequestException as e:
        res_dict["error"] = f"Request failed: {e}"

    return res_dict


def process_rows_drive(mode_type, dataframe):
    data_list = dataframe
    if not isinstance(dataframe, dict):
        data_list = dataframe.to_dict("records")
        # for data in data_list:
        data_list.update({"mode_type": mode_type})
        # delayed_tasks = [delayed(traveltimeDrive)(data) for data in data_list]

        # results = []
        # for result in tqdm(compute(*delayed_tasks, scheduler="threads"), total=len(data_list)):
        # results.append(result)
        result = [traveltimeDrive(data_list)]
    else:
        data_list.update({"mode_type": mode_type})
        result = [traveltimeDrive(data_list)]

    return result


def get_travel_time(mode_type, trips_df):
    results = process_rows_drive(mode_type, trips_df)
    for res in results:
        matching_rows = trips_df[
            (trips_df["origin_loc_lat"] == res["origin_loc_lat"])
            & (trips_df["origin_loc_lon"] == res["origin_loc_lon"])
            & (trips_df["dest_loc_lat"] == res["dest_loc_lat"])
            & (trips_df["dest_loc_lon"] == res["dest_loc_lon"])
        ]

        if not matching_rows.empty:
            for index in matching_rows.index:
                if "transit_time" in res:
                    trips_df.loc[index, f"{mode_type}_time"] = res["transit_time"]
                    trips_df.loc[index, "total_distance"] = res["distance_m"]
                    trips_df.loc[index, "distance_miles"] = res["distance_m"] / 1000 * KM_TO_MILES
                else:
                    trips_df.loc[index, f"{mode_type}_time"] = None
                    trips_df.loc[index, "total_distance"] = None
                    trips_df.loc[index, "distance_miles"] = None

    return trips_df


def get_travel_time_dict(mode_type, trips_dict):
    results = process_rows_drive(mode_type, trips_dict)
    result_dict = {}
    for res in results:
        if "transit_time" in res:
            result_dict[f"{mode_type}_time"] = res["transit_time"]
            result_dict["total_distance"] = res["distance_m"]
            result_dict["distance_miles"] = res["distance_m"] / 1000 * KM_TO_MILES
        else:
            result_dict[f"{mode_type}_time"] = None
            result_dict["total_distance"] = None
            result_dict["distance_miles"] = None

    return result_dict


def get_census_data(api_key, api_url, table_name, state_fips, county_fips, block_groups, county_only):
    url = api_url

    # Use anonymous access if key is not a real key
    use_key = api_key and not api_key.startswith("NO_KEY") and not api_key.startswith("YOUR_")

    if county_only:
        params = {
            "get": f"NAME,group({table_name})",
            "for": f"county:{county_fips}",
            "in": f"state:{state_fips}",
        }
    else:
        params = {
            "get": f"NAME,group({table_name})",
            "for": f"block group:{block_groups}",
            "in": f"state:{state_fips} county:{county_fips}",
        }

    if use_key:
        params["key"] = api_key

    response = requests.get(url, params=params)
    # Never print the API key: redact it from the logged URL
    print(re.sub(r"([?&]key=)[^&]*", r"\1<redacted>", response.url))

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    else:
        print("Failed to retrieve data:", response.status_code)


def get_census_data_wrapper(
    table,
    api_url,
    state_fips,
    county_fips,
    block_groups,
    county_only,
):
    # tables = 'B08134'
    # table = "B08302"
    # api_url = "https://api.census.gov/data/2022/acs/acs5"

    # tables = 'P034'
    # api_url = "https://api.census.gov/data/2000/dec/sf3"

    df = get_census_data(CENSUS_API_KEY, api_url, table, state_fips, county_fips, block_groups, county_only)

    columns_to_be_renamed = {
        f"{table}_001E": "total_estimate",
        f"{table}_001M": "total_margin",
        f"{table}_002E": "12am_to_4:59am_estimate",
        f"{table}_002M": "12am_to_4:59am_margin",
        f"{table}_003E": "5am_to_5:29am_estimate",
        f"{table}_003M": "5am_to_5:29am_margin",
        f"{table}_004E": "5:30am_to_5:59am_estimate",
        f"{table}_004M": "5:30am_to_5:59am_margin",
        f"{table}_005E": "6am_to_6:29am_estimate",
        f"{table}_005M": "6am_to_6:29am_margin",
        f"{table}_006E": "6:30am_to_6:59am_estimate",
        f"{table}_006M": "6:30am_to_6:59am_margin",
        f"{table}_007E": "7am_to_7:29am_estimate",
        f"{table}_007M": "7am_to_7:29am_margin",
        f"{table}_008E": "7:30am_to_7:59am_estimate",
        f"{table}_008M": "7:30am_to_7:59am_margin",
        f"{table}_009E": "8am_to_8:29am_estimate",
        f"{table}_009M": "8am_to_8:29am_margin",
        f"{table}_010E": "8:30am_to_8:59am_estimate",
        f"{table}_010M": "8:30am_to_8:59am_margin",
        f"{table}_011E": "9am_to_9:59am_estimate",
        f"{table}_011M": "9am_to_9:59am_margin",
        f"{table}_012E": "10am_to_10:59am_estimate",
        f"{table}_012M": "10am_to_10:59am_margin",
        f"{table}_013E": "11am_to_11:59am_estimate",
        f"{table}_013M": "11am_to_11:59am_margin",
        f"{table}_014E": "12pm_to_3:59pm_estimate",
        f"{table}_014M": "12pm_to_3:59pm_margin",
        f"{table}_015E": "4pm_to_11:59pm_estimate",
        f"{table}_015M": "4pm_to_11:59pm_margin",
    }

    other_cols = ["GEO_ID", "state", "county", "tract", "block group"]
    if county_only:
        other_cols = ["GEO_ID", "state", "county"]

    df_renamed = df[list(columns_to_be_renamed.keys()) + other_cols].rename(columns_to_be_renamed, axis=1)
    df_renamed["GEO_ID"] = df_renamed["GEO_ID"].apply(lambda x: x.split("US")[1])
    df_renamed["total_estimate"] = df_renamed["total_estimate"].astype(int)

    for column in df_renamed.columns:
        if "estimate" in column or "margin" in column:
            df_renamed[column] = df_renamed[column].astype("int")

    return df_renamed


def get_census_work_time(
    table,
    api_url,
    state_fips,
    county_fips,
    block_groups,
    county_only,
):

    df = get_census_data(CENSUS_API_KEY, api_url, table, state_fips, county_fips, block_groups, county_only)

    columns_to_be_renamed = {
        f"{table}_001E": "total_estimate",
        f"{table}_001M": "total_margin",
    }

    other_cols = ["GEO_ID", "state", "county", "tract", "block group"]
    if county_only:
        other_cols = ["GEO_ID", "state", "county"]

    df_renamed = df[list(columns_to_be_renamed.keys()) + other_cols].rename(columns_to_be_renamed, axis=1)
    df_renamed["GEO_ID"] = df_renamed["GEO_ID"].apply(lambda x: x.split("US")[1])
    df_renamed["total_estimate"] = df_renamed["total_estimate"].astype(float)

    for column in df_renamed.columns:
        if "estimate" in column or "margin" in column:
            df_renamed[column] = df_renamed[column].astype(float)

    return df_renamed


def get_census_travel_time_data(
    table="B08302",
    api_url="https://api.census.gov/data/2022/acs/acs5",
    state_fips="47",
    county_fips="065",
    block_groups="*",
    county_only=False,
):
    # tables = 'B08134'
    # table = "B08302"
    # api_url = "https://api.census.gov/data/2022/acs/acs5"

    # tables = 'P034'
    # api_url = "https://api.census.gov/data/2000/dec/sf3"

    df = get_census_data(CENSUS_API_KEY, api_url, table, state_fips, county_fips, block_groups, county_only)

    columns_to_be_renamed = {
        f"{table}_001E": "total_estimate",
        f"{table}_001M": "total_margin",
        f"{table}_002E": "under_5_minutes_estimate",
        f"{table}_002M": "under_5_minutes_margin",
        f"{table}_003E": "5_to_9_minutes_estimate",
        f"{table}_003M": "5_to_9_minutes_margin",
        f"{table}_004E": "10_to_14_minutes_estimate",
        f"{table}_004M": "10_to_14_minutes_margin",
        f"{table}_005E": "15_to_19_minutes_estimate",
        f"{table}_005M": "15_to_19_minutes_margin",
        f"{table}_006E": "20_to_24_minutes_estimate",
        f"{table}_006M": "20_to_24_minutes_margin",
        f"{table}_007E": "25_to_29_minutes_estimate",
        f"{table}_007M": "25_to_29_minutes_margin",
        f"{table}_008E": "30_to_34_minutes_estimate",
        f"{table}_008M": "30_to_34_minutes_margin",
        f"{table}_009E": "35_to_39_minutes_estimate",
        f"{table}_009M": "35_to_39_minutes_margin",
        f"{table}_010E": "40_to_44_minutes_estimate",
        f"{table}_010M": "40_to_44_minutes_margin",
        f"{table}_011E": "45_to_59_minutes_estimate",
        f"{table}_011M": "45_to_59_minutes_margin",
        f"{table}_012E": "60_to_89_minutes_estimate",
        f"{table}_012M": "60_to_89_minutes_margin",
        f"{table}_013E": "90_minutes_and_over_estimate",
        f"{table}_013M": "90_minutes_and_over_margin",
    }

    df_renamed = df[list(columns_to_be_renamed.keys()) + ["GEO_ID", "state", "county"]].rename(
        columns_to_be_renamed, axis=1
    )
    df_renamed["GEO_ID"] = df_renamed["GEO_ID"].apply(lambda x: x.split("US")[1])
    df_renamed["total_estimate"] = df_renamed["total_estimate"].astype(int)

    for column in df_renamed.columns:
        if "estimate" in column or "margin" in column:
            df_renamed[column] = df_renamed[column].astype("int")

    return df_renamed


def time_to_datetime(time_str):
    return datetime.datetime.strptime(time_str, "%I:%M %p")


def parse_time_range(day, time_str):
    # base_date = day
    base_date = datetime.date(year=2024, month=5, day=1)

    time_range = time_str.replace("_estimate", "").split("_to_")
    start_time_str, end_time_str = time_range[0], time_range[1]

    def get_time(t_str):
        period = "AM" if "am" in t_str else "PM"
        t_str = t_str.replace("am", "").replace("pm", "")

        if ":" not in t_str:
            t_str += ":00"

        return datetime.datetime.strptime(t_str + " " + period, "%I:%M %p").time()

    start_time = get_time(start_time_str)
    end_time = get_time(end_time_str)

    start_datetime = datetime.datetime.combine(base_date, start_time)
    end_datetime = datetime.datetime.combine(base_date, end_time)

    if end_datetime <= start_datetime:
        end_datetime += datetime.timedelta(days=1)

    return start_datetime, end_datetime


def calculate_heading(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def heading_to_direction(heading):
    """
    Convert a heading in degrees to a cardinal direction.
    """
    directions = ["N", "E", "S", "W"]
    index = round(heading / 90) % 4
    return directions[index]


def convert_path_str_to_list(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def add_segment_headings_and_directions(df, graph):
    df = convert_path_str_to_list(df, "shortest_path")

    segment_headings = []
    segment_directions = []

    for index, path in enumerate(df["shortest_path"]):
        headings = []
        directions = []
        if path is not None:
            for u, v in zip(path[:-1], path[1:]):
                try:
                    u_lat, u_lon = graph.nodes[u]["y"], graph.nodes[u]["x"]
                    v_lat, v_lon = graph.nodes[v]["y"], graph.nodes[v]["x"]
                    heading = calculate_heading(u_lat, u_lon, v_lat, v_lon)
                    headings.append(heading)
                    direction = heading_to_direction(heading)
                    directions.append(direction)
                except KeyError as e:
                    print(f"KeyError: {e} for nodes {u}, {v} at index {index}")
                    print(f"Path: {path}")
                    print(f"Graph Nodes: {list(graph.nodes)[:10]} ...")
                    headings.append(None)
                    directions.append(None)
        else:
            print(f"No path found for index {index}")

        segment_headings.append(headings)
        segment_directions.append(directions)

    df["segment_headings"] = segment_headings
    df["segment_directions"] = segment_directions
    return df


def get_nearest_node(graph, lat, lon):
    try:
        return ox.distance.nearest_nodes(graph, lon, lat)
    except Exception as e:
        print(f"Error finding nearest node for coordinates ({lat}, {lon}): {e}")
        return None


def calculate_shortest_path(graph, origin_lat, origin_lon, destination_lat, destination_lon):
    origin_node = get_nearest_node(graph, origin_lat, origin_lon)
    destination_node = get_nearest_node(graph, destination_lat, destination_lon)
    shortest_path = None
    try:
        shortest_path = nx.shortest_path(graph, origin_node, destination_node, weight="length")
    except:
        return None
    return shortest_path


def parse_speed_limit(speed_str):
    if "mph" in speed_str or "mi/h" in speed_str:
        speed_value = re.search(r"\d+", speed_str)
        if speed_value:
            return float(speed_value.group()) * 0.44704
        speed_value = re.search(r"\d+", speed_str)
        if speed_value:
            return float(speed_value.group()) * 1000 / 3600
    else:
        speed_value = re.search(r"\d+", speed_str)
        if speed_value:
            return float(speed_value.group()) * 1000 / 3600
    return None


def preprocess_inrix_data(inrix_data):
    inrix_dict = {}
    for idx, row in inrix_data.iterrows():
        key = (row["OSMWayIDs"], row["bearing"], row["measurement_tstamp"])
        inrix_dict[key] = (row["speed"], row["historical_average_speed"])
    return inrix_dict


def fetch_inrix_speed(osm_way_id, heading, direction, timestamp, inrix_dict):
    if not isinstance(osm_way_id, list):
        osm_way_id = [osm_way_id]  # Ensure osm_way_id is a list

    # Iterate through all osm_way_ids to find the matching data
    for way_id in osm_way_id:
        key = (way_id, direction, timestamp)
        if key in inrix_dict:
            inrix_segment = inrix_dict[key]
            return inrix_segment["speed"], inrix_segment["historical_average_speed"]

    return None, None  # Handle missing data


def convert_speed_to_mps(speed, unit="mph"):
    # Convert speed to meters per second
    if unit == "mph":
        return speed * 0.44704
    elif unit == "kph":
        return speed * 1000 / 3600
    else:
        return speed


def calculate_average_speed_at_timestamp(timestamp, inrix_data, target_col):
    # Filter the INRIX data for the given timestamp
    timestamp_data = inrix_data[inrix_data["measurement_tstamp"] == timestamp]
    if not timestamp_data.empty:
        average_speed = timestamp_data[target_col].mean()
        return average_speed
    else:
        return None  # Handle missing data


def calculate_shortest_path_time_and_distance(graph, origin_lat, origin_lon, destination_lat, destination_lon):
    origin_node = get_nearest_node(graph, origin_lat, origin_lon)
    destination_node = get_nearest_node(graph, destination_lat, destination_lon)
    if origin_node is None or destination_node is None:
        return None, None, None

    shortest_path = None
    travel_time = None
    total_distance = None
    try:
        shortest_path = nx.shortest_path(graph, origin_node, destination_node, weight="length")
        # Calculate travel time and distance based on edge lengths and speed limits
        travel_time = 0
        total_distance = 0
        for u, v in zip(shortest_path[:-1], shortest_path[1:]):
            edge_data = min(graph.get_edge_data(u, v).values(), key=lambda x: x["length"])
            length = edge_data["length"]  # in meters
            total_distance += length  # Accumulate total distance
            # Get the speed limit, if available, otherwise use a default speed (e.g., 50 km/h)
            speed_str = edge_data.get("maxspeed", "50 km/h")
            if isinstance(speed_str, list):  # Handle cases where speed limit is a list
                speed_str = speed_str[0]
            speed_mps = parse_speed_limit(speed_str)
            if speed_mps is None:
                speed_mps = 50 * 1000 / 3600  # Default speed if parsing fails, assumed in km/h
            travel_time += length / speed_mps
    except Exception as e:
        print(
            f"Error calculating shortest path from ({origin_lat}, {origin_lon}) to ({destination_lat}, {destination_lon}): {e}"
        )
        return None, None, None
    return shortest_path, travel_time, total_distance


def add_shortest_path_time_and_distance_columns(df, graph):
    shortest_paths = []
    travel_times = []
    travel_distances = []
    for index, row in df.iterrows():
        path, time, distance = calculate_shortest_path_time_and_distance(
            graph, row["origin_loc_lat"], row["origin_loc_lon"], row["dest_loc_lat"], row["dest_loc_lon"]
        )
        shortest_paths.append(path)
        travel_times.append(time)
        travel_distances.append(distance)
    df["shortest_path"] = shortest_paths
    df["travel_time_secs"] = travel_times
    df["travel_distance_m"] = travel_distances
    return df


def find_shortest_path(G, row):
    orig_lat, orig_lon = row["origin_loc_lat"], row["origin_loc_lon"]
    dest_lat, dest_lon = row["dest_loc_lat"], row["dest_loc_lon"]

    orig_node = ox.distance.nearest_nodes(G, orig_lon, orig_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lon, dest_lat)

    try:
        shortest_path = nx.shortest_path(G, orig_node, dest_node, weight="length")
    except nx.NetworkXNoPath:
        shortest_path = []  # or use None or any other sentinel value
    return shortest_path


def find_shortest_path_dict(G, dict):
    orig_lat, orig_lon = dict["origin_loc_lat"], dict["origin_loc_lon"]
    dest_lat, dest_lon = dict["dest_loc_lat"], dict["dest_loc_lon"]

    orig_node = ox.distance.nearest_nodes(G, orig_lon, orig_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lon, dest_lat)

    try:
        shortest_path = nx.shortest_path(G, orig_node, dest_node, weight="length")
    except nx.NetworkXNoPath:
        shortest_path = []  # or use None or any other sentinel value
    return shortest_path


def calculate_shortest_path_with_speeds(
    graph,
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    shortest_path,
    timestamp,
    inrix_dict,
    average_speed_mps,
    average_speed_historical_mps,
):
    # start_time = time.time()

    if shortest_path == []:
        return None, None, None, None, None

    travel_time_osm = None
    travel_time_inrix = None
    total_distance = None
    travel_time_historical = None
    # total_time_taken = 0
    try:

        travel_time_osm = 0
        travel_time_inrix = 0
        travel_time_historical = 0
        total_distance = 0
        for u, v in zip(shortest_path[:-1], shortest_path[1:]):
            # edge_data_start_time = time.time()
            edge_data = graph.get_edge_data(u, v)
            # edge_data_end_time = time.time()
            # print(f"Time to get edge data for ({u}, {v}): {edge_data_end_time - edge_data_start_time:.8f} seconds")
            # total_time_taken += edge_data_end_time - edge_data_start_time
            if not edge_data:
                print(f"Missing edge data for edge ({u}, {v})")
                continue

            edge_data = min(edge_data.values(), key=lambda x: x.get("length", float("inf")))
            length = edge_data.get("length", 0)  # in meters
            total_distance += length
            osm_way_id = edge_data.get("osmid", None)
            u_lat, u_lon = graph.nodes[u]["y"], graph.nodes[u]["x"]
            v_lat, v_lon = graph.nodes[v]["y"], graph.nodes[v]["x"]

            # heading_start_time = time.time()
            heading = calculate_heading(u_lat, u_lon, v_lat, v_lon)
            direction = heading_to_direction(heading)
            # heading_end_time = time.time()
            # print(f"Time to calculate heading and direction: {heading_end_time - heading_start_time:.8f} seconds")

            # total_time_taken += heading_end_time - heading_start_time

            if osm_way_id is not None:
                # fetch_speed_start_time = time.time()
                speed_inrix, speed_inrix_historical = fetch_inrix_speed(
                    osm_way_id, heading, direction, timestamp, inrix_dict
                )
                # fetch_speed_end_time = time.time()
                # print(f"Time to fetch INRIX speed: {fetch_speed_end_time - fetch_speed_start_time:.8f} seconds")
                # total_time_taken += fetch_speed_end_time - fetch_speed_start_time

                if speed_inrix is not None:
                    speed_inrix_mps = convert_speed_to_mps(speed_inrix, "mph")
                    speed_inrix_historical = convert_speed_to_mps(speed_inrix_historical, "mph")
                else:
                    speed_inrix_mps = average_speed_mps
                    speed_inrix_historical = average_speed_historical_mps
            else:
                speed_inrix_mps = average_speed_mps
                speed_inrix_historical = average_speed_historical_mps

            travel_time_inrix += length / speed_inrix_mps
            travel_time_historical += length / speed_inrix_historical

            # Uncomment if needed for OSM maxspeed
            # speed_str = edge_data.get("speed_kph", "55 km/h")
            # if isinstance(speed_str, list):  # Handle cases where speed limit is a list
            #     speed_str = speed_str[0]
            # speed_osm_mps = parse_speed_limit(speed_str)
            # if speed_osm_mps is None:
            #     speed_osm_mps = 50 * 1000 / 3600  # Default speed if parsing fails, assumed in km/h
            # travel_time_osm += length / speed_osm_mps

    except Exception as e:
        print(
            f"Error calculating shortest path from ({origin_lat}, {origin_lon}) to ({destination_lat}, {destination_lon}): {e}"
        )
        return None, None, None, None, None

    # end_time = time.time()
    # print(
    #     f"Total time for calculate_shortest_path_with_speeds: {end_time - start_time:.8f} seconds, actual :{total_time_taken}s"
    # )

    return shortest_path, travel_time_osm, travel_time_inrix, travel_time_historical, total_distance


def get_average_speed_at_timestamp(timestamp, inrix_data):
    average_speed = calculate_average_speed_at_timestamp(timestamp, inrix_data, "speed")
    average_speed_historical = calculate_average_speed_at_timestamp(timestamp, inrix_data, "historical_average_speed")
    if average_speed is not None:
        average_speed_mps = convert_speed_to_mps(average_speed, "mph")  # Assuming speed is in mph
    else:
        average_speed_mps = 50 * 1000 / 3600  # Default speed if average speed is not available
    if average_speed_historical is not None:
        average_speed_historical_mps = convert_speed_to_mps(
            average_speed_historical, "mph"
        )  # Assuming speed is in mph
    else:
        average_speed_historical_mps = 55 * 1000 / 3600  # Default speed if average speed is not available

    return average_speed_mps, average_speed_historical_mps


def add_shortest_path_and_speeds(df, graph, timestamp, inrix_dict, average_speed_mps, average_speed_historical_mps):
    shortest_paths = []
    travel_times_osm = []
    travel_times_inrix = []
    travel_times_historical = []
    travel_distances = []

    for index, row in df.iterrows():
        path, time_osm, time_inrix, time_historical, distance = calculate_shortest_path_with_speeds(
            graph,
            row["origin_loc_lat"],
            row["origin_loc_lon"],
            row["dest_loc_lat"],
            row["dest_loc_lon"],
            row["shortest_path"],
            timestamp,
            inrix_dict,
            average_speed_mps,
            average_speed_historical_mps,
        )
        shortest_paths.append(path)
        travel_times_osm.append(time_osm)
        travel_times_inrix.append(time_inrix)
        travel_times_historical.append(time_historical)
        travel_distances.append(distance)
    # df["travel_time_osm_secs"] = travel_times_osm
    df["travel_time_inrix_secs"] = travel_times_inrix
    df["travel_time_historical_secs"] = travel_times_historical
    df["travel_distance_m"] = travel_distances
    df["inrix_to_historical_speeds"] = travel_times_historical / travel_times_inrix  # higher means more congestion

    return df


def add_shortest_path_and_speeds_parallel(
    df, graph, timestamp, inrix_dict, average_speed_mps, average_speed_historical_mps
):
    num_partitions = int(cpu_count() / 8)
    df_split = np.array_split(df, num_partitions)

    with Pool(processes=num_partitions) as pool:
        results = pool.starmap(
            add_shortest_path_and_speeds,
            [
                (chunk, graph, timestamp, inrix_dict, average_speed_mps, average_speed_historical_mps)
                for chunk in df_split
            ],
        )

    df_combined = pd.concat(results, ignore_index=True)
    return df_combined


def format_probabilities(probabilities):

    probabilities = np.nan_to_num(probabilities, nan=0.0001)
    # probabilities = pd.to_numeric(trip_distribution, errors="coerce").fillna(0).values
    # probabilities /= probabilities.sum() if probabilities.sum() != 0 else 1

    indices = np.arange(len(probabilities))
    nan_positions = probabilities == 0.0001

    for idx in indices[(nan_positions) & (indices >= 3) & (indices < len(probabilities))]:
        if idx <= 9:
            probabilities[idx] = 0.01

    probabilities = np.array(probabilities, dtype=float)  # Convert to float to avoid type issues
    probabilities = np.nan_to_num(probabilities, nan=0.0001, posinf=0.0001, neginf=0.0001)
    probabilities /= probabilities.sum() if probabilities.sum() != 0 else len(probabilities)

    return probabilities


def preprocessing_probabilites(subset_df):

    time_columns = [col for col in subset_df.columns if "estimate" in col and "total" not in col]
    total = subset_df["total_estimate"].sum()

    distribution = subset_df[time_columns].sum() / total

    times = distribution.index.tolist()

    probabilities = distribution.values
    probabilities = format_probabilities(probabilities)

    return probabilities, times


def define_time_blocks(time_descriptions):
    # Convert time descriptions to seconds
    time_blocks = []
    for description in time_descriptions:
        if description.startswith("under"):
            max_minutes = int(description.split("_")[1])
            time_blocks.append((0, max_minutes * 60 - 1))
        elif description.endswith("and_over_estimate"):
            min_minutes = int(description.split("_")[0])
            time_blocks.append((min_minutes * 60, float("inf")))
        else:
            min_minutes, max_minutes = (
                int(description.split("_to_")[0]),
                int(description.split("_to_")[1].split("_")[0]),
            )
            time_blocks.append((min_minutes * 60, max_minutes * 60 - 1))

    return time_blocks


def stratified_subsampling(df):

    # Combine origin, destination, and departure_time to create strata
    df["strata"] = df["origin"] + "-" + df["destination"] + "-" + df["departure_time"]

    # Get unique strata and their counts
    strata_counts = df["strata"].value_counts()

    # Define sample size (e.g., 50% of the original data)
    sample_size = int(len(df) * 0.5)

    # Perform stratified sampling
    sub_sample = df.groupby("strata", group_keys=False).apply(lambda x: x.sample(frac=0.5))

    # Reset index of the sub-sample
    sub_sample.reset_index(drop=True, inplace=True)


def sample_gaussian_dist(df, num_samples):
    mean = df["total_estimate"].iloc[0]
    margin_of_error = df["total_margin"].iloc[0]
    z_score = 1.645  # For 90% confidence interval
    standard_deviation = margin_of_error / z_score
    samples = np.random.normal(mean, standard_deviation, num_samples)
    return samples / 5  # from weekly to daily


def to_closest_monday(timestamp_str):
    # Convert the string timestamp to a datetime object
    timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

    # Calculate the difference to the closest Monday
    # weekday() returns 0 for Monday and 6 for Sunday
    days_to_monday = (timestamp.weekday() - 0) % 7
    if days_to_monday > 3:  # If the difference is more than 3 days, move to the previous Monday
        days_to_monday -= 7

    # Adjust the date to the closest Monday
    closest_monday = timestamp - datetime.timedelta(days=days_to_monday)

    return closest_monday


# def get_OSM_graph(county, state):
#     place_name = f"{county} County, {state}, USA"
#     network_type = "drive"
#     graph = ox.graph_from_place(place_name, network_type=network_type)

#     G = ox.graph_from_place(place_name, network_type=network_type)
#     return G


def get_departure_times():

    table = "B08301"  # Replace with your actual table code
    columns_to_be_renamed = {
        f"{table}_001E": "total_estimate",
        f"{table}_001M": "total_margin",
        f"{table}_002E": "12am_to_4:59am_estimate",
        f"{table}_002M": "12am_to_4:59am_margin",
        f"{table}_003E": "5am_to_5:29am_estimate",
        f"{table}_003M": "5am_to_5:29am_margin",
        f"{table}_004E": "5:30am_to_5:59am_estimate",
        f"{table}_004M": "5:30am_to_5:59am_margin",
        f"{table}_005E": "6am_to_6:29am_estimate",
        f"{table}_005M": "6am_to_6:29am_margin",
        f"{table}_006E": "6:30am_to_6:59am_estimate",
        f"{table}_006M": "6:30am_to_6:59am_margin",
        f"{table}_007E": "7am_to_7:29am_estimate",
        f"{table}_007M": "7am_to_7:29am_margin",
        f"{table}_008E": "7:30am_to_7:59am_estimate",
        f"{table}_008M": "7:30am_to_7:59am_margin",
        f"{table}_009E": "8am_to_8:29am_estimate",
        f"{table}_009M": "8am_to_8:29am_margin",
        f"{table}_010E": "8:30am_to_8:59am_estimate",
        f"{table}_010M": "8:30am_to_8:59am_margin",
        f"{table}_011E": "9am_to_9:59am_estimate",
        f"{table}_011M": "9am_to_9:59am_margin",
        f"{table}_012E": "10am_to_10:59am_estimate",
        f"{table}_012M": "10am_to_10:59am_margin",
        f"{table}_013E": "11am_to_11:59am_estimate",
        f"{table}_013M": "11am_to_11:59am_margin",
        f"{table}_014E": "12pm_to_3:59pm_estimate",
        f"{table}_014M": "12pm_to_3:59pm_margin",
        f"{table}_015E": "4pm_to_11:59pm_estimate",
        f"{table}_015M": "4pm_to_11:59pm_margin",
    }

    base_date = "2021-01-04"  # Change this base date as needed
    time_columns = [col for col in columns_to_be_renamed.values() if "estimate" in col and "total" not in col]

    departure_times = []
    for col in time_columns:
        time_range = col.split("_to_")[0].replace("_estimate", "")
        if ":" in time_range:
            hour_min = time_range.split("am")[0] if "am" in time_range else time_range.split("pm")[0]
            start_hour, start_min = map(int, hour_min.split(":"))
        else:
            hour_min = time_range.split("am")[0] if "am" in time_range else time_range.split("pm")[0]
            start_hour = int(hour_min)
            start_min = 0
        if "pm" in time_range and start_hour != 12:
            start_hour += 12
        if "am" in time_range and start_hour == 12:
            start_hour = 0
        departure_time = f"{start_hour:02d}:{start_min:02d}:00"
        # departure_time = f"{base_date} {start_hour:02d}:{start_min:02d}:00"
        departure_times.append(departure_time)

    departure_times = pd.to_datetime(departure_times)
    next_day_midnight = (departure_times[-1] + pd.Timedelta(days=1)).normalize()
    departure_times = departure_times.append(pd.DatetimeIndex([next_day_midnight]))
    departure_times = [pd.to_datetime(time).time() for time in departure_times]
    return departure_times


def filter_inrix(G_projected, state, county, inrix_path, segments, translation, inrix_segments):

    use_cols = ["xd_id", "measurement_tstamp", "speed", "historical_average_speed"]
    chunk_iter = pd.read_csv(f"{inrix_path}/Hamilton-2021-Inrix-Data.csv", chunksize=100000, usecols=use_cols)
    inrix = pd.DataFrame()
    for chunk in chunk_iter:
        inrix = pd.concat([inrix, chunk])
        # break
        # count += 1

    inrix["measurement_tstamp"] = pd.to_datetime(inrix["measurement_tstamp"])
    inrix["xd_id"] = inrix["xd_id"].astype(int)

    inrix.set_index("measurement_tstamp", inplace=True)
    hourly_inrix = (
        inrix.groupby("xd_id", as_index=False).resample("30T").mean().reset_index().drop(["level_0"], axis=1)
    )
    inrix.reset_index(inplace=True)
    hourly_inrix = hourly_inrix.dropna(subset=["speed"])
    hourly_inrix["xd_id"] = hourly_inrix["xd_id"].astype(int)

    departure_times = get_departure_times()
    hourly_inrix["measurement_tstamp"] = pd.to_datetime(hourly_inrix["measurement_tstamp"])
    hourly_inrix["time"] = hourly_inrix["measurement_tstamp"].dt.time
    filtered_inrix = hourly_inrix[hourly_inrix["time"].isin(departure_times)]
    filtered_inrix["xd_id"] = filtered_inrix["xd_id"].astype(int)
    filtered_inrix.to_csv(f"{inrix_path}/filtered_datetime_inrix.csv")

    inrix = filtered_inrix
    inrix["xd_id"] = inrix["xd_id"].astype(str)
    segments = gpd.read_file(f"{inrix_path}/USA_TN_OSM_20231201_segments_shapefile.zip")
    translation = gpd.read_file(f"{inrix_path}/USA_Tennessee.csv")
    translation = (
        translation.set_index(
            [
                "XDSegID",
                "WayStartOffset_m",
                "WayEndOffset_m",
                "WayStartOffset_percent",
                "WayEndOffset_percent",
                "geometry",
            ]
        )
        .apply(lambda x: x.str.split(";").explode())
        .reset_index()
    )

    translation["OSMWayIDs"] = translation["OSMWayIDs"].astype(str)
    translation["OSMWayDirections"] = translation["OSMWayDirections"].astype(str)
    # ox.save_graphml(G_projected, 'hamilton_county_osm.graphml')

    inrix_segments = gpd.read_file(f"{inrix_path}/XD_Identification.csv")
    translated_segments = translation.drop(["geometry"], axis=1).merge(
        segments, left_on="OSMWayIDs", right_on="sseg_id"
    )
    inrix_seg_osm = inrix_segments[
        ["xd", "bearing", "start_latitude", "start_longitude", "end_latitude", "end_longitude"]
    ].merge(translated_segments, left_on=["xd"], right_on=["XDSegID"])[
        [
            "xd",
            "bearing",
            "start_latitude",
            "start_longitude",
            "end_latitude",
            "end_longitude",
            "OSMWayIDs",
            "OSMWayDirections",
        ]
    ]

    inrix_merged_with_osm = inrix.merge(inrix_seg_osm, left_on="xd_id", right_on="xd")
    inrix_dict = preprocess_inrix_data(inrix_merged_with_osm)

    return inrix_dict


def get_travel_time_osmnx(G, od_record):
    start_point = (od_record["origin_loc_lat"], od_record["origin_loc_lon"])
    end_point = (od_record["dest_loc_lat"], od_record["dest_loc_lon"])
    # Find the nearest nodes to the start and end points
    orig_node = ox.distance.nearest_nodes(G, start_point[1], start_point[0])
    dest_node = ox.distance.nearest_nodes(G, end_point[1], end_point[0])

    # Use NetworkX to find the shortest path
    try:
        shortest_path = nx.shortest_path(G, orig_node, dest_node, weight="length")
        total_distance = sum(ox.utils_graph.get_route_edge_attributes(G, shortest_path, "length"))
        move_time = total_distance / (50 * 1000 / 60 / 60)  # Assuming an average speed of 50 km/h

        distance_miles = total_distance * 0.000621371  # Convert meters to miles
    except Exception as e:
        print(e)
        move_time = total_distance = distance_miles = np.nan

    return move_time, total_distance, distance_miles


# Saving the dictionaries
def save_building_dictionaries(origin_buildings, dest_buildings, output_path):
    """Save building dictionaries using a custom approach"""
    # Create directories
    os.makedirs(f"{output_path}/origin_buildings", exist_ok=True)
    os.makedirs(f"{output_path}/dest_buildings", exist_ok=True)

    # Save origin buildings
    origin_metadata = {}
    for geocode, data in origin_buildings.items():
        # Save the buildings DataFrame
        buildings_file = f"{output_path}/origin_buildings/{geocode}_buildings.parquet"
        data["buildings"].to_parquet(buildings_file)

        # Save metadata
        origin_metadata[geocode] = {"buildings_file": buildings_file, "locations": data["locations"]}

    # Save dest buildings
    dest_metadata = {}
    for geocode, data in dest_buildings.items():
        # Save the buildings DataFrame
        buildings_file = f"{output_path}/dest_buildings/{geocode}_buildings.parquet"
        data["buildings"].to_parquet(buildings_file)

        # Save metadata
        dest_metadata[geocode] = {"buildings_file": buildings_file, "locations": data["locations"]}

    # Save metadata files
    with open(f"{output_path}/origin_metadata.json", "w") as f:
        json.dump(origin_metadata, f)

    with open(f"{output_path}/dest_metadata.json", "w") as f:
        json.dump(dest_metadata, f)

    print(f"Building dictionaries saved to {output_path}")


# Loading the dictionaries
def load_building_dictionaries(output_path):
    """Load building dictionaries using a custom approach"""
    # Load metadata
    with open(f"{output_path}/origin_metadata.json", "r") as f:
        origin_metadata = json.load(f)

    with open(f"{output_path}/dest_metadata.json", "r") as f:
        dest_metadata = json.load(f)

    # Reconstruct dictionaries
    origin_buildings = {}
    for geocode, metadata in origin_metadata.items():
        buildings = pd.read_parquet(metadata["buildings_file"])
        origin_buildings[geocode] = {"buildings": buildings, "locations": metadata["locations"]}

    dest_buildings = {}
    for geocode, metadata in dest_metadata.items():
        buildings = pd.read_parquet(metadata["buildings_file"])
        dest_buildings[geocode] = {"buildings": buildings, "locations": metadata["locations"]}

    return origin_buildings, dest_buildings


def calculate_speed_shift(routing_df, travel_time_to_work_by_geoid):
    """
    Calculate the Mean Speed Shift Ratio (MSSr) based on simulated and census travel times.

    Parameters:
    -----------
    routing_df : DataFrame
        DataFrame containing the simulated travel times
    travel_time_to_work_by_geoid : DataFrame
        DataFrame containing the census travel times

    Returns:
    --------
    float
        Mean Speed Shift Ratio (MSSr)
    """
    # Calculate mean simulated travel time
    mean_simulated_time = routing_df["travel_time_min"].mean()

    # Calculate mean census travel time (weighted average from bins)
    time_columns = [col for col in travel_time_to_work_by_geoid.columns if "_estimate" in col and "total_" not in col]

    # Define midpoint values for each time bin (in minutes)
    time_midpoints = {
        "under_5_minutes_estimate": 2.5,
        "5_to_9_minutes_estimate": 7.0,
        "10_to_14_minutes_estimate": 12.0,
        "15_to_19_minutes_estimate": 17.0,
        "20_to_24_minutes_estimate": 22.0,
        "25_to_29_minutes_estimate": 27.0,
        "30_to_34_minutes_estimate": 32.0,
        "35_to_39_minutes_estimate": 37.0,
        "40_to_44_minutes_estimate": 42.0,
        "45_to_59_minutes_estimate": 52.0,
        "60_to_89_minutes_estimate": 75.0,
        "90_minutes_and_over_estimate": 105.0,
    }

    # Calculate weighted average
    total_commuters = travel_time_to_work_by_geoid["total_estimate"].sum()
    weighted_sum = 0

    for col in time_columns:
        # if col in time_midpoints.keys():
        weighted_sum += travel_time_to_work_by_geoid[col].sum() * time_midpoints[col]

    mean_census_time = weighted_sum / total_commuters if total_commuters > 0 else 0

    # Calculate MSSr
    mssr = (mean_census_time - mean_simulated_time) / mean_census_time if mean_census_time > 0 else 0

    print(f"Mean Simulated Travel Time: {mean_simulated_time:.2f} minutes")
    print(f"Mean Census Travel Time: {mean_census_time:.2f} minutes")
    print(f"Mean Speed Shift Ratio (MSSr): {mssr:.4f}")

    return mssr


def apply_mssr_to_existing_graphs(hourly_graphs, mssr):
    """
    Apply Mean Speed Shift Ratio to existing hourly graphs.
    Only modifies edges that don't have INRIX data.
    """
    # Calculate speed adjustment factor
    speed_adjustment_factor = 1.0 / (1.0 + mssr)
    adjusted_graphs = {}

    print(f"Applying speed adjustment factor: {speed_adjustment_factor:.4f}")

    for hour, G_hour in hourly_graphs.items():
        # Create a copy of the graph
        G_adjusted = G_hour.copy()

        default_edges = 0
        inrix_edges = 0

        # Adjust speeds for non-INRIX edges
        for u, v, k, data in G_adjusted.edges(keys=True, data=True):
            # Check if this edge has INRIX data
            if data.get("inrix_speed", False):
                inrix_edges += 1
                continue

            # This is a default speed edge
            default_edges += 1

            # Get the current speed
            if "speed_kph" in data:
                original_speed = data["speed_kph"]
            else:
                # Calculate speed from length and travel time
                length = data.get("length", 0)
                travel_time = data.get("travel_time", 0)
                original_speed = (3.6 * length) / travel_time if travel_time > 0 else 80

            # Apply speed shift
            adjusted_speed = original_speed * speed_adjustment_factor

            # Update edge data
            if "length" in data and adjusted_speed > 0:
                travel_time = (3.6 * data["length"]) / adjusted_speed
                data["travel_time"] = travel_time
                data["weight"] = travel_time
                data["speed_kph"] = adjusted_speed

        print(f"Hour {hour}: Adjusted {default_edges} default edges, kept {inrix_edges} INRIX edges")
        adjusted_graphs[hour] = G_adjusted

    return adjusted_graphs


def create_hourly_graphs_with_speed_shift(all_hours, hourly_inrix_df, conversion_df, mssr, default_speed_kmh=80):
    """
    Create hourly graphs with speed shift applied to default speed edges.

    Parameters:
    -----------
    all_hours : list
        List of hour timestamps
    hourly_inrix_df : DataFrame
        DataFrame with INRIX speed data
    conversion_df : DataFrame
        DataFrame for mapping INRIX IDs to OSM edges
    mssr : float
        Mean Speed Shift Ratio
    default_speed_kmh : float, optional
        Default speed for edges without INRIX data (km/h)

    Returns:
    --------
    dict
        Dictionary of hourly graphs with adjusted speeds
    """
    # Calculate speed adjustment factor (inverse of MSSR effect on time)
    speed_adjustment_factor = 1.0 / (1.0 + mssr)
    adjusted_default_speed = default_speed_kmh * speed_adjustment_factor

    print(
        f"Speed Adjustment: Original default = {default_speed_kmh} km/h, Adjusted = {adjusted_default_speed:.1f} km/h"
    )
    print(
        f"Adjustment factor = {speed_adjustment_factor:.4f} (speeds are being {'increased' if speed_adjustment_factor > 1 else 'decreased'})"
    )

    hourly_graphs = {}  # Store G_hour per timestamp

    for hour in all_hours:
        # Filter INRIX for this hour
        inrix_snapshot = hourly_inrix_df[hourly_inrix_df["measurement_tstamp"] == hour]

        # Merge with conversion
        merged_df = pd.merge(inrix_snapshot, conversion_df, left_on="xd_id", right_on="xd")

        # Create LineStrings
        geometries = [
            LineString([(x1, y1), (x2, y2)])
            for x1, y1, x2, y2 in zip(
                merged_df["start_longitude"],
                merged_df["start_latitude"],
                merged_df["end_longitude"],
                merged_df["end_latitude"],
            )
        ]
        inrix_gdf = gpd.GeoDataFrame(merged_df, geometry=geometries, crs="EPSG:4326")

        # Fetch OSM network only once (outside loop for efficiency)
        if "G_base" not in locals():
            G_base = ox.graph_from_place("Hamilton County, TN, USA", network_type="drive")
            edges_base = ox.graph_to_gdfs(G_base, nodes=False, edges=True).to_crs(epsg=3857)

        # Project INRIX and join to edges
        inrix_proj = inrix_gdf.to_crs(epsg=3857)
        joined = gpd.sjoin_nearest(inrix_proj, edges_base, how="left", distance_col="dist")

        # Clone graph
        G_hour = G_base.copy()

        # Track which edges have INRIX speeds
        inrix_edges = set()

        # Update edge weights in G_hour using INRIX speeds
        for idx, row in joined.iterrows():
            u, v, key = row["index_right0"], row["index_right1"], row["index_right2"]
            speed = row["speed"]
            length = row["length"]
            if pd.notnull(speed) and speed > 0:
                # Mark this as an INRIX edge
                inrix_edges.add((u, v, key))

                # Calculate travel time using INRIX speed
                travel_time = (3.6 * length) / speed

                # Update graph edge
                G_hour[u][v][key]["travel_time"] = travel_time
                G_hour[u][v][key]["weight"] = travel_time
                G_hour[u][v][key]["speed_kph"] = speed
                G_hour[u][v][key]["inrix_speed"] = True

        # Assign adjusted default speeds to edges without INRIX data
        default_edges = 0
        for u, v, k, data in G_hour.edges(keys=True, data=True):
            if (u, v, k) not in inrix_edges:
                length = data.get("length", None)
                if length:
                    # Apply speed shift to default speed
                    speed = adjusted_default_speed
                    travel_time = (3.6 * length) / speed

                    # Update edge data
                    data["travel_time"] = travel_time
                    data["weight"] = travel_time
                    data["speed_kph"] = speed
                    data["inrix_speed"] = False
                    default_edges += 1

        # Store statistics for this hour
        inrix_coverage = (
            len(inrix_edges) / (len(inrix_edges) + default_edges) * 100
            if (len(inrix_edges) + default_edges) > 0
            else 0
        )
        print(
            f"Hour {hour}: {len(inrix_edges)} INRIX edges, {default_edges} default edges ({inrix_coverage:.1f}% INRIX coverage)"
        )

        # Store graph
        hourly_graphs[hour] = G_hour

    return hourly_graphs
