"""
FastAPI Backend for MOVE-OD Application
Handles data processing, OD generation, and calibration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import datetime
import pandas as pd
import geopandas as gpd
import zipfile
import io
import json
import uuid
import shutil
from pathlib import Path

# Import existing MOVE-OD modules
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Change to parent directory to ensure correct relative paths
os.chdir(Path(__file__).parent.parent)

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

app = FastAPI(title="MOVE-OD API", version="1.0.0")

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for job status (in production, use Redis or database)
job_store = {}


class ProcessingRequest(BaseModel):
    state: str
    county: str
    start_date: str  # ISO format: YYYY-MM-DD
    end_date: str
    lodes_year: str = "2022"
    tiger_shapefile_year: str = "2024"
    inrix_path: Optional[str] = None
    inrix_conversion_path: Optional[str] = None
    use_safegraph: bool = False
    use_ms_buildings: bool = True
    od_option: str = "Origin and Destination in same County"


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """API health check"""
    return {"message": "MOVE-OD API is running", "version": "1.0.0"}


@app.get("/api/states")
async def get_states():
    """Get list of available states and counties"""
    try:
        states, state_fips, counties_in_state, county_fips = get_states_and_counties()

        # Convert to serializable format
        states_data = []
        for state_name, state_id in states.items():
            counties = sorted(counties_in_state.get(state_name, []))
            county_data = [
                {"name": county, "fips": str(county_fips.get((state_name, county), [""])[0])[-3:]}
                for county in counties
            ]

            states_data.append(
                {"name": state_name, "id": state_id, "fips": state_fips.get(state_name, ""), "counties": county_data}
            )

        return {"states": states_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process")
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start OD generation and calibration processing"""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Initialize job status
        job_store[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued for processing",
            "result": None,
            "error": None,
        }

        # Add processing task to background
        background_tasks.add_task(process_od_data, job_id, request)

        return {"job_id": job_id, "message": "Processing started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_store[job_id]


@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    """Download calibrated results as ZIP file"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    output_path = job["result"]["output_path"]

    # Create ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_path)
                zipf.write(file_path, arcname=arcname)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=calibrated_move_od_{job_id}.zip"},
    )


@app.get("/api/map-data/{job_id}")
async def get_map_data(
    job_id: str, data_type: str = "both", sample_size: int = 1000  # "both", "origins", "destinations", "heatmap"
):
    """Get map data for visualization"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    result = job["result"]

    # Sample and convert to GeoJSON
    map_data = {"type": "FeatureCollection", "features": []}

    # For heatmap, include both origins and destinations
    if data_type == "heatmap":
        data_type = "both"

    if data_type in ["both", "origins"] and "origins_geojson" in result:
        origins_df = gpd.read_file(result["origins_geojson"])
        origins_sample = origins_df.sample(n=min(sample_size, len(origins_df)), random_state=42)

        for _, row in origins_sample.iterrows():
            feature = {
                "type": "Feature",
                "geometry": json.loads(gpd.GeoSeries([row.geometry]).to_json())["features"][0]["geometry"],
                "properties": {"type": "origin", "GEOID": row.get("GEOID", "")},
            }
            map_data["features"].append(feature)

    if data_type in ["both", "destinations"] and "destinations_geojson" in result:
        destinations_df = gpd.read_file(result["destinations_geojson"])
        destinations_sample = destinations_df.sample(n=min(sample_size, len(destinations_df)), random_state=42)

        for _, row in destinations_sample.iterrows():
            feature = {
                "type": "Feature",
                "geometry": json.loads(gpd.GeoSeries([row.geometry]).to_json())["features"][0]["geometry"],
                "properties": {"type": "destination", "GEOID": row.get("GEOID", "")},
            }
            map_data["features"].append(feature)

    return map_data


async def process_od_data(job_id: str, request: ProcessingRequest):
    """Background task to process OD data"""
    try:
        # Update status
        job_store[job_id]["status"] = "processing"
        job_store[job_id]["message"] = "Starting processing..."
        job_store[job_id]["progress"] = 0.05

        # Parse dates
        start_date = datetime.datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(request.end_date, "%Y-%m-%d").date()

        # Get state/county info
        states, state_fips, counties_in_state, county_fips = get_states_and_counties()
        state_id = states[request.state]
        county_fips_code = str(county_fips[request.state, request.county][0])[-3:]
        state_fips_id = state_fips[request.state]

        # Setup paths
        output_path = f"./move_OD/{request.state}/{request.county}/{start_date}_{end_date}"
        os.makedirs(output_path, exist_ok=True)

        calibrated_output_path = f"{output_path}/calibrated_move_od/"
        os.makedirs(calibrated_output_path, exist_ok=True)

        # Initialize logger
        logger = Logger(f"{output_path}/{request.county}_{request.state}_{start_date}_{end_date}")

        # Update progress
        job_store[job_id]["progress"] = 0.1
        job_store[job_id]["message"] = "Downloading required files..."

        # Download shapefiles
        county_geoid_path = f"./data/states/{request.state}/tl_{request.tiger_shapefile_year}_{state_fips_id}_bg.zip"
        if not os.path.exists(county_geoid_path):
            compressed_path = county_geoid_path
            url = f"https://www2.census.gov/geo/tiger/TIGER{request.tiger_shapefile_year}/BG/tl_{request.tiger_shapefile_year}_{state_fips_id}_bg.zip"
            download_shapefile(logger, url=url, compressed_path=compressed_path)

        # Download MS Buildings if enabled
        ms_path = None
        if request.use_ms_buildings:
            state_stripped = request.state.replace(" ", "")
            ms_path = f"./data/states/{request.state}/{state_stripped}.geojson"
            if not os.path.exists(ms_path):
                download_ms_buildings(logger, request.state, state_stripped)

        # Download LODES data
        job_store[job_id]["progress"] = 0.2
        job_store[job_id]["message"] = "Downloading LODES data..."

        county_lodes_paths = [
            f"./data/states/{request.state}/{state_id.lower()}_od_main_JT00_{request.lodes_year}.csv"
        ]
        if request.od_option != "Origin and Destination in same County":
            county_lodes_paths.append(
                f"./data/states/{request.state}/{state_id.lower()}_od_aux_JT00_{request.lodes_year}.csv"
            )

        flag = False
        for county_lodes_path in county_lodes_paths:
            if not os.path.exists(county_lodes_path):
                flag = True

        if flag:
            download_lodes(logger, request.state, states[request.state].lower(), lodes_code=0, year=request.lodes_year)

        # Process data
        job_store[job_id]["progress"] = 0.3
        job_store[job_id]["message"] = "Processing GEOID data..."

        county_geoid_df, success = read_county_geoid_df(county_geoid_path, output_path, logger)
        if not success:
            raise Exception("Failed to process GEOID data")

        job_store[job_id]["progress"] = 0.4
        job_store[job_id]["message"] = "Processing LODES data..."

        county_lodes_df, unique_countyfps, success = read_lodes_df(
            county_fips_code, county_lodes_paths, county_geoid_df, output_path, logger, request.od_option
        )
        if not success:
            raise Exception("Failed to process LODES data")

        # Filter county_geoid_df
        county_geoid_df = county_geoid_df[
            county_geoid_df["COUNTYFP"].astype(str).isin([str(fp) for fp in unique_countyfps])
        ]

        job_store[job_id]["progress"] = 0.5
        job_store[job_id]["message"] = "Processing building data..."

        # MS Buildings
        ms_buildings_df = pd.DataFrame()
        if request.use_ms_buildings:
            ms_buildings_df, success = read_ms_buildings_data(
                county_fips_code, county_geoid_df, logger, output_path, request.state
            )

        # Generate origin/destination locations
        res_locations, combined_work_locations, success = read_origin_dest_locations(
            county_fips_code,
            request.county,
            county_geoid_df,
            request.use_safegraph,
            output_path,
            logger,
            request.od_option,
        )
        if not success:
            raise Exception("Failed to generate origin/destination locations")

        job_store[job_id]["progress"] = 0.6
        job_store[job_id]["message"] = "Processing INRIX data and generating graphs..."

        # Process INRIX
        inrix_df = None
        conversion_df = None

        if request.inrix_path and os.path.exists(request.inrix_path):
            inrix_df = pd.read_csv(request.inrix_path)
            if request.inrix_conversion_path and os.path.exists(request.inrix_conversion_path):
                conversion_df = pd.read_csv(request.inrix_conversion_path)

            inrix_df["measurement_tstamp"] = pd.to_datetime(inrix_df["measurement_tstamp"])
            inrix_df = inrix_df[inrix_df["measurement_tstamp"].dt.date == start_date]

        G, hourly_graphs = process_inrix(request.state, request.county, inrix_df, conversion_df, start_date)

        job_store[job_id]["progress"] = 0.7
        job_store[job_id]["message"] = "Generating OD combinations..."

        # Generate datetime ranges
        datetime_ranges = get_datetime_ranges(start_date, end_date, timedelta=15)

        # LODES combinations
        lodes_combs = LodesComb(
            county_geoid_df,
            output_path,
            request.use_ms_buildings,
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
            county_fips_code,
            G,
            hourly_graphs,
            block_groups="*",
        )

        job_store[job_id]["progress"] = 0.8
        job_store[job_id]["message"] = "Calibrating OD data..."

        # Process each day
        for day, lodes_output_df in zip(days, lodes_output_dfs):
            # Get routed trips
            routing_df = get_routed(od_df=lodes_output_df, desired_date=start_date, hourly_graphs_arg=hourly_graphs)

            # Perform mean speed shift
            hourly_graphs_adjusted = perform_mean_speed_shift(
                routing_df=routing_df, travel_time_to_work_by_geoid=travel_time_to_work_df
            )

            # Get routed trips post MSSR
            post_mssr_routing_df = get_routed(
                od_df=lodes_output_df,
                desired_date=start_date,
                hourly_graphs_arg=hourly_graphs_adjusted,
            )

            # Calibrate
            calibrated_df = calibrate_with_ilp(
                lodes_output_df,
                routing_df,
                res_locations,
                combined_work_locations,
                ms_buildings_df,
                census_depart_times_df,
                travel_time_to_work_df,
            )

            # Get travel times for calibrated df
            routing_df = get_routed(
                od_df=calibrated_df,
                desired_date=start_date,
                hourly_graphs_arg=hourly_graphs,
                post_calibration=True,
            )

            # Save output
            calibrated_df_output_path = f"{calibrated_output_path}/{day}.csv"
            routing_df.to_csv(calibrated_df_output_path)

        # Save location data for map
        origins_geojson = f"{output_path}/county_residential_buildings.geojson"
        destinations_geojson = f"{output_path}/county_work_locations.geojson"

        job_store[job_id]["progress"] = 1.0
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["message"] = "Processing completed successfully"
        job_store[job_id]["result"] = {
            "output_path": calibrated_output_path,
            "origins_geojson": origins_geojson,
            "destinations_geojson": destinations_geojson,
            "origins_count": len(res_locations),
            "destinations_count": len(combined_work_locations),
            "census_block_groups": len(county_geoid_df),
        }

    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
        job_store[job_id]["message"] = f"Processing failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {str(e)}")


def read_county_geoid_df(county_geoid_path, output_path, logger):
    """Helper function to read county GEOID data"""
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
        success = county_geoid_df.shape[0] > 0

    return county_geoid_df, success


def read_lodes_df(county_fips, county_lodes_paths, county_geoid_df, output_path, logger, od_option):
    """Helper function to read LODES data"""
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


def read_ms_buildings_data(county_fips, county_geoid_df, logger, output_path, state):
    """Helper function to read MS Buildings data"""
    success = False
    ms_buldings_path = f"{output_path}/county_buildings_MS.geojson"
    if os.path.exists(ms_buldings_path):
        ms_buildings_df = gpd.read_file(ms_buldings_path)
        ms_buildings_df["geo_centers"] = ms_buildings_df.geometry.centroid
        ms_buildings_df["location"] = ms_buildings_df.geo_centers.apply(lambda p: [p.y, p.x])
        ms_buildings_df = ms_buildings_df[["geometry", "GEOID", "geo_centers", "location"]]
        success = True
    else:
        state_stripped = state.replace(" ", "")
        ms_path = f"./data/states/{state}/{state_stripped}.geojson"
        ms_builds = MSBuildings(county_fips, county_geoid_df, ms_path, output_path, logger)
        ms_buildings_df, success = ms_builds.buildings()

    return ms_buildings_df, success


def read_origin_dest_locations(county_fips, county, county_geoid_df, sg_enabled, output_path, logger, od_option):
    """Helper function to read origin/destination locations"""
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
