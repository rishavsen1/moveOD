import pandas as pd
from shapely.geometry import LineString
import osmnx as ox
import geopandas as gpd
import multiprocessing as mp
from tqdm.notebook import tqdm
import networkx as nx
from generate.config import *


def find_origin_dest_nodes(df, G):
    origin_lons = df["origin_loc_lon"].to_list()
    origin_lats = df["origin_loc_lat"].to_list()
    dest_lons = df["dest_loc_lon"].to_list()
    dest_lats = df["dest_loc_lat"].to_list()

    origin_nodes = ox.distance.nearest_nodes(G, X=origin_lons, Y=origin_lats)
    dest_nodes = ox.distance.nearest_nodes(G, X=dest_lons, Y=dest_lats)

    df["origin_nodes"] = origin_nodes
    df["dest_nodes"] = dest_nodes

    # origin_loc_to_node = {loc: node for loc, node in zip(origin_locs, origin_nodes)}
    # dest_loc_to_node = {loc: node for loc, node in zip(dest_locs, dest_nodes)}

    origin_nodes = list(set(origin_nodes))
    dest_nodes = list(set(dest_nodes))

    return df


def process_inrix(state, county, inrix_df=None, conversion_df=None, desired_date=None):
    """
    Process INRIX data or fallback to OSM speed limits if INRIX is not available
    """

    # Fetch OSM network
    G_base = ox.graph_from_place(f"{county} County, {state}, USA", network_type="drive")

    if inrix_df is None or inrix_df.empty:
        print("INRIX data not available, using OSM speed limits as fallback")
        return create_graphs_from_osm_speeds(G_base, desired_date)

    # Process INRIX data as before
    hourly_inrix_df = (
        inrix_df.set_index("measurement_tstamp")
        .groupby("xd_id")
        .resample(TIME_INTERVAL)
        .agg(
            {
                "speed": "mean",
                "historical_average_speed": "mean",
                "reference_speed": "mean",
                "travel_time_minutes": "mean",
                "confidence_score": "mean",
                "cvalue": "mean",
            }
        )
        .reset_index()
    )

    hourly_inrix_df = hourly_inrix_df.round(2)
    all_hours = hourly_inrix_df["measurement_tstamp"].drop_duplicates().sort_values()

    hourly_graphs = {}
    edges_base = ox.graph_to_gdfs(G_base, nodes=False, edges=True).to_crs(epsg=3857)

    for hour in all_hours:
        # Filter INRIX for this hour
        inrix_snapshot = hourly_inrix_df[hourly_inrix_df["measurement_tstamp"] == hour]

        # Merge with conversion
        merged_df = pd.merge(inrix_snapshot, conversion_df, left_on="xd_id", right_on="xd")

        # Create LineStrings and process as before
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
        inrix_proj = inrix_gdf.to_crs(epsg=3857)
        joined = gpd.sjoin_nearest(inrix_proj, edges_base, how="left", distance_col="dist")

        # Clone graph and update with INRIX data
        G_hour = G_base.copy()

        # Update edge weights with INRIX data
        for idx, row in joined.iterrows():
            u, v, key = row["index_right0"], row["index_right1"], row["index_right2"]
            speed = row["speed"]
            length = row["length"]
            if pd.notnull(speed) and speed > 0:
                travel_time = (3.6 * length) / speed
                G_hour[u][v][key]["travel_time"] = travel_time
                G_hour[u][v][key]["weight"] = travel_time

        # Use OSM speed limits for edges without INRIX data
        assign_osm_speeds_to_graph(G_hour)
        hourly_graphs[hour] = G_hour

    G_0 = list(hourly_graphs.values())[0]
    return G_0, hourly_graphs


def create_graphs_from_osm_speeds(G_base, desired_date):
    """
    Create hourly graphs using only OSM speed limits (fallback when no INRIX data)
    """
    print("Creating graphs based on OSM speed limits...")

    # Create a base graph with OSM speeds
    G_osm = G_base.copy()
    assign_osm_speeds_to_graph(G_osm)

    # Without time-varying data there are only two distinct graphs -- peak and
    # off-peak -- so build them once and point every timestamp at the shared
    # object instead of materialising 48 identical copies.  Consumers treat these
    # graphs as read-only; apply_mssr_to_existing_graphs copies before mutating.
    G_offpeak = G_osm
    G_peak = G_osm.copy()
    apply_peak_hour_adjustments(G_peak, reduction_factor=0.7)

    hourly_graphs = {}
    base_date = pd.Timestamp(f"{desired_date}")
    for hour in range(0, 24):
        for minute in [0, 30]:  # keys stay at 30-min resolution
            timestamp = base_date.replace(hour=hour, minute=minute)
            hourly_graphs[timestamp] = G_peak if is_peak_hour(hour) else G_offpeak

    G_0 = list(hourly_graphs.values())[0]
    return G_0, hourly_graphs


def assign_osm_speeds_to_graph(G):
    """
    Assign travel times based on OSM speed limits and road types
    """
    # Default speeds by road type (km/h)
    default_speeds = {
        "motorway": 110,
        "trunk": 90,
        "primary": 70,
        "secondary": 60,
        "tertiary": 50,
        "residential": 40,
        "service": 30,
        "unclassified": 50,
        "living_street": 20,
        "track": 25,
        "path": 15,
        "footway": 5,
        "cycleway": 15,
        "steps": 5,
    }

    for u, v, k, data in G.edges(keys=True, data=True):
        # Try to get speed from OSM data first
        speed_kmh = None

        # Check for maxspeed tag
        if "maxspeed" in data:
            maxspeed = data["maxspeed"]
            if isinstance(maxspeed, str):
                try:
                    # Handle different formats: "50", "50 mph", etc.
                    if "mph" in maxspeed.lower():
                        speed_kmh = float(maxspeed.replace("mph", "").strip()) * 1.60934
                    else:
                        speed_kmh = float(maxspeed.strip())
                except (ValueError, AttributeError):
                    pass
            elif isinstance(maxspeed, (int, float)):
                speed_kmh = float(maxspeed)

        # If no speed found, use highway type
        if speed_kmh is None:
            highway = data.get("highway", "unclassified")
            if isinstance(highway, list):
                highway = highway[0]  # Take first if multiple
            speed_kmh = default_speeds.get(highway, 50)  # Default to 50 km/h

        # Calculate travel time
        length = data.get("length", 100)  # Default length if missing
        travel_time = (3.6 * length) / speed_kmh  # Convert to seconds

        # Update graph
        data["travel_time"] = travel_time
        data["weight"] = travel_time
        data["speed_kmh"] = speed_kmh


def is_peak_hour(hour):
    """
    Determine if the given hour is during peak traffic
    """
    morning_peak = 7 <= hour <= 9
    evening_peak = 17 <= hour <= 19
    return morning_peak or evening_peak


def apply_peak_hour_adjustments(G, reduction_factor=0.7):
    """
    Apply speed reductions during peak hours
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        if "weight" in data:
            # Increase travel time (reduce effective speed) during peak hours
            data["weight"] = data["weight"] / reduction_factor
            if "travel_time" in data:
                data["travel_time"] = data["travel_time"] / reduction_factor


# Update your main calling code to handle missing INRIX data:
def get_hourly_graphs(state, county, inrix_df=None, conversion_df=None):
    """
    Wrapper function to get hourly graphs with fallback
    """
    try:
        if inrix_df is not None and not inrix_df.empty and conversion_df is not None:
            print("Using INRIX data for traffic conditions")
            return process_inrix(state, county, inrix_df, conversion_df)
        else:
            print("INRIX data unavailable, using OSM speed limits")
            return process_inrix(state, county, None, None)
    except Exception as e:
        print(f"Error processing INRIX data: {e}")
        print("Falling back to OSM speed limits")
        G_base = ox.graph_from_place(f"{county} County, {state}, USA", network_type="drive")
        return create_graphs_from_osm_speeds(G_base)
