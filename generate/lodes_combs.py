#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import numpy as np
import math
import os
import random

from shapely.geometry import Point
from collections import defaultdict
from dask import delayed, compute
from datetime import datetime, timedelta
from generate.process_inrix import find_origin_dest_nodes

from generate.utils import (
    get_census_data_wrapper,
    get_census_travel_time_data,
    define_time_blocks,
    get_census_work_time,
    save_building_dictionaries,
    parse_time_range,
    preprocessing_probabilites,
)


def od_assign_start_end_optimized(params):
    """
    Optimized version that uses pre-built building dictionaries
    instead of repeatedly filtering the building dataframes.
    """
    (county_lodes_df, origin_buildings, dest_buildings) = params

    if county_lodes_df.empty:
        return gpd.GeoDataFrame()

    od_data = []

    for _, movement in county_lodes_df.iterrows():
        h_geocode = movement["h_geocode"]
        w_geocode = movement["w_geocode"]
        jobs = movement["total_jobs"]

        # Skip if we don't have buildings for either origin or destination
        if h_geocode not in origin_buildings or w_geocode not in dest_buildings:
            continue

        origin_data = origin_buildings[h_geocode]
        dest_data = dest_buildings[w_geocode]

        # Generate random indices for sampling buildings
        origin_buildings_count = len(origin_data["buildings"])
        dest_buildings_count = len(dest_data["buildings"])

        # Generate random indices with replacement if needed
        origin_indices = np.random.choice(origin_buildings_count, jobs, replace=jobs > origin_buildings_count)
        dest_indices = np.random.choice(dest_buildings_count, jobs, replace=jobs > dest_buildings_count)

        # Create OD records
        for i in range(jobs):
            origin_building = origin_data["buildings"].iloc[origin_indices[i]]
            dest_building = dest_data["buildings"].iloc[dest_indices[i]]

            od_record = {
                "h_geocode": h_geocode,
                "w_geocode": w_geocode,
                "total_jobs": jobs,
                "origin_loc": origin_building["location"],
                "origin_loc_lat": origin_building["location"][0],
                "origin_loc_lon": origin_building["location"][1],
                "dest_loc": dest_building["location"],
                "dest_loc_lat": dest_building["location"][0],
                "dest_loc_lon": dest_building["location"][1],
                # "possible_home_loc": origin_data["locations"],
                # "possible_dest_loc": dest_data["locations"],
            }

            od_data.append(od_record)

    # Convert to GeoDataFrame
    return gpd.GeoDataFrame(od_data)


class LodesComb:
    def __init__(
        self,
        county_geoid,
        output_path,
        ms_enabled,
        datetime_ranges,
        logger,
    ) -> None:
        self.county_geoid = county_geoid
        self.output_path = output_path
        self.ms_enabled = ms_enabled
        self.datetime_ranges = datetime_ranges
        self.logger = logger
        self.logger.info("Initalizing lodes_comb.py")

    def intpt_func(self, row):
        return Point(row["INTPTLON"], row["INTPTLAT"])

    def func_origin_pt(self, row):
        return Point(row.origin_loc_lon, row.origin_loc_lat)

    def func_dest_pt(self, row):
        return Point(row.dest_loc_lon, row.dest_loc_lat)

    def datetime_range(self, start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    def preprocess_buildings(
        self, county_geoid_df, res_build, com_build, ms_build, unique_h_geocodes, unique_w_geocodes
    ):
        """
        Preprocess buildings data to create lookup dictionaries for origins and destinations.
        This is done once before parallel processing to avoid redundant computations.
        """
        origin_buildings = {}
        dest_buildings = {}

        # Process origin buildings (h_geocode)
        for h_geocode in unique_h_geocodes:
            # Try residential buildings first
            buildings = res_build[res_build["GEOID"] == h_geocode]

            # If empty, try Microsoft buildings
            if buildings.empty and hasattr(self, "ms_enabled") and self.ms_enabled:
                buildings = ms_build[ms_build["GEOID"] == h_geocode]

            # If still empty, use county centroid
            if buildings.empty:
                buildings = county_geoid_df[county_geoid_df["GEOID"] == h_geocode]

            # Store the buildings and locations for this h_geocode
            if not buildings.empty:
                origin_buildings[h_geocode] = {
                    "buildings": buildings,
                    "locations": buildings["location"].values.tolist(),
                }

        # Process destination buildings (w_geocode)
        for w_geocode in unique_w_geocodes:
            # Try commercial buildings first
            buildings = com_build[com_build["GEOID"] == w_geocode]

            # If empty, try Microsoft buildings
            if buildings.empty and hasattr(self, "ms_enabled") and self.ms_enabled:
                buildings = ms_build[ms_build["GEOID"] == w_geocode]

            # If still empty, use county centroid
            if buildings.empty:
                buildings = county_geoid_df[county_geoid_df["GEOID"] == w_geocode]

            # Store the buildings and locations for this w_geocode
            if not buildings.empty:
                dest_buildings[w_geocode] = {
                    "buildings": buildings,
                    "locations": buildings["location"].values.tolist(),
                }

        return origin_buildings, dest_buildings

    def process_county_data(
        self, county_lodes_df, county_geoid_df, res_locations, combined_work_locations, ms_buildings_df, G, days
    ):
        # Get unique geocodes from the entire dataset
        unique_h_geocodes = county_lodes_df["h_geocode"].unique()
        unique_w_geocodes = county_lodes_df["w_geocode"].unique()

        # Preprocess buildings once - this creates lookup dictionaries for all geocodes
        origin_buildings, dest_buildings = self.preprocess_buildings(
            county_geoid_df,
            res_locations,
            combined_work_locations,
            ms_buildings_df,
            unique_h_geocodes,
            unique_w_geocodes,
        )

        for day in days:
            # Process h_geocodes in chunks
            unique_h_geocodes = county_lodes_df["h_geocode"].unique()
            chunk_size = 50  # Adjust based on your data size
            h_geocode_chunks = [
                unique_h_geocodes[i : i + chunk_size] for i in range(0, len(unique_h_geocodes), chunk_size)
            ]

            delayed_tasks = []

            for chunk in h_geocode_chunks:
                # Create a task for each chunk
                filtered_lodes = county_lodes_df[county_lodes_df["h_geocode"].isin(chunk)]

                delayed_task = delayed(od_assign_start_end_optimized)(
                    (filtered_lodes, origin_buildings, dest_buildings)
                )
                delayed_tasks.append(delayed_task)

            # Compute all tasks for this day
            chunk_results = compute(*delayed_tasks)

            # Combine results
            if chunk_results:
                result_df = pd.concat(chunk_results, ignore_index=True)
                result_df = find_origin_dest_nodes(result_df, G)

        return result_df, origin_buildings, dest_buildings

    def assign_departure_times_by_cbg(self, result_df, census_depart_times_df, day):
        """
        Assign departure times to OD pairs while maintaining the distribution
        for each origin Census Block Group (h_geocode).
        """
        assigned_od = pd.DataFrame()

        departure_times = [col for col in census_depart_times_df.columns if "_estimate" in col and "total_" not in col]

        for h_geocode, group_df in result_df.groupby("h_geocode"):
            census_row = census_depart_times_df[census_depart_times_df["GEO_ID"] == h_geocode]

            if census_row.empty:
                self.logger.warning(f"No census data found for h_geocode {h_geocode}. Using average distribution.")
                probabilities = (
                    census_depart_times_df[departure_times].sum() / census_depart_times_df["total_estimate"].sum()
                )
            else:
                row_values = census_row[departure_times].values[0]
                total = row_values.sum() if row_values.sum() > 0 else 1.0
                probabilities = row_values / total

            prob_dict = {period: prob for period, prob in zip(departure_times, probabilities)}

            remaining_df = group_df.copy()
            group_size = len(remaining_df)

            counts_by_period = {}
            for period, prob in prob_dict.items():
                counts_by_period[period] = int(round(group_size * prob))

            total_assigned = sum(counts_by_period.values())
            if total_assigned < group_size:
                remainder = group_size - total_assigned
                fractions = {p: (group_size * prob_dict[p]) % 1 for p in departure_times}
                top_periods = sorted(departure_times, key=lambda x: fractions[x], reverse=True)[:remainder]
                for period in top_periods:
                    counts_by_period[period] += 1
            elif total_assigned > group_size:
                excess = total_assigned - group_size
                for period in sorted(departure_times, key=lambda x: prob_dict[x])[:excess]:
                    if counts_by_period[period] > 0:
                        counts_by_period[period] -= 1

            remaining_indices = remaining_df.index.tolist()
            for period, count in counts_by_period.items():
                if count <= 0 or not remaining_indices:
                    continue

                start_datetime, end_datetime = parse_time_range(day, period)

                selected_indices = np.random.choice(
                    remaining_indices, min(count, len(remaining_indices)), replace=False
                )
                remaining_indices = [idx for idx in remaining_indices if idx not in selected_indices]

                selected_df = remaining_df.loc[selected_indices].copy()

                selected_df["departure_time"] = selected_df.apply(
                    lambda _: start_datetime
                    + timedelta(seconds=random.randint(0, int((end_datetime - start_datetime).total_seconds()))),
                    axis=1,
                )

                selected_df["departure_time_secs"] = selected_df["departure_time"].apply(
                    lambda dt: dt.hour * 3600 + dt.minute * 60 + dt.second
                )
                selected_df["departure_time_str"] = selected_df["departure_time"].apply(
                    lambda dt: dt.strftime("%H:%M:%S")
                )

                assigned_od = pd.concat([assigned_od, selected_df])

        return assigned_od

    def od_assign_start_end(self, params):
        (
            county_lodes_df,
            county_geoid,
            res_build,
            com_build,
            ms_build,
        ) = params

        od_data = []

        for index, movement in county_lodes_df.iterrows():
            # Retrieve location data
            res = res_build[res_build["GEOID"] == movement["h_geocode"]]
            com = com_build[com_build["GEOID"] == movement["w_geocode"]]

            # In case of missing data, use fallback data structures
            if res.empty and hasattr(self, "ms_enabled") and self.ms_enabled:
                res = ms_build[ms_build["GEOID"] == movement["w_geocode"]]
            if com.empty and hasattr(self, "ms_enabled") and self.ms_enabled:
                com = ms_build[ms_build["GEOID"] == movement["w_geocode"]]

            if res.empty:
                res = county_geoid[county_geoid["GEOID"] == movement["w_geocode"]]
            if com.empty:
                com = county_geoid[county_geoid["GEOID"] == movement["w_geocode"]]

            # Calculate proportions for origin and destination
            num_res = len(res)
            num_com = len(com)
            repeat_res = movement["total_jobs"] // num_res if num_res else 0
            repeat_com = movement["total_jobs"] // num_com if num_com else 0
            additional_res = movement["total_jobs"] % num_res
            additional_com = movement["total_jobs"] % num_com

            # Sample residential and commercial buildings accordingly
            sampled_res = pd.concat(
                [res] * repeat_res + [res.sample(n=additional_res, random_state=42)], ignore_index=True
            )
            sampled_com = pd.concat(
                [com] * repeat_com + [com.sample(n=additional_com, random_state=42)], ignore_index=True
            )

            sampled_res = sampled_res.sample(frac=1, random_state=42).reset_index(drop=True)
            sampled_com = sampled_com.sample(frac=1, random_state=42).reset_index(drop=True)

            for i in range(movement["total_jobs"]):

                od_record = {
                    "h_geocode": movement["h_geocode"],
                    "w_geocode": movement["w_geocode"],
                    "total_jobs": movement["total_jobs"],
                    "origin_loc": sampled_res.iloc[i][["location"]].values[0],
                    "origin_loc_lat": sampled_res.iloc[i]["location"][0],
                    "origin_loc_lon": sampled_res.iloc[i]["location"][1],
                    "dest_loc": sampled_com.iloc[i][["location"]].values[0],
                    # "possible_home_loc": sampled_res["location"].values.tolist(),
                    # "possible_dest_loc": sampled_com["location"].values.tolist(),
                    "dest_loc_lat": sampled_com.iloc[i]["location"][0],
                    "dest_loc_lon": sampled_com.iloc[i]["location"][1],
                }

                # shortest_path = find_shortest_path_dict(G, od_record)
                # moving the time calculation to later - need to assign time based on the combined probabilities
                # move_time, total_distance, distance_miles = list(get_travel_time_dict(mode_type, od_record).values())
                # od_record.update(
                #     {
                #         "time_taken": move_time,
                #         "total_distance": total_distance,
                #         "distance_miles": distance_miles,
                #     }
                # )

                od_data.append(od_record)

        # Convert the list of dictionaries to a GeoDataFrame
        od_frame = gpd.GeoDataFrame(od_data)
        return od_frame

    def od_assign_time(
        self,
        od_df,
        departure_counts,
        start_datetime,
        end_datetime,
        total_travel_time_to_work,
        total_travel_time_to_work_proportion,
        travel_time_block_probabilities,
        time_blocks,
        hours_worked,
    ):
        od_time_blocks = define_time_blocks(time_blocks)
        travel_time_block_probabilities = np.array(travel_time_block_probabilities)
        travel_time_block_probabilities /= travel_time_block_probabilities.sum()

        bins = [block[0] for block in od_time_blocks] + [np.inf]

        counts, _ = np.histogram(od_df["time_taken"], bins=bins)

        bin_probabilities = np.array(travel_time_block_probabilities)
        bin_probabilities *= counts
        bin_probabilities /= bin_probabilities.sum()

        od_df["bin_index"] = np.digitize(od_df["time_taken"], bins=bins[:-1], right=True) - 1
        od_df["sample_probability"] = od_df["bin_index"].apply(lambda x: bin_probabilities[x])

        expected_sum_travel_time = total_travel_time_to_work * total_travel_time_to_work_proportion
        threshold = 0.3 * expected_sum_travel_time

        selected_od_indices = []
        current_sum_travel_time = 0
        od_df = od_df.reset_index().drop("index", axis=1)
        available_od_df = od_df.copy()

        while len(selected_od_indices) < min(departure_counts, len(available_od_df)):
            remaining_departures = departure_counts - len(selected_od_indices)
            required_average_remaining = (expected_sum_travel_time - current_sum_travel_time) / remaining_departures

            if available_od_df.empty or available_od_df["sample_probability"].sum() == 0:
                available_od_df["sample_probability"] = 1

            available_od_df["sample_probability"] /= available_od_df["sample_probability"].sum()

            if (
                current_sum_travel_time > threshold
                or len(selected_od_indices) > 0.5 * departure_counts
                or (required_average_remaining * remaining_departures) < current_sum_travel_time
            ):
                if required_average_remaining < 0:
                    inverse_weights = 1 / available_od_df["time_taken"]
                    if inverse_weights.sum() == 0:
                        inverse_weights.fillna(0.001, inplace=True)
                    available_od_df["adjusted_probability"] = available_od_df["sample_probability"] * inverse_weights
                else:
                    adjustment_factor = (required_average_remaining / available_od_df["time_taken"]).clip(0, 2)
                    available_od_df["adjusted_probability"] = available_od_df["sample_probability"] * adjustment_factor
            else:
                available_od_df["adjusted_probability"] = available_od_df["sample_probability"]

            available_od_df["adjusted_probability"] /= available_od_df["adjusted_probability"].sum()
            sampled_index = available_od_df.sample(weights="adjusted_probability", random_state=42).index[0]
            selected_od_indices.append(sampled_index)

            current_sum_travel_time += available_od_df.loc[sampled_index, "time_taken"] / 60  # seconds to minutes
            available_od_df = available_od_df.drop(sampled_index)

        selected_od_df = od_df.loc[selected_od_indices]
        remaining_od_df = od_df.drop(selected_od_indices)

        selected_od_df["departure_time"] = selected_od_df.apply(
            lambda row: start_datetime
            + timedelta(seconds=random.randint(0, int((end_datetime - start_datetime).total_seconds()))),
            axis=1,
        )
        selected_od_df["departure_time_secs"] = selected_od_df["departure_time"].apply(
            lambda dt: dt.hour * 3600 + dt.minute * 60 + dt.second
        )
        selected_od_df["departure_time_str"] = selected_od_df["departure_time"].apply(
            lambda dt: dt.strftime("%H:%M:%S")
        )

        # TODO: check if we need to add return time

        # stay_duration_samples = sample_gaussian_dist(hours_worked, 1)
        # stay_duration_secs = stay_duration_samples[0] * 3600  # Convert hours to seconds

        # selected_od_df["return_time"] = selected_od_df.apply(
        #     lambda row: row["departure_time"] + timedelta(seconds=row["time_taken"] + stay_duration_secs), axis=1
        # )

        # delta = int((end_datetime - start_datetime).total_seconds())
        # random_second = random.randint(0, delta)
        # final_departure_time = start_datetime + timedelta(seconds=random_second)
        # seconds_from_midnight = start_datetime.hour * 3600 + start_datetime.minute * 60 + random_second
        # "departure_time": final_departure_time,
        # "departure_time_secs": seconds_from_midnight,
        # "departure_time_str": final_departure_time.strftime("%H:%M:%S"),

        return selected_od_df, remaining_od_df

    def main(
        self,
        county_geoid_df,
        res_locations,
        combined_work_locations,
        ms_buildings_df,
        county_lodes_df,
        state_fips,
        county_fips,
        G,
        hourly_graphs,
        block_groups,
    ):

        np.random.seed(42)
        random.seed(42)

        census_depart_times_path = f"{self.output_path}/census_data/census_depart_times.parquet"
        travel_time_to_work_path = f"{self.output_path}/census_data/travel_time_to_work.parquet"

        if os.path.exists(census_depart_times_path) and os.path.exists(travel_time_to_work_path):
            self.logger.info("Loading cached census data from parquet files.")
            census_depart_times_df = pd.read_parquet(census_depart_times_path)
            travel_time_to_work_df = pd.read_parquet(travel_time_to_work_path)
        else:
            self.logger.info("Census data CSVs not found. Fetching from API.")
            census_depart_times_df = get_census_data_wrapper(
            table="B08302",
            api_url="https://api.census.gov/data/2021/acs/acs5",
            state_fips=state_fips,
            county_fips=county_fips,
            block_groups=block_groups,
            county_only=False,
            )
            travel_time_to_work_df = get_census_travel_time_data(
            table="B08303",
            api_url="https://api.census.gov/data/2021/acs/acs5",
            state_fips=state_fips,
            county_fips=county_fips,
            block_groups=block_groups,
            county_only=False,
            )
            os.makedirs(f"{self.output_path}/census_data", exist_ok=True)
        
        census_depart_times_df.to_parquet(f"{self.output_path}/census_data/census_depart_times.parquet")
        travel_time_to_work_df.to_parquet(f"{self.output_path}/census_data/travel_time_to_work.parquet")

        days = sorted(set(day for day in self.datetime_ranges))
        # days = list(range(1))

        # delayed_tasks = []
        results = []

        total_census_departs = census_depart_times_df["total_estimate"].sum()
        time_columns = [col for col in census_depart_times_df.columns if "estimate" in col and "total" not in col]
        census_depart_times_current_df = census_depart_times_df[time_columns]
        departure_times = census_depart_times_current_df.columns.tolist()

        # current_totals = county_lodes_df.groupby("h_geocode")["total_jobs"].sum().reset_index()
        # current_totals = current_totals.rename(columns={"total_jobs": "current_total_jobs"})

        # b08302_totals = census_depart_times_df.groupby("GEO_ID")["total_estimate"].sum().reset_index()
        # b08302_totals = b08302_totals.rename(
        #     columns={"GEO_ID": "h_geocode", "total_estimate": "b08302_total_estimate"}
        # )

        # merged_lodes_acs_df = pd.merge(current_totals, b08302_totals, on="h_geocode", how="inner")
        # merged_lodes_acs_df["scaling_factor"] = (
        #     merged_lodes_acs_df["b08302_total_estimate"] / merged_lodes_acs_df["current_total_jobs"]
        # )
        # county_lodes_df = pd.merge(
        #     county_lodes_df, merged_lodes_acs_df[["h_geocode", "scaling_factor"]], on="h_geocode", how="left"
        # )
        # county_lodes_df["total_jobs"] = (
        #     (county_lodes_df["total_jobs"] * county_lodes_df["scaling_factor"]).round().astype(int)
        # )

        # 1. Aggregate current total jobs per h_geocode
        current_totals = county_lodes_df.groupby("h_geocode")["total_jobs"].sum().reset_index()
        current_totals = current_totals.rename(columns={"total_jobs": "current_total_jobs"})

        # 2. Aggregate census total estimate per GEO_ID
        b08302_totals = census_depart_times_df.groupby("GEO_ID")["total_estimate"].sum().reset_index()
        b08302_totals = b08302_totals.rename(
            columns={"GEO_ID": "h_geocode", "total_estimate": "b08302_total_estimate"}
        )

        # 3. Merge to compute scaling factor
        merged_lodes_acs_df = pd.merge(current_totals, b08302_totals, on="h_geocode", how="inner")
        merged_lodes_acs_df["scaling_factor"] = (
            merged_lodes_acs_df["b08302_total_estimate"] / merged_lodes_acs_df["current_total_jobs"]
        )

        # 4. Merge scaling factor back to county_lodes_df
        county_lodes_df = pd.merge(
            county_lodes_df, merged_lodes_acs_df[["h_geocode", "scaling_factor"]], on="h_geocode", how="left"
        )

        # 5. Scale total_jobs
        county_lodes_df["total_jobs"] = (
            (county_lodes_df["total_jobs"] * county_lodes_df["scaling_factor"]).round().astype(int)
        )
        county_lodes_df.to_csv(f"{self.output_path}/census_data/county_lodes_adjusted.csv")

        # create dict of average_speeds for each timestamp, pass to od_assign_start_end

        for day in days:
            result_df, origin_buildings, dest_buildings = self.process_county_data(
                county_lodes_df,
                county_geoid_df,
                res_locations,
                combined_work_locations,
                ms_buildings_df,
                G,
                days,
            )

            # Assign departure times using our new function
            assigned_od = self.assign_departure_times_by_cbg(result_df, census_depart_times_df, day)

            # Validate results
            original_od_count = result_df.shape[0]
            if len(assigned_od) != original_od_count:
                self.logger.warning(
                    f"Warning: Not all ODs were assigned. {original_od_count - len(assigned_od)} ODs remain unassigned."
                )

            # if len(result_df) > 0:
            #     self.logger.warning(f"Warning: Not all ODs were assigned. {len(result_df)} ODs remain unassigned.")

            results.append((day, assigned_od))
            # Nothing is written here -- the frame is returned to the caller.
            self.logger.info(f"Processed day {day}")

        try:
            save_building_dictionaries(origin_buildings, dest_buildings, self.output_path)
        except Exception as _e:
            self.logger.warning(f"Could not save building dictionaries (non-critical): {_e}")

        lodes_output_dfs = []
        days = []
        for result in results:
            day, df = result            
            lodes_output_dfs.append(df)
            days.append(day)

        return lodes_output_dfs, days, travel_time_to_work_df, census_depart_times_df
