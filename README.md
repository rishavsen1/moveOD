<!-- README: user-focused web app guide. Short, practical, avoids backend/API internals. -->

# MoveOD: Synthesizing Origin-Destination Commute Distribution from U.S. Census Data

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](https://linux.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MoveOD generates calibrated origin-destination commute data for a U.S. state and county using Census, road network, and building-footprint data. The project is designed to run either from the command line or through a Streamlit web app.

You can access the paper here: [ArXiv](https://arxiv.org/abs/2510.18858)

<p align="center">
  <img src="files/moveod_pipeline.png" alt="MoveOD pipeline" width="800" />
</p>

## Overview

MoveOD is a transportation data generation pipeline that combines public spatial and commute data to synthesize origin-destination trip tables.

### What it does

1. Processes LODES employment flows from the U.S. Census.
2. Integrates building footprints from Microsoft Global Buildings and OpenStreetMap.
3. Routes trips on road graphs using OSM defaults or INRIX speeds when available.
4. Calibrates generated trips with Integer Linear Programming.
5. Produces county-level OD outputs with sampled locations and departure times.

## Quickstart

### 1. Requirements

- Python 3.12 on Linux.
- A free U.S. Census API key for LODES and geographic downloads.

### 2. Configure the Census API key

1. Get a free key from [Census API Key Signup](https://api.census.gov/data/key_signup.html).
2. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

3. Edit `.env` and replace `YOUR_CENSUS_API_KEY_HERE` with your key.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you want to use the backend web app, install its extra dependencies too:

```bash
pip install -r backend/requirements.txt
```

Optional: if you have IBM CPLEX installed, set its path before running the pipeline to speed up ILP calibration.

```bash
export CPLEX_PATH=</path/to/cplex>
```

### 4. Start the app

Command line:

```bash
python cli.py -i
```

Streamlit:

```bash
streamlit run app.py
```



<p align="center">
  <img src="files/moveod_interface.png" alt="MoveOD interface" width="800" />
</p>

## Using the web app

1. Select the state and county you want to process.
2. Choose the available data sources for the region.
3. Add optional inputs such as INRIX speeds if you have them.
4. Pick a date or date range when using time-dependent traffic data.
5. Click the start button to run the pipeline.
6. Watch progress in the UI and download the output when it finishes.

Typical runtime is a few minutes. Hamilton County, TN (~158,000 commuters)
takes about 5 minutes end to end on a 32-thread machine, with a peak memory of
roughly 6 GB. Routing runs in parallel across all available cores and is
throttled automatically if memory is tight.

## Output

Outputs are written under `move_OD/{state}/{county}/{start_date}_{end_date}/`.

Key results include:

- `calibrated_move_od/{day}.csv` for the final OD trips. **One row is one trip.**
- `intermediate/{day}/routing_df.parquet` for routed trips.
- `intermediate/{day}/post_mssr_routing_df.parquet` after the mean road speed shift.
- `intermediate/{day}/hourly_graphs_adjusted.json` for adjusted graphs. Only
  written when INRIX speeds are supplied; without them the speed shift is
  uniform, so the shifted travel times are derived arithmetically rather than
  by re-routing.

Typical columns include origin and destination GEOIDs, sampled home and work coordinates, departure time, and estimated travel time.

## Reproducibility

The paper figures can be regenerated from the output directory after a run:

```bash
python analysis/figures_from_output.py --state Tennessee --county Hamilton
```

The notebook in `analysis/figures_1_2.ipynb` shows the same workflow for Figures 1 and 2. Figure 3 in the paper is a downstream demonstration of how the generated data can be used.

Because road speeds may be sampled and INRIX is optional, reruns are expected to be qualitatively similar rather than byte-for-byte identical.

<p align="center">
  <img src="files/moveod_plots.png" alt="MoveOD plots" width="800" />
</p>

## Troubleshooting

If you see Streamlit warnings about missing script context during parallel processing, they are expected and harmless.

If you want to force sequential execution, set:

```bash
export FORCE_SEQUENTIAL_OSM=true
```

If OSM queries time out, try a smaller county, wait a few minutes, or rerun during off-peak hours. The pipeline already retries failed requests and falls back to sequential processing when needed.

## Citation

If you use MoveOD in academic work, please cite the project:

```bibtex
@misc{sen2025moveodsynthesizingorigindestinationcommute,
      title={MoveOD: Synthesizing Origin-Destination Commute Distribution from U.S. Census Data},
      author={Rishav Sen and Abhishek Dubey and Ayan Mukhopadhyay and Samitha Samaranayake and Aron Laszka},
      year={2025},
      eprint={2510.18858},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2510.18858},
}
```

## License

MIT. See the `LICENSE` file.
