# MOVE-OD User Guide

## Complete Guide to Origin-Destination Data Generation

Welcome to MOVE-OD, a comprehensive tool for generating high-quality origin-destination (OD) point-to-point data from multiple data sources including LODES, SafeGraph, Microsoft Buildings, and OpenStreetMap.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Web Application Usage](#web-application-usage)
4. [Data Sources](#data-sources)
5. [Configuration Options](#configuration-options)
6. [Understanding the Output](#understanding-the-output)
7. [Visualization Features](#visualization-features)
8. [Common Workflows](#common-workflows)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Overview

### What is MOVE-OD?

MOVE-OD (Mobility Origin-Destination) is a tool that generates calibrated origin-destination data for transportation analysis. It combines:

- **LODES (Longitudinal Employer-Household Dynamics)**: Commute flow data
- **SafeGraph**: Points of interest and building footprints
- **Microsoft Buildings**: Comprehensive building footprint data
- **OpenStreetMap**: Detailed location information
- **INRIX**: Traffic speed data for calibration
- **TIGER/Line**: Census geographic boundaries

### Key Features

✅ **Multi-Source Integration**: Combines multiple authoritative data sources  
✅ **Web-Based Interface**: Modern, user-friendly web application  
✅ **Real-Time Progress**: Live updates during processing  
✅ **Interactive Maps**: Visualize origins, destinations, and heatmaps  
✅ **Calibrated Output**: Data calibrated against real traffic speeds  
✅ **Flexible Downloads**: Export results in multiple formats

---

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/move_od.git
cd move_od

# Install dependencies
pip install -r requirements.txt

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Starting the Application

**Linux/Mac:**

```bash
chmod +x start_webapp_fixed.sh
./start_webapp_fixed.sh
```

**Windows:**

```batch
start_webapp.bat
```

**Docker:**

```bash
docker-compose up -d
```

The application will be available at: `http://localhost:8080`

---

## Web Application Usage

### Step 1: Select Location

1. **Choose a State**: Select from the dropdown menu
2. **Choose a County**: Pick your county of interest
3. The output path will automatically update

![Location Selection](images/location-selection.png)

### Step 2: Configure Date Range

- **Start Date**: Beginning of the analysis period
- **End Date**: End of the analysis period
- Dates should match your INRIX data availability

### Step 3: Select Data Years

- **LODES Year**: Year for employment data (2019, 2020, 2021)
- **TIGER/Line Year**: Year for census boundaries (2019-2022)

### Step 4: Provide Data Paths

- **INRIX Data Path**: Path to your INRIX speed data
- **INRIX Conversion Path**: Path to TMC-to-road conversion data
- **Output Path**: Where results will be saved (auto-generated)

### Step 5: Choose Data Sources

Toggle the data sources you want to use:

- ☑️ **Microsoft Buildings**: High-quality building footprints
- ☑️ **SafeGraph**: Points of interest data

### Step 6: Begin Processing

Click **"🚀 Begin Processing"** to start. You'll see:

- Real-time status updates
- Progress percentage
- Current processing step
- Estimated completion time

### Step 7: View Results

Once complete, you'll see:

- **Total Origins**: Number of residential locations
- **Total Destinations**: Number of workplace locations
- **Census Block Groups**: Geographic units processed
- **Interactive Map**: Visualize your data

---

## Data Sources

### LODES (Census Bureau)

**What it provides**: Origin-destination flow data for workers

- Job counts between home and work locations
- Broken down by census blocks
- Annual updates
- Covers all US states

**Downloaded automatically by the tool**

### SafeGraph

**What it provides**: Point of interest data

- Building footprints
- Business locations
- Visit patterns
- Category information

**Requires**: SafeGraph data subscription

### Microsoft Buildings

**What it provides**: Building footprints

- High-precision polygons
- Covers entire US
- Regularly updated
- Open data

**Downloaded automatically by the tool**

### OpenStreetMap

**What it provides**: Detailed location data

- Roads and paths
- Buildings
- Land use
- Points of interest

**Queried automatically via Overpass API**

### INRIX

**What it provides**: Traffic speed data for calibration

- Real-world travel times
- Speed observations
- TMC (Traffic Message Channel) segments

**Requires**: INRIX data license

---

## Configuration Options

### Basic Configuration

| Option     | Description         | Default         |
| ---------- | ------------------- | --------------- |
| State      | US State            | None (required) |
| County     | County within state | None (required) |
| Start Date | Analysis start date | None (required) |
| End Date   | Analysis end date   | None (required) |

### Advanced Configuration

| Option           | Description              | Default    |
| ---------------- | ------------------------ | ---------- |
| LODES Year       | Employment data year     | 2019       |
| TIGER Year       | Census boundary year     | 2019       |
| Use MS Buildings | Include MS building data | ☑️ Enabled |
| Use SafeGraph    | Include SafeGraph data   | ☑️ Enabled |

### Data Paths

The tool automatically generates appropriate paths, but you can customize:

- **INRIX Data Path**: `./data/inrix/[county]_[dates].csv`
- **INRIX Conversion**: `./data/inrix/conversion.csv`
- **Output Path**: `./outputs/[state]_[county]_[dates]/`

---

## Understanding the Output

### Directory Structure

```
outputs/
└── [State]_[County]_[StartDate]_[EndDate]/
    ├── origins.geojson           # Residential locations
    ├── destinations.geojson       # Workplace locations
    ├── census_blocks.geojson      # Census geometries
    ├── calibrated_od_matrix.csv   # Calibrated OD flows
    ├── uncalibrated_od_matrix.csv # Raw OD flows
    └── metadata.json              # Processing metadata
```

### File Descriptions

#### origins.geojson

GeoJSON file containing residential (home) locations:

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [-85.2345, 34.5678]
  },
  "properties": {
    "GEOID": "470650001001",
    "population": 150,
    "building_count": 45
  }
}
```

#### destinations.geojson

GeoJSON file containing workplace locations:

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [-85.2456, 34.5789]
  },
  "properties": {
    "GEOID": "470650001002",
    "jobs": 250,
    "poi_count": 12
  }
}
```

#### calibrated_od_matrix.csv

Calibrated origin-destination flow matrix:

```csv
origin_geoid,destination_geoid,flow,distance,travel_time,calibrated_flow
470650001001,470650001002,45,5.2,12.5,42
470650001001,470650001003,23,8.1,18.3,21
...
```

---

## Visualization Features

### Map Types

#### 1. Both (Origins + Destinations)

- 🟢 **Green markers**: Residential origins
- 🔴 **Red markers**: Workplace destinations
- Best for understanding spatial distribution

#### 2. Origins Only

- Shows only residential locations
- Useful for housing density analysis

#### 3. Destinations Only

- Shows only workplace locations
- Useful for employment center analysis

#### 4. Heatmap

- Density visualization
- Color gradient from blue (low) to red (high)
- Great for identifying clusters

### Interactive Map Controls

- **Zoom**: Mouse wheel or +/- buttons
- **Pan**: Click and drag
- **Popup**: Click any marker for details
- **Sample Size**: Adjust slider to show more/fewer points (100-10,000)
- **Update Map**: Refresh visualization after changing settings

---

## Common Workflows

### Workflow 1: Basic County Analysis

**Goal**: Generate OD data for a single county

1. Select state and county
2. Set date range (e.g., 2019-01-01 to 2019-12-31)
3. Use default settings
4. Click "Begin Processing"
5. Wait for completion (~10-30 minutes)
6. Download results

### Workflow 2: Multi-County Comparison

**Goal**: Compare multiple counties in the same state

For each county:

1. Process as in Workflow 1
2. Download results
3. Use external tools (QGIS, Python) to compare

### Workflow 3: Temporal Analysis

**Goal**: Analyze changes over time

1. Process same county with different date ranges
2. Compare calibrated OD matrices
3. Identify seasonal patterns or long-term trends

### Workflow 4: Integration with SUMO

**Goal**: Use data in traffic simulation

1. Generate OD data
2. Download calibrated files
3. Convert to SUMO format using provided scripts
4. Run traffic simulation

---

## Troubleshooting

### Common Issues

#### ❌ "Failed to load states"

**Cause**: Backend not running or connection error

**Solution**:

```bash
# Check if backend is running
curl http://localhost:8000/

# Restart backend
cd backend
python app.py
```

#### ❌ "Processing failed" or stuck at certain percentage

**Cause**: Data source timeout or missing data

**Solution**:

- Check internet connection
- Verify INRIX data paths
- Review backend logs: `backend/backend.log`

#### ❌ "No map data available"

**Cause**: Processing incomplete or no results

**Solution**:

- Ensure processing completed (100%)
- Check output directory exists
- Verify data files were created

#### ❌ Destinations not showing in red

**Cause**: Browser cache or map rendering issue

**Solution**:

- Clear browser cache
- Hard refresh (Ctrl+Shift+R / Cmd+Shift+R)
- Increase sample size slider
- Check browser console for errors (F12)

#### ❌ Heatmap not displaying

**Cause**: Missing Leaflet.heat plugin or insufficient data

**Solution**:

- Verify internet connection (plugin loaded from CDN)
- Increase sample size
- Switch to "Both" view first to verify data exists

### Getting Help

1. **Check Documentation**: Review this guide and troubleshooting docs
2. **Backend Logs**: Check `backend/backend.log` for detailed errors
3. **Browser Console**: Open DevTools (F12) and check console
4. **GitHub Issues**: Report bugs or ask questions

<!-- ---

## API Reference

### REST Endpoints

#### GET `/api/states`

Get all available states and counties.

**Response**:

```json
{
  "states": [
    {
      "name": "Tennessee",
      "fips": "47",
      "counties": [
        { "name": "Hamilton County", "fips": "065" },
        { "name": "Davidson County", "fips": "037" }
      ]
    }
  ]
}
```

#### POST `/api/process`

Start OD data processing.

**Request Body**:

```json
{
  "state": "Tennessee",
  "county": "Hamilton County",
  "start_date": "2019-01-01",
  "end_date": "2019-12-31",
  "lodes_year": 2019,
  "tiger_year": 2019,
  "inrix_path": "./data/inrix/data.csv",
  "inrix_conversion_path": "./data/inrix/conversion.csv",
  "use_ms_buildings": true,
  "use_safegraph": true,
  "output_path": "./outputs/TN_Hamilton_2019/"
}
```

**Response**:

```json
{
  "job_id": "abc123",
  "status": "processing"
}
```

#### GET `/api/job/{job_id}`

Get job status and progress.

**Response**:

```json
{
  "job_id": "abc123",
  "status": "processing",
  "progress": 0.65,
  "message": "Processing destinations...",
  "result": null
}
```

#### GET `/api/map-data/{job_id}`

Get map visualization data.

**Parameters**:

- `data_type`: "both", "origins", "destinations", or "heatmap"
- `sample_size`: Number of points (100-10000)

**Response**: GeoJSON FeatureCollection

#### GET `/api/download/{job_id}`

Download all results as ZIP file.

**Response**: Binary ZIP file

--- -->

## Best Practices

### 1. Data Selection

- ✅ Use the most recent LODES year available
- ✅ Match TIGER year to LODES year when possible
- ✅ Enable both MS Buildings and SafeGraph for best results
- ✅ Ensure date range matches INRIX data availability

### 2. Performance Optimization

- ✅ Start with a small county for testing
- ✅ Use smaller sample sizes for initial map viewing
- ✅ Process during off-peak hours for large counties
- ✅ Ensure sufficient disk space (5-10 GB per county)

### 3. Quality Assurance

- ✅ Compare origin/destination counts with census data
- ✅ Verify geographic distribution makes sense
- ✅ Check calibrated flows against known patterns
- ✅ Validate travel times are reasonable

### 4. Data Management

- ✅ Organize outputs by date and location
- ✅ Keep metadata files with results
- ✅ Document any custom configurations
- ✅ Backup important results

---

<!--
## Advanced Topics

### Custom Data Integration

You can integrate custom data sources by:

1. Modifying `generate/` scripts
2. Adding new data readers in `backend/app.py`
3. Extending the processing pipeline

See `MIGRATION_GUIDE.md` for details.

### Batch Processing

For processing multiple counties:

```python
import requests

counties = [
    ("Tennessee", "Hamilton County"),
    ("Tennessee", "Davidson County"),
    ("Tennessee", "Knox County")
]

for state, county in counties:
    response = requests.post("http://localhost:8000/api/process", json={
        "state": state,
        "county": county,
        # ... other parameters
    })
    print(f"Started job for {county}: {response.json()['job_id']}")
```

### Custom Calibration

Modify calibration parameters in `calibration.ipynb`:

- Distance decay functions
- Travel time estimation
- Flow adjustment factors

--- -->

## Citation

If you use MOVE-OD in your research, please cite:

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

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<!--
## Support

- **Documentation**: [GitHub Wiki](https://github.com/yourusername/move_od/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/move_od/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/move_od/discussions)

--- -->

## Acknowledgments

This tool integrates data from:

- US Census Bureau (LODES, TIGER/Line)
- Microsoft (Building Footprints)
- OpenStreetMap Contributors
- SafeGraph
- INRIX

---

_Last Updated: October 23, 2025_
