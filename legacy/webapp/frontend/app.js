// Configuration
// Auto-detect API URL based on environment
const API_BASE_URL =
  window.location.hostname === "localhost" && window.location.port === "8080"
    ? "http://localhost:8000" // Development mode
    : ""; // Production mode (uses same host via nginx proxy)
const POLL_INTERVAL = 2000; // Poll every 2 seconds

// Global state
let currentJobId = null;
let pollInterval = null;
let map = null;
let currentMarkers = [];

// DOM Elements
const elements = {
  stateSelect: document.getElementById("state-select"),
  countySelect: document.getElementById("county-select"),
  startDate: document.getElementById("start-date"),
  endDate: document.getElementById("end-date"),
  lodesYear: document.getElementById("lodes-year"),
  tigerYear: document.getElementById("tiger-year"),
  inrixPath: document.getElementById("inrix-path"),
  inrixConversionPath: document.getElementById("inrix-conversion-path"),
  useMsBuildings: document.getElementById("use-ms-buildings"),
  useSafegraph: document.getElementById("use-safegraph"),
  outputPath: document.getElementById("output-path"),
  beginBtn: document.getElementById("begin-btn"),
  resetBtn: document.getElementById("reset-btn"),
  downloadBtn: document.getElementById("download-btn"),
  updateMapBtn: document.getElementById("update-map-btn"),
  configSection: document.getElementById("config-section"),
  progressSection: document.getElementById("progress-section"),
  resultsSection: document.getElementById("results-section"),
  progressStatus: document.getElementById("progress-status"),
  progressPercentage: document.getElementById("progress-percentage"),
  progressFill: document.getElementById("progress-fill"),
  progressMessage: document.getElementById("progress-message"),
  errorSection: document.getElementById("error-section"),
  errorMessage: document.getElementById("error-message"),
  originsCount: document.getElementById("origins-count"),
  destinationsCount: document.getElementById("destinations-count"),
  blockGroupsCount: document.getElementById("block-groups-count"),
  mapType: document.getElementById("map-type"),
  sampleSize: document.getElementById("sample-size"),
  sampleSizeValue: document.getElementById("sample-size-value"),
  mapLegendText: document.getElementById("map-legend-text"),
};

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  initializeApp();
  setupEventListeners();
});

async function initializeApp() {
  try {
    await loadStates();
  } catch (error) {
    console.error("Failed to initialize app:", error);
    showError(
      "Failed to load states and counties. Please check your backend connection."
    );
  }
}

function setupEventListeners() {
  elements.stateSelect.addEventListener("change", handleStateChange);
  elements.countySelect.addEventListener("change", updateOutputPath);
  elements.startDate.addEventListener("change", updateOutputPath);
  elements.endDate.addEventListener("change", updateOutputPath);
  elements.beginBtn.addEventListener("click", handleBeginProcessing);
  elements.resetBtn.addEventListener("click", handleReset);
  elements.downloadBtn.addEventListener("click", handleDownload);
  elements.updateMapBtn.addEventListener("click", updateMap);
  elements.sampleSize.addEventListener("input", (e) => {
    elements.sampleSizeValue.textContent = e.target.value;
  });
  elements.mapType.addEventListener("change", updateMapLegend);
}

// API Functions
async function loadStates() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/states`);
    if (!response.ok) throw new Error("Failed to fetch states");

    const data = await response.json();
    populateStates(data.states);
  } catch (error) {
    console.error("Error loading states:", error);
    throw error;
  }
}

function populateStates(states) {
  elements.stateSelect.innerHTML =
    '<option value="">Select a state...</option>';

  states.forEach((state) => {
    const option = document.createElement("option");
    option.value = state.name;
    option.textContent = state.name;
    option.dataset.stateId = state.id;
    option.dataset.stateFips = state.fips;
    option.dataset.counties = JSON.stringify(state.counties);
    elements.stateSelect.appendChild(option);
  });
}

function handleStateChange() {
  const selectedOption =
    elements.stateSelect.options[elements.stateSelect.selectedIndex];

  if (selectedOption.value) {
    const counties = JSON.parse(selectedOption.dataset.counties || "[]");
    populateCounties(counties);
    elements.countySelect.disabled = false;
  } else {
    elements.countySelect.innerHTML =
      '<option value="">Select a state first</option>';
    elements.countySelect.disabled = true;
  }

  updateOutputPath();
}

function populateCounties(counties) {
  elements.countySelect.innerHTML =
    '<option value="">Select a county...</option>';

  counties.forEach((county) => {
    const option = document.createElement("option");
    option.value = county.name;
    option.textContent = county.name;
    option.dataset.fips = county.fips;
    elements.countySelect.appendChild(option);
  });
}

function updateOutputPath() {
  const state = elements.stateSelect.value;
  const county = elements.countySelect.value;
  const startDate = elements.startDate.value;
  const endDate = elements.endDate.value;

  if (state && county && startDate && endDate) {
    elements.outputPath.value = `./move_OD/${state}/${county}/${startDate}_${endDate}`;
  } else {
    elements.outputPath.value = "";
  }
}

async function handleBeginProcessing() {
  // Validation
  if (!elements.stateSelect.value || !elements.countySelect.value) {
    showError("Please select both state and county");
    return;
  }

  if (!elements.startDate.value || !elements.endDate.value) {
    showError("Please select start and end dates");
    return;
  }

  // Prepare request
  const requestData = {
    state: elements.stateSelect.value,
    county: elements.countySelect.value,
    start_date: elements.startDate.value,
    end_date: elements.endDate.value,
    lodes_year: elements.lodesYear.value,
    tiger_shapefile_year: elements.tigerYear.value,
    inrix_path: elements.inrixPath.value || null,
    inrix_conversion_path: elements.inrixConversionPath.value || null,
    use_safegraph: elements.useSafegraph
      ? elements.useSafegraph.checked
      : false,
    use_ms_buildings: elements.useMsBuildings
      ? elements.useMsBuildings.checked
      : true,
    od_option: "Origin and Destination in same County",
  };

  try {
    // Disable button
    elements.beginBtn.disabled = true;

    // Start processing
    const response = await fetch(`${API_BASE_URL}/api/process`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to start processing");
    }

    const data = await response.json();
    currentJobId = data.job_id;

    // Show progress section
    elements.configSection.style.display = "none";
    elements.progressSection.style.display = "block";
    elements.errorSection.style.display = "none";

    // Start polling
    startPolling();
  } catch (error) {
    console.error("Error starting processing:", error);
    showError(error.message);
    elements.beginBtn.disabled = false;
  }
}

function startPolling() {
  if (pollInterval) {
    clearInterval(pollInterval);
  }

  pollInterval = setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/job/${currentJobId}`);
      if (!response.ok) throw new Error("Failed to get job status");

      const job = await response.json();
      updateProgress(job);

      if (job.status === "completed") {
        clearInterval(pollInterval);
        showResults(job);
      } else if (job.status === "failed") {
        clearInterval(pollInterval);
        showError(job.error || "Processing failed");
      }
    } catch (error) {
      console.error("Error polling job status:", error);
    }
  }, POLL_INTERVAL);
}

function updateProgress(job) {
  const percentage = Math.round(job.progress * 100);

  elements.progressStatus.textContent = job.status;
  elements.progressPercentage.textContent = `${percentage}%`;
  elements.progressFill.style.width = `${percentage}%`;
  elements.progressMessage.textContent = job.message;

  // Update status color
  if (job.status === "completed") {
    elements.progressStatus.style.color = "#28a745";
  } else if (job.status === "failed") {
    elements.progressStatus.style.color = "#dc3545";
  } else {
    elements.progressStatus.style.color = "#667eea";
  }
}

function showResults(job) {
  elements.progressSection.style.display = "none";
  elements.resultsSection.style.display = "block";

  // Update metrics
  const result = job.result;
  elements.originsCount.textContent = result.origins_count.toLocaleString();
  elements.destinationsCount.textContent =
    result.destinations_count.toLocaleString();
  elements.blockGroupsCount.textContent =
    result.census_block_groups.toLocaleString();

  // Initialize map
  initializeMap();
  updateMap();
}

function initializeMap() {
  if (map) {
    map.remove();
  }

  // Create map centered on USA
  map = L.map("map").setView([37.0902, -95.7129], 4);

  // Add tile layer
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(map);
}

async function updateMap() {
  if (!map || !currentJobId) return;

  try {
    elements.updateMapBtn.disabled = true;
    elements.updateMapBtn.textContent = "Loading...";

    // Clear existing markers
    currentMarkers.forEach((marker) => map.removeLayer(marker));
    currentMarkers = [];

    // Get map data
    const mapType = elements.mapType.value;
    const sampleSize = elements.sampleSize.value;

    const response = await fetch(
      `${API_BASE_URL}/api/map-data/${currentJobId}?data_type=${mapType}&sample_size=${sampleSize}`
    );

    if (!response.ok) throw new Error("Failed to load map data");

    const geojson = await response.json();

    if (geojson.features.length === 0) {
      showError("No map data available");
      return;
    }

    console.log(
      `Loaded ${geojson.features.length} features for map type: ${mapType}`
    );

    // Calculate bounds
    let bounds = [];

    if (mapType === "heatmap") {
      // Create heatmap
      const heatData = geojson.features.map((feature) => {
        const coords = feature.geometry.coordinates;
        return [coords[1], coords[0], 1]; // [lat, lng, intensity]
      });

      console.log(`Creating heatmap with ${heatData.length} points`);

      const heat = L.heatLayer(heatData, {
        radius: 15,
        blur: 10,
        maxZoom: 17,
        gradient: {
          0.4: "blue",
          0.6: "cyan",
          0.7: "lime",
          0.8: "yellow",
          1.0: "red",
        },
      }).addTo(map);

      currentMarkers.push(heat);

      // Get bounds from heat data
      bounds = heatData.map((point) => [point[0], point[1]]);
    } else {
      // Add markers
      let originCount = 0;
      let destinationCount = 0;

      geojson.features.forEach((feature) => {
        const coords = feature.geometry.coordinates;
        const type = feature.properties.type;

        // Count types for debugging
        if (type === "origin") originCount++;
        if (type === "destination") destinationCount++;

        const isOrigin = type === "origin";
        const markerColor = isOrigin ? "#2ecc71" : "#e74c3c"; // Green for origins, Red for destinations

        const marker = L.circleMarker([coords[1], coords[0]], {
          radius: 4,
          fillColor: markerColor,
          color: markerColor,
          weight: 1,
          opacity: 0.8,
          fillOpacity: 0.6,
        }).addTo(map);

        marker.bindPopup(
          isOrigin ? "Origin (Residential)" : "Destination (Workplace)"
        );

        currentMarkers.push(marker);
        bounds.push([coords[1], coords[0]]);
      });

      console.log(
        `Added ${originCount} origins (green) and ${destinationCount} destinations (red)`
      );
    }

    // Fit bounds
    if (bounds.length > 0) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  } catch (error) {
    console.error("Error updating map:", error);
    showError("Failed to load map data");
  } finally {
    elements.updateMapBtn.disabled = false;
    elements.updateMapBtn.textContent = "Update Map";
  }
}

function updateMapLegend() {
  const mapType = elements.mapType.value;

  let legendText = "";
  switch (mapType) {
    case "both":
      legendText =
        "🟢 Green: Origins (Residential) | 🔴 Red: Destinations (Workplace)";
      break;
    case "origins":
      legendText = "🟢 Green: Origins (Residential)";
      break;
    case "destinations":
      legendText = "🔴 Red: Destinations (Workplace)";
      break;
    case "heatmap":
      legendText =
        "🔥 Heatmap: Density visualization of origins and destinations";
      break;
  }

  elements.mapLegendText.textContent = legendText;
}

async function handleDownload() {
  if (!currentJobId) return;

  try {
    elements.downloadBtn.disabled = true;
    elements.downloadBtn.textContent = "Preparing Download...";

    // Download file
    const response = await fetch(
      `${API_BASE_URL}/api/download/${currentJobId}`
    );
    if (!response.ok) throw new Error("Failed to download results");

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `calibrated_move_od_${currentJobId}.zip`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (error) {
    console.error("Error downloading results:", error);
    showError("Failed to download results");
  } finally {
    elements.downloadBtn.disabled = false;
    elements.downloadBtn.textContent = "📥 Download All Calibrated Files (ZIP)";
  }
}

function handleReset() {
  // Clear job
  currentJobId = null;

  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }

  // Clear map
  if (map) {
    currentMarkers.forEach((marker) => map.removeLayer(marker));
    currentMarkers = [];
  }

  // Reset UI
  elements.resultsSection.style.display = "none";
  elements.progressSection.style.display = "none";
  elements.configSection.style.display = "block";
  elements.beginBtn.disabled = false;
  elements.errorSection.style.display = "none";
}

function showError(message) {
  elements.errorSection.style.display = "block";
  elements.errorMessage.textContent = message;

  // Auto-hide after 5 seconds
  setTimeout(() => {
    elements.errorSection.style.display = "none";
  }, 5000);
}
