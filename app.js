const TILE_SERVERS = {
  Aucun: null,
  OpenStreetMap: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  "Satellite ESRI": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  "CyclOSM (FR)": "https://a.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png",
  "CyclOSM Forest": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
  OpenSnowMap: "https://tiles.opensnowmap.org/pistes/{z}/{x}/{y}.png",
};

const state = {
  map: null,
  baseLayer: null,
  marker: null,
  fullPath: null,
  progressPath: null,
  compass: document.getElementById("compass"),
  data: [],
  totalDuration: 0,
  animationId: null,
  startTime: 0,
  isPlaying: false,
  charts: {},
  recorder: null,
  recordingChunks: [],
  captureStream: null,
};

const controls = {
  gpxFile: document.getElementById("gpxFile"),
  clipDuration: document.getElementById("clipDuration"),
  smoothing: document.getElementById("smoothing"),
  tileSelect: document.getElementById("tileSelect"),
  playBtn: document.getElementById("playBtn"),
  resetBtn: document.getElementById("resetBtn"),
  recordBtn: document.getElementById("recordBtn"),
  autoZoom: document.getElementById("autoZoom"),
  fullPath: document.getElementById("fullPath"),
  chartsToggle: document.getElementById("chartsToggle"),
};

const indicators = {
  speed: document.getElementById("speedValue"),
  altitude: document.getElementById("altitudeValue"),
  distance: document.getElementById("distanceValue"),
  time: document.getElementById("timeValue"),
  grade: document.getElementById("gradeValue"),
  pace: document.getElementById("paceValue"),
  hr: document.getElementById("hrValue"),
  gaugeFill: document.getElementById("gaugeFill"),
  gaugeLabel: document.getElementById("gaugeLabel"),
};

const CHART_CONFIGS = {
  altitudeChart: { label: "Altitude (m)", color: "#5ca9ff" },
  speedChart: { label: "Vitesse (km/h)", color: "#f0b30a" },
  paceChart: { label: "Allure (min/km)", color: "#6ee7a2" },
  hrChart: { label: "Fréquence cardiaque (bpm)", color: "#ff7aa2" },
};

function init() {
  populateTileOptions();
  initMap();
  bindEvents();
  initCharts();
}

function populateTileOptions() {
  Object.keys(TILE_SERVERS).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    controls.tileSelect.appendChild(option);
  });
  controls.tileSelect.value = "OpenStreetMap";
}

function initMap() {
  state.map = L.map("map", {
    zoomControl: false,
    attributionControl: false,
  }).setView([0, 0], 2);
  updateBaseLayer();
}

function bindEvents() {
  controls.tileSelect.addEventListener("change", updateBaseLayer);
  controls.gpxFile.addEventListener("change", handleFile);
  controls.playBtn.addEventListener("click", togglePlay);
  controls.resetBtn.addEventListener("click", resetAnimation);
  controls.recordBtn.addEventListener("click", handleRecord);
  controls.chartsToggle.addEventListener("change", () => {
    document.querySelector(".charts").style.display =
      controls.chartsToggle.checked ? "grid" : "none";
  });
  controls.fullPath.addEventListener("change", () => {
    if (state.fullPath) {
      state.fullPath.setStyle({ opacity: controls.fullPath.checked ? 0.6 : 0 });
    }
  });
}

function initCharts() {
  Object.entries(CHART_CONFIGS).forEach(([id, cfg]) => {
    const ctx = document.getElementById(id);
    state.charts[id] = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: cfg.label,
            data: [],
            borderColor: cfg.color,
            backgroundColor: "transparent",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "#9aa0b1" } },
          y: { grid: { color: "rgba(255,255,255,0.08)" }, ticks: { color: "#9aa0b1" } },
        },
      },
    });
  });
}

function updateBaseLayer() {
  if (state.baseLayer) {
    state.map.removeLayer(state.baseLayer);
  }
  const selected = controls.tileSelect.value;
  const url = TILE_SERVERS[selected];
  if (!url) {
    state.baseLayer = L.layerGroup().addTo(state.map);
    return;
  }
  state.baseLayer = L.tileLayer(url, { maxZoom: 19 }).addTo(state.map);
}

async function handleFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const text = await file.text();
  state.data = parseGpx(text);
  if (!state.data.length) {
    alert("Aucune donnée GPX lisible");
    return;
  }
  updateMapLayers();
  updateChartsData();
  controls.playBtn.disabled = false;
  controls.resetBtn.disabled = false;
  controls.recordBtn.disabled = false;

}

function parseGpx(text) {
  const xml = new DOMParser().parseFromString(text, "text/xml");
  const points = Array.from(xml.getElementsByTagName("trkpt"));

  const routePoints = Array.from(xml.getElementsByTagName("rtept"));
  const segments = points.length ? points : routePoints;
  if (!segments.length) return [];

  const samples = [];
  let cumulativeDistance = 0;
  let startTime = null;


  segments.forEach((pt, idx) => {

    const lat = parseFloat(pt.getAttribute("lat"));
    const lon = parseFloat(pt.getAttribute("lon"));
    const eleNode = pt.getElementsByTagName("ele")[0];
    const ele = eleNode ? parseFloat(eleNode.textContent || "0") : 0;
    const timeNode = pt.getElementsByTagName("time")[0];
    const time = timeNode ? new Date(timeNode.textContent) : new Date();
    startTime = startTime || time;
    const elapsed = (time.getTime() - startTime.getTime()) / 1000;

    let hr = null;
    const hrNode =
      pt.getElementsByTagName("gpxtpx:hr")[0] ||
      pt.getElementsByTagName("hr")[0];
    if (hrNode) {
      hr = parseFloat(hrNode.textContent || "0");
    }

    if (idx > 0) {
      const prev = samples[idx - 1];
      const dist = haversineDistance(prev.lat, prev.lon, lat, lon);
      cumulativeDistance += dist;
    }

    samples.push({ lat, lon, ele, time, elapsed, distance: cumulativeDistance, hr });
  });

  // Enrichissement : vitesse, allure, pente
  for (let i = 1; i < samples.length; i++) {
    const prev = samples[i - 1];
    const cur = samples[i];
    const dist = cur.distance - prev.distance; // km
    const dt = cur.elapsed - prev.elapsed; // s
    const speed = dt > 0 ? (dist / dt) * 3600 : 0;
    const pace = speed > 0 ? 60 / speed : null;
    const grade = dist > 0 ? ((cur.ele - prev.ele) / (dist * 1000)) * 100 : 0;
    cur.speed = speed;
    cur.pace = pace;
    cur.grade = grade;
  }

  state.totalDuration = samples.length ? samples[samples.length - 1].elapsed : 0;
  return samples;
}

function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const toRad = (deg) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c; // km
}

function updateMapLayers() {
  const coords = state.data.map((p) => [p.lat, p.lon]);
  if (state.fullPath) state.map.removeLayer(state.fullPath);
  if (state.progressPath) state.map.removeLayer(state.progressPath);
  if (state.marker) state.map.removeLayer(state.marker);

  state.fullPath = L.polyline(coords, { color: "#c4981d", weight: 4, opacity: controls.fullPath.checked ? 0.6 : 0 }).addTo(state.map);
  state.progressPath = L.polyline([], { color: "#f0b30a", weight: 6, opacity: 0.9 }).addTo(state.map);
  state.marker = L.circleMarker(coords[0], { radius: 8, color: "#ff0040", weight: 3, fillOpacity: 1 }).addTo(state.map);

  fitMap();
}

function fitMap() {
  if (!controls.autoZoom.checked || !state.fullPath) return;
  const bounds = state.fullPath.getBounds();
  state.map.fitBounds(bounds.pad(0.05));
}

function smoothSeries(values, timestamps, windowSeconds) {
  if (!windowSeconds || windowSeconds <= 0) return values;
  const smoothed = [];
  let left = 0;
  for (let i = 0; i < values.length; i++) {
    const centerTime = timestamps[i];
    while (centerTime - timestamps[left] > windowSeconds) {
      left++;
    }
    let sum = 0;
    let count = 0;
    for (let j = left; j < values.length; j++) {
      if (timestamps[j] - centerTime > windowSeconds) break;
      if (!Number.isNaN(values[j])) {
        sum += values[j];
        count++;
      }
    }
    smoothed.push(count ? sum / count : values[i]);
  }
  return smoothed;
}

function updateChartsData() {
  const smoothingSeconds = Number(controls.smoothing.value);
  const distances = state.data.map((p) => p.distance);
  const timestamps = state.data.map((p) => p.elapsed);

  const altitude = smoothSeries(
    state.data.map((p) => p.ele),
    timestamps,
    smoothingSeconds
  );
  const speed = smoothSeries(
    state.data.map((p) => p.speed || 0),
    timestamps,
    smoothingSeconds
  );
  const pace = smoothSeries(
    state.data.map((p) => (p.pace ? p.pace : NaN)),
    timestamps,
    smoothingSeconds
  );
  const hr = smoothSeries(
    state.data.map((p) => (p.hr !== null ? p.hr : NaN)),
    timestamps,
    smoothingSeconds
  );

  updateChart("altitudeChart", distances, altitude);
  updateChart("speedChart", distances, speed);
  updateChart("paceChart", distances, pace);
  updateChart("hrChart", distances, hr);
}

function updateChart(id, labels, data) {
  const chart = state.charts[id];
  if (!chart) return;
  chart.data.labels = labels.map((d) => d.toFixed(2));
  chart.data.datasets[0].data = data;
  chart.update("none");
}

function togglePlay() {
  if (state.isPlaying) {
    stopAnimation();
    return;
  }
  startAnimation();
}

function startAnimation() {
  if (!state.data.length) return;
  state.isPlaying = true;
  state.startTime = performance.now();
  controls.playBtn.textContent = "Pause";
  animate();
}

function stopAnimation() {
  state.isPlaying = false;
  controls.playBtn.textContent = "Reprendre";
  if (state.animationId) cancelAnimationFrame(state.animationId);
}

function resetAnimation() {
  stopAnimation();
  state.startTime = 0;
  controls.playBtn.textContent = "Démarrer";
  if (state.progressPath) state.progressPath.setLatLngs([]);
  if (state.marker && state.data.length) state.marker.setLatLng([state.data[0].lat, state.data[0].lon]);
  updateIndicators(state.data[0], 0);
}

function animate() {
  const clipDuration = Number(controls.clipDuration.value);
  const now = performance.now();
  const elapsed = (now - state.startTime) / 1000;
  const progress = Math.min(1, elapsed / clipDuration);

  updatePlayback(progress);

  if (progress < 1 && state.isPlaying) {
    state.animationId = requestAnimationFrame(animate);
  } else {
    stopAnimation();
  }
}

function updatePlayback(progress) {
  const targetTime = state.totalDuration * progress;
  const data = state.data;
  let idx = data.findIndex((p) => p.elapsed >= targetTime);
  if (idx === -1) idx = data.length - 1;
  const prev = idx > 0 ? data[idx - 1] : data[0];
  const cur = data[idx];
  const segmentProgress = cur.elapsed === prev.elapsed ? 0 : (targetTime - prev.elapsed) / (cur.elapsed - prev.elapsed);

  const lat = interpolate(prev.lat, cur.lat, segmentProgress);
  const lon = interpolate(prev.lon, cur.lon, segmentProgress);
  const ele = interpolate(prev.ele, cur.ele, segmentProgress);
  const distance = interpolate(prev.distance, cur.distance, segmentProgress);
  const speed = interpolate(prev.speed || 0, cur.speed || 0, segmentProgress);
  const pace = speed > 0 ? 60 / speed : null;
  const hr = cur.hr !== null ? interpolate(prev.hr ?? cur.hr ?? 0, cur.hr ?? prev.hr ?? 0, segmentProgress) : null;
  const grade = interpolate(prev.grade || 0, cur.grade || 0, segmentProgress);

  if (state.marker) state.marker.setLatLng([lat, lon]);
  if (state.progressPath) {
    const existing = state.progressPath.getLatLngs();
    const newPoints = [...existing];
    if (!newPoints.length || newPoints[newPoints.length - 1].lat !== lat || newPoints[newPoints.length - 1].lng !== lon) {
      newPoints.push([lat, lon]);
      state.progressPath.setLatLngs(newPoints);
    }
  }

  if (controls.autoZoom.checked) {
    state.map.setView([lat, lon], Math.max(state.map.getZoom(), 13));
  }

  updateCompass(prev, cur);
  updateIndicators({ speed, ele, distance, time: new Date(state.data[0].time.getTime() + targetTime * 1000), grade, pace, hr }, progress);
}

function interpolate(a, b, t) {
  return a + (b - a) * t;
}

function updateCompass(prev, cur) {
  const bearing = computeBearing(prev.lat, prev.lon, cur.lat, cur.lon);
  state.compass.style.transform = `rotate(${bearing}deg)`;
}

function computeBearing(lat1, lon1, lat2, lon2) {
  const toRad = (deg) => (deg * Math.PI) / 180;
  const y = Math.sin(toRad(lon2 - lon1)) * Math.cos(toRad(lat2));
  const x =
    Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
    Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(toRad(lon2 - lon1));
  const brng = (Math.atan2(y, x) * 180) / Math.PI;
  return (brng + 360) % 360;
}

function updateIndicators(sample, progress) {
  if (!sample) return;
  indicators.speed.textContent = `${sample.speed?.toFixed(1) || 0} km/h`;
  indicators.altitude.textContent = `${Math.round(sample.ele || 0)} m`;
  indicators.distance.textContent = `${sample.distance?.toFixed(2) || 0} km`;
  indicators.time.textContent = formatTime(sample.time);
  indicators.grade.textContent = `${(sample.grade || 0).toFixed(1)} %`;
  indicators.pace.textContent = sample.pace ? `${formatPace(sample.pace)}` : "--";
  indicators.hr.textContent = sample.hr ? `${Math.round(sample.hr)} bpm` : "--";

  const gaugeProgress = Math.min(1, (sample.speed || 0) / Number(getComputedStyle(document.documentElement).getPropertyValue("--speed-max")));
  indicators.gaugeFill.style.width = `${(gaugeProgress * 100).toFixed(0)}%`;
  indicators.gaugeLabel.textContent = indicators.speed.textContent;
}

function formatTime(date) {
  if (!date) return "--:--:--";
  return date.toLocaleTimeString("fr-FR", { hour12: false });
}

function formatPace(paceMin) {
  const minutes = Math.floor(paceMin);
  const seconds = Math.round((paceMin - minutes) * 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}/km`;
}

function handleRecord() {
  if (state.recorder && state.recorder.state === "recording") {
    state.recorder.stop();
    return;
  }

  const stream = document.getElementById("renderArea").captureStream(25);
  state.captureStream = stream;
  state.recordingChunks = [];
  state.recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  state.recorder.ondataavailable = (e) => state.recordingChunks.push(e.data);
  state.recorder.onstop = saveRecording;
  state.recorder.start();

  if (!state.isPlaying) {
    startAnimation();
  }
  controls.recordBtn.textContent = "Arrêter l'enregistrement";
}

function saveRecording() {
  const blob = new Blob(state.recordingChunks, { type: "video/webm" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "overlaygpx.webm";
  link.click();
  URL.revokeObjectURL(url);
  controls.recordBtn.textContent = "Exporter en WebM";
}

init();
