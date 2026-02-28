/**
 * routing.js – Route request, display, and UI interactions.
 *
 * Responsibilities:
 *  - Geocode start/end addresses via Nominatim
 *  - Send route request to /api/route
 *  - Draw colour-coded route segments on the Leaflet map
 *  - Update the route stats panel
 *  - Handle slider and algorithm toggle interactions
 */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let currentRouteLayer = null;  // Leaflet layer group for the drawn route
let startMarker  = null;
let endMarker    = null;
let selectedAlgo = "astar";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const startInput    = document.getElementById("startInput");
const endInput      = document.getElementById("endInput");
const alphaSlider   = document.getElementById("alphaSlider");
const alphaLabel    = document.getElementById("alphaLabel");
const findRouteBtn  = document.getElementById("findRouteBtn");
const routeResult   = document.getElementById("routeResult");
const routeError    = document.getElementById("routeError");
const useMyLocation = document.getElementById("useMyLocation");

// ── Slider label ──────────────────────────────────────────────────────────────
alphaSlider.addEventListener("input", function () {
  const pct = Math.round(parseFloat(this.value) * 100);
  if (pct >= 50) {
    alphaLabel.textContent = `Safe (${pct}%)`;
  } else {
    alphaLabel.textContent = `Fast (${100 - pct}%)`;
  }
});

// ── Algorithm toggle ──────────────────────────────────────────────────────────
document.querySelectorAll("[data-algo]").forEach(btn => {
  btn.addEventListener("click", function () {
    document.querySelectorAll("[data-algo]").forEach(b => b.classList.remove("active"));
    this.classList.add("active");
    selectedAlgo = this.dataset.algo;
  });
});

// ── Geolocation ───────────────────────────────────────────────────────────────
useMyLocation.addEventListener("click", function () {
  if (!navigator.geolocation) return alert("Geolocation not supported by your browser.");

  this.disabled = true;
  navigator.geolocation.getCurrentPosition(
    pos => {
      const { latitude: lat, longitude: lng } = pos.coords;
      startInput.value = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
      startInput.dataset.lat = lat;
      startInput.dataset.lng = lng;
      placeStartMarker(lat, lng);
      this.disabled = false;
    },
    () => {
      alert("Could not get your location.");
      this.disabled = false;
    }
  );
});

// ── Geocoding via Nominatim ───────────────────────────────────────────────────

/**
 * Geocode a place name using OpenStreetMap's Nominatim service.
 * Returns { lat, lng } or null.
 */
async function geocode(query) {
  // If the query looks like "lat, lng" coordinates, parse directly.
  const coordMatch = query.match(/^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$/);
  if (coordMatch) {
    return { lat: parseFloat(coordMatch[1]), lng: parseFloat(coordMatch[2]) };
  }

  // Bias the search towards Edinburgh
  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query + ", Edinburgh, UK")}&format=json&limit=1&bounded=1&viewbox=-3.25,55.92,-3.10,55.98`;
  const res  = await fetch(url, { headers: { "Accept-Language": "en" } });
  const data = await res.json();

  if (data.length === 0) return null;
  return { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon) };
}

// ── Marker helpers ────────────────────────────────────────────────────────────

function makePin(colour, letter) {
  return L.divIcon({
    className: "",
    html: `
      <div style="
        width:28px; height:28px; border-radius:50% 50% 50% 0;
        transform:rotate(-45deg);
        background:${colour}; border:3px solid white;
        box-shadow:0 3px 8px rgba(0,0,0,.3);
        display:flex; align-items:center; justify-content:center;
      ">
        <span style="transform:rotate(45deg); font-size:11px; font-weight:700; color:white">${letter}</span>
      </div>
    `,
    iconSize:   [28, 28],
    iconAnchor: [14, 28],
  });
}

function placeStartMarker(lat, lng) {
  if (startMarker) map.removeLayer(startMarker);
  startMarker = L.marker([lat, lng], { icon: makePin("#7C3AED", "A") }).addTo(map);
}

function placeEndMarker(lat, lng) {
  if (endMarker) map.removeLayer(endMarker);
  endMarker = L.marker([lat, lng], { icon: makePin("#F472B6", "B") }).addTo(map);
}

// ── Route display ─────────────────────────────────────────────────────────────

/**
 * Draw colour-coded route segments on the map.
 * Segments are provided by the backend as:
 *   [{ coords: [[lat,lng],[lat,lng]], color: "#hex", safety: float }, ...]
 */
function drawRoute(segments, coordinates) {
  if (currentRouteLayer) map.removeLayer(currentRouteLayer);

  const group = L.layerGroup();

  if (segments && segments.length > 0) {
    // Colour-coded polylines per segment
    segments.forEach(seg => {
      L.polyline(seg.coords, {
        color:   seg.color,
        weight:  5,
        opacity: 0.85,
        lineCap: "round",
      }).addTo(group);
    });
  } else if (coordinates && coordinates.length > 0) {
    // Fallback: single grey polyline if no segments provided
    L.polyline(coordinates, { color: "#7C3AED", weight: 5 }).addTo(group);
  }

  group.addTo(map);
  currentRouteLayer = group;

  // Fit map to the route bounds
  if (coordinates && coordinates.length > 0) {
    map.fitBounds(coordinates, { padding: [40, 40] });
  }
}

/**
 * Update the stats panel with route data.
 */
function updateRoutePanel(result) {
  const walkMinutes = Math.round((result.distance_km / 5.0) * 60); // 5 km/h average walking
  const safetyPct   = result.safety_score * 100;
  const safetyOut10 = (result.safety_score * 10).toFixed(1);

  document.getElementById("routeDistance").textContent = result.distance_km.toFixed(2);
  document.getElementById("routeTime").textContent     = walkMinutes;
  document.getElementById("routeSafety").textContent   = safetyOut10;

  // Safety bar colour
  const bar = document.getElementById("safetyBar");
  bar.style.width = safetyPct + "%";
  if (result.safety_score >= 0.65) {
    bar.style.background = "#22C55E";
  } else if (result.safety_score >= 0.45) {
    bar.style.background = "#EAB308";
  } else {
    bar.style.background = "#EF4444";
  }

  routeResult.classList.remove("hidden");
  routeError.classList.add("hidden");
}

// ── Find Route button ─────────────────────────────────────────────────────────

findRouteBtn.addEventListener("click", async function () {
  routeResult.classList.add("hidden");
  routeError.classList.add("hidden");

  const startQuery = startInput.value.trim();
  const endQuery   = endInput.value.trim();

  if (!startQuery || !endQuery) {
    showError("Please enter both a start and destination.");
    return;
  }

  findRouteBtn.disabled    = true;
  findRouteBtn.textContent = "Finding route…";

  try {
    // Geocode both locations in parallel
    const [startCoords, endCoords] = await Promise.all([
      geocode(startQuery),
      geocode(endQuery),
    ]);

    if (!startCoords) { showError(`Could not find "${startQuery}" in Edinburgh.`); return; }
    if (!endCoords)   { showError(`Could not find "${endQuery}" in Edinburgh.`); return; }

    // Place markers
    placeStartMarker(startCoords.lat, startCoords.lng);
    placeEndMarker(endCoords.lat,   endCoords.lng);

    // Request route from the backend
    const payload = {
      start_lat:  startCoords.lat,
      start_lng:  startCoords.lng,
      end_lat:    endCoords.lat,
      end_lng:    endCoords.lng,
      alpha:      parseFloat(alphaSlider.value),
      algorithm:  selectedAlgo,
    };

    const res  = await fetch("/api/route", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "Route request failed.");
      return;
    }

    drawRoute(data.segments, data.coordinates);
    updateRoutePanel(data);

  } catch (err) {
    showError("An unexpected error occurred: " + err.message);
  } finally {
    findRouteBtn.disabled    = false;
    findRouteBtn.textContent = "Find Safe Route";
  }
});

function showError(msg) {
  routeError.textContent = msg;
  routeError.classList.remove("hidden");
  routeResult.classList.add("hidden");
}

// ── Enter key triggers route ──────────────────────────────────────────────────
[startInput, endInput].forEach(input => {
  input.addEventListener("keydown", e => {
    if (e.key === "Enter") findRouteBtn.click();
  });
});
