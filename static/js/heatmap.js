/**
 * heatmap.js – Safety heatmap rendering with Leaflet.heat.
 *
 * Fetches area safety data from /api/heatmap_data and renders it as a
 * colour-coded heatmap overlay on the Leaflet map.
 *
 * Leaflet.heat expects points as [lat, lng, intensity] where intensity
 * is typically in [0, 1].  We pass safety scores directly.
 */

"use strict";

// ── Map setup ─────────────────────────────────────────────────────────────────
const heatMap = L.map("heatmapMap", {
  center: [55.9533, -3.1883],
  zoom:   13,
});

L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
  {
    attribution: '© OpenStreetMap © CARTO',
    subdomains: "abcd",
    maxZoom: 19,
  }
).addTo(heatMap);

// ── Heatmap layer state ───────────────────────────────────────────────────────
let heatLayer   = null;
let currentMode = "composite";

/**
 * Fetch heatmap data for the given mode and re-render the layer.
 *
 * Leaflet.heat colour scale (default):
 *   0.0 = blue  (coldest / least safe — mapped from low scores)
 *   0.5 = yellow
 *   1.0 = red   (hottest)
 *
 * Since higher score = SAFER, we invert: intensity = 1 - score
 * so that dangerous areas are "hot" (red) and safe areas are "cool" (blue/green).
 */
function loadHeatmap(mode) {
  currentMode = mode;

  fetch(`/api/heatmap_data?mode=${mode}`)
    .then(r => r.json())
    .then(data => {
      if (heatLayer) heatMap.removeLayer(heatLayer);

      // Convert to Leaflet.heat format: [lat, lng, intensity]
      // Invert score so high safety = cool (blue), low safety = hot (red)
      const points = data.map(d => [d.lat, d.lng, 1.0 - d.score]);

      const intensity = parseFloat(
        document.getElementById("intensitySlider").value
      );

      heatLayer = L.heatLayer(points, {
        radius:    40,
        blur:      30,
        maxZoom:   16,
        max:       1.0,
        minOpacity: 0.3,
        gradient: {
          0.0: "#22c55e",   // green  = safe
          0.4: "#eab308",   // yellow = moderate
          0.7: "#ef4444",   // red    = unsafe
          1.0: "#7f1d1d",   // dark red = very unsafe
        },
      }).addTo(heatMap);

      // Update the area list in the side panel
      updateAreaList(data);
    })
    .catch(err => console.error("[heatmap] Failed to load data:", err));
}

// ── Area list in sidebar ──────────────────────────────────────────────────────
function updateAreaList(data) {
  const container = document.getElementById("areaList");
  if (!container) return;

  // Sort by score descending (safest first)
  const sorted = [...data].sort((a, b) => b.score - a.score);

  container.innerHTML = `<p style="font-size:.8rem;font-weight:600;color:#6B7280;margin-bottom:8px">All areas (safest first)</p>`;

  sorted.forEach(area => {
    const scoreClass =
      area.score >= 0.65 ? "score-high" :
      area.score >= 0.45 ? "score-mid"  : "score-low";

    const div = document.createElement("div");
    div.className = "area-list-item";
    div.innerHTML = `
      <span>${area.name}</span>
      <span class="area-score-badge ${scoreClass}">
        ${(area.score * 10).toFixed(1)}/10
      </span>
    `;
    // Click to pan map to area
    div.style.cursor = "pointer";
    div.addEventListener("click", () => {
      heatMap.setView([area.lat, area.lng], 15);
    });
    container.appendChild(div);
  });
}

// ── Mode toggle ───────────────────────────────────────────────────────────────
document.querySelectorAll("[data-mode]").forEach(btn => {
  btn.addEventListener("click", function () {
    document.querySelectorAll("[data-mode]").forEach(b => b.classList.remove("active"));
    this.classList.add("active");
    loadHeatmap(this.dataset.mode);
  });
});

// ── Intensity slider ──────────────────────────────────────────────────────────
document.getElementById("intensitySlider").addEventListener("input", function () {
  // Re-render with new intensity
  loadHeatmap(currentMode);
});

// ── Initial load ──────────────────────────────────────────────────────────────
loadHeatmap("composite");
