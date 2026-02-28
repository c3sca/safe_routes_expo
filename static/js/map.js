/**
 * map.js – Leaflet map initialisation and area-info click handler.
 *
 * Responsibilities:
 *  - Initialise the Leaflet map centred on Edinburgh
 *  - Show area safety markers on the map
 *  - Handle map clicks → fetch /api/area_info → show area popup
 *  - Export the map instance so routing.js can use it
 */

"use strict";

// ── Map initialisation ────────────────────────────────────────────────────────

const map = L.map("map", {
  center: [55.9533, -3.1883],  // Edinburgh city centre
  zoom:   14,
  zoomControl: true,
});

// ─── Drop Pin ──────────────────────────────────────────────────────────────────────
let clickMarker = null;

const purpleIcon = L.divIcon({
  className: "",
  html: `<div style="
    width:14px;height:14px;
    background:#7C3AED;
    border-radius:50%;
    border:2px solid white;
    box-shadow:0 2px 6px rgba(0,0,0,.3);
  "></div>`,
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

// CartoDB Voyager – clean, modern tile layer
L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
  {
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> ' +
                 '© <a href="https://carto.com/">CARTO</a>',
    subdomains:  "abcd",
    maxZoom:     19,
  }
).addTo(map);


// ── Area safety markers ───────────────────────────────────────────────────────

/**
 * Convert a safety score (0–1) to a colour string.
 * 0 = red, 0.5 = yellow, 1 = green.
 */
function scoreToColour(score) {
  if (score >= 0.65) return "#22C55E";
  if (score >= 0.45) return "#EAB308";
  return "#EF4444";
}

/**
 * Create a coloured circle div-icon for an area marker.
 */
function areaIcon(score) {
  const colour = scoreToColour(score);
  return L.divIcon({
    className: "",
    html: `<div style="
      width:14px; height:14px; border-radius:50%;
      background:${colour}; border:2px solid white;
      box-shadow:0 2px 6px rgba(0,0,0,.25);
    "></div>`,
    iconSize:   [14, 14],
    iconAnchor: [7, 7],
  });
}

// Fetch all areas and place markers
fetch("/api/areas")
  .then(r => r.json())
  .then(areas => {
    areas.forEach(area => {
      const marker = L.marker([area.lat, area.lng], { icon: areaIcon(area.composite_score) });
      marker.addTo(map).bindTooltip(
        `<strong>${area.name}</strong><br>Safety: ${(area.composite_score * 10).toFixed(1)}/10`,
        { direction: "top", offset: [0, -8] }
      );
    });
  })
  .catch(console.error);


// ── Map click → area info popup ───────────────────────────────────────────────

const areaPopup      = document.getElementById("areaPopup");
const closeAreaPopup = document.getElementById("closeAreaPopup");

closeAreaPopup.addEventListener("click", () => {
  areaPopup.classList.add("hidden");
});

map.on("click", function (e) {
  const { lat, lng } = e.latlng;

  // Remove previous marker
  if (clickMarker) map.removeLayer(clickMarker);

  // Add new marker
  clickMarker = L.marker([lat, lng], { icon: purpleIcon }).addTo(map);
  
// map.on("click", function (e) {
//  const { lat, lng } = e.latlng;

  fetch(`/api/area_info?lat=${lat}&lng=${lng}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) return;

      document.getElementById("popupAreaName").textContent  = data.name;
      document.getElementById("popupComposite").textContent =
        (data.composite_score * 10).toFixed(1) + "/10";
      document.getElementById("popupUser").textContent  =
        (data.avg_user_score * 10).toFixed(1) + "/10";
      document.getElementById("popupLighting").textContent =
        (data.lighting * 100).toFixed(0) + "%";

      // Recent ratings list
      const ratingsDiv = document.getElementById("recentRatings");
      if (data.recent_ratings && data.recent_ratings.length > 0) {
        const stars = r => "★".repeat(r.score) + "☆".repeat(5 - r.score);
        ratingsDiv.innerHTML = `
          <p style="font-size:.75rem;color:#9CA3AF;margin:12px 0 6px">Recent ratings:</p>
          ${data.recent_ratings.map(r =>
            `<div style="font-size:.8rem;color:#4B5563;margin-bottom:4px">
              <span style="color:#FBBF24">${stars(r)}</span>
              <span style="margin-left:4px;color:#C4B5FD">${r.time_of_day}</span>
              ${r.comment ? `<br><em style="color:#6B7280">${r.comment}</em>` : ""}
            </div>`
          ).join("")}
        `;
      } else {
        ratingsDiv.innerHTML = '<p style="font-size:.75rem;color:#9CA3AF;margin:8px 0">No ratings yet.</p>';
      }

      areaPopup.classList.remove("hidden");
    })
    .catch(console.error);
});
