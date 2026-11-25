// ============================================================
// RAJAPETA WATERSHED INTERACTIVE MAP
// NDVI / LST / LULC
// ============================================================


// ============================================================
// 1. STUDY AREA
// ============================================================

const rajapetaBoundary = [

    [16.368712, 78.021089],
    [16.366574, 78.022436],
    [16.366611, 78.024050],
    [16.365875, 78.025284],
    [16.363715, 78.025862],
    [16.362217, 78.025553],
    [16.361880, 78.026070],
    [16.362013, 78.026756],
    [16.361708, 78.026891],
    [16.361157, 78.026350],
    [16.360873, 78.026565],
    [16.360639, 78.025628],
    [16.357653, 78.025478],
    [16.357024, 78.025846],
    [16.355900, 78.025795],
    [16.347262, 78.032489],
    [16.340451, 78.036701],
    [16.337309, 78.032642],
    [16.338165, 78.016312],
    [16.338535, 77.997197],
    [16.339891, 77.995594],
    [16.342004, 77.994258],
    [16.346133, 77.990636],
    [16.348536, 77.990199],
    [16.353571, 77.990584],
    [16.355095, 77.991089],
    [16.355450, 77.990749],
    [16.355239, 77.990666],
    [16.356204, 77.990699],
    [16.356398, 77.992443],
    [16.357479, 77.992633],
    [16.358917, 77.992318],
    [16.359462, 77.992898],
    [16.360394, 77.993185],
    [16.362935, 77.992312],
    [16.363016, 77.992945],
    [16.363488, 77.993092],
    [16.365463, 77.992728],
    [16.365383, 77.993035],
    [16.366442, 77.992404],
    [16.366124, 77.995181],
    [16.365466, 77.997355],
    [16.365427, 77.999559],
    [16.364910, 78.002218],
    [16.364836, 78.011998],
    [16.368712, 78.021089]

];


// ============================================================
// 2. DETERMINE PAGE TYPE
// ============================================================

const analysisType =
    document.body.dataset.analysis;


// ============================================================
// 3. CREATE MAP
// ============================================================

const map = L.map("map", {

    zoomControl: true,

    attributionControl: true

}).setView(

    [16.353, 78.015],

    13

);


// ============================================================
// 4. BASEMAP
// ============================================================

// Esri Street Map
// Used instead of OpenStreetMap because OSM gave 403.

const streetMap = L.tileLayer(

    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",

    {

        maxZoom: 19,

        attribution:
            "Tiles © Esri"

    }

).addTo(map);


// Optional satellite layer

const satellite = L.tileLayer(

    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",

    {

        maxZoom: 19,

        attribution:
            "Tiles © Esri"

    }

);


// ============================================================
// 5. RAJAPETA BOUNDARY
// ============================================================

const boundaryLayer = L.polygon(

    rajapetaBoundary,

    {

        color: "#ff5722",

        weight: 2.5,

        fillColor: "#4caf50",

        fillOpacity: 0.08

    }

).addTo(map);


// Popup

boundaryLayer.bindPopup(`

    <b>Rajapeta Village</b>

    <br><br>

    Mandal: Wanaparthy

    <br>

    State: Telangana, India

    <br>

    Study: Watershed Development

`);


// ============================================================
// 6. FIT MAP
// ============================================================

map.fitBounds(

    boundaryLayer.getBounds(),

    {

        padding: [30, 30]

    }

);


// ============================================================
// 7. FILE CONFIGURATION
// ============================================================

const rasterFiles = {

    ndvi: {

        "2004":
            "../Rajapet/GeoTiff/Rajapeta_NDVI_2004.tif",

        "2024":
            "../Rajapet/GeoTiff/Rajapeta_NDVI_2024.tif",

        "change":
            "../Rajapet/GeoTiff/Rajapeta_NDVI_Change_2004_2024.tif"

    },


    lst: {

        "2004":
            "../Rajapet/GeoTiff/Rajapeta_LST_2004.tif",

        "2024":
            "../Rajapet/GeoTiff/Rajapeta_LST_2024.tif",

        "change":
            "../Rajapet/GeoTiff/Rajapeta_LST_Change_2004_2024.tif"

    },


    lulc: {

        "2004":
            "../Rajapet/GeoTiff/Rajapeta_LULC_2004.tif",

        "2024":
            "../Rajapet/GeoTiff/Rajapeta_LULC_2024.tif",

        "change":
            "../Rajapet/GeoTiff/Rajapeta_LULC_Change_2004_2024.tif"

    }

};


// ============================================================
// 8. RASTER LAYER
// ============================================================

let rasterLayer = null;


// ============================================================
// 9. COLOR FUNCTION
// ============================================================

function getColorFunction(type, year) {


    // --------------------------------------------------------
    // NDVI
    // --------------------------------------------------------

    if (type === "ndvi") {

        if (year === "change") {

            return function(value) {

                if (value === null ||
                    value === undefined) {

                    return null;

                }

                if (value < -0.10)
                    return "#d73027";

                if (value < -0.03)
                    return "#fc8d59";

                if (value < 0.03)
                    return "#ffffbf";

                if (value < 0.10)
                    return "#91cf60";

                return "#1a9850";

            };

        }


        return function(value) {

            if (value === null ||
                value === undefined) {

                return null;

            }

            if (value < 0.05)
                return "#8c510a";

            if (value < 0.10)
                return "#d8b365";

            if (value < 0.15)
                return "#f6e8c3";

            if (value < 0.20)
                return "#c7eae5";

            if (value < 0.30)
                return "#5ab4ac";

            return "#01665e";

        };

    }


    // --------------------------------------------------------
    // LST
    // --------------------------------------------------------

    if (type === "lst") {

        if (year === "change") {

            return function(value) {

                if (value === null ||
                    value === undefined) {

                    return null;

                }

                if (value < -5)
                    return "#2166ac";

                if (value < -2)
                    return "#67a9cf";

                if (value < 2)
                    return "#f7f7f7";

                if (value < 5)
                    return "#ef8a62";

                return "#b2182b";

            };

        }


        return function(value) {

            if (value === null ||
                value === undefined) {

                return null;

            }

            if (value < 32)
                return "#313695";

            if (value < 35)
                return "#74add1";

            if (value < 38)
                return "#ffffbf";

            if (value < 41)
                return "#f46d43";

            return "#a50026";

        };

    }


    // --------------------------------------------------------
    // LULC
    // --------------------------------------------------------

    if (type === "lulc") {

        return function(value) {

            if (value === null ||
                value === undefined) {

                return null;

            }


            const colors = [

                "#006400",

                "#7fcdbb",

                "#ffff33",

                "#d95f02",

                "#999999"

            ];


            const classNumber =
                Math.round(value);


            return colors[
                classNumber
            ] || "#ffffff";

        };

    }

}


// ============================================================
// 10. LOAD GEOTIFF
// ============================================================

async function loadRaster(year) {

    // Remove previous raster

    if (rasterLayer !== null) {

        map.removeLayer(
            rasterLayer
        );

        rasterLayer = null;

    }


    const filepath =
        rasterFiles[
            analysisType
        ][year];


    console.log(
        "Loading:",
        filepath
    );


    try {

        const response =
            await fetch(filepath);


        if (!response.ok) {

            throw new Error(
                "GeoTIFF not found: " +
                filepath
            );

        }


        const arrayBuffer =
            await response.arrayBuffer();


        const georaster =
            await parseGeoraster(
                arrayBuffer
            );


        console.log(
            "GeoTIFF loaded:",
            georaster
        );


        const colorFunction =
            getColorFunction(
                analysisType,
                year
            );


        rasterLayer =
            new GeoRasterLayer({

                georaster:

                    georaster,

                opacity:

                    0.75,

                resolution:

                    256,

                pixelValuesToColorFn:

                    colorFunction

            });


        rasterLayer.addTo(
            map
        );


        // Keep boundary visible

        boundaryLayer.bringToFront();


        updateLegend(
            analysisType,
            year
        );


    }

    catch(error) {

        console.error(
            error
        );


        alert(
            "GeoTIFF could not be loaded.\n\n" +
            filepath +
            "\n\n" +
            "Check whether the file exists."
        );

    }

}


// ============================================================
// 11. FILTER BUTTONS
// ============================================================

const buttons =
    document.querySelectorAll(
        ".analysis-btn"
    );


buttons.forEach(
    function(button) {

        button.addEventListener(
            "click",
            function() {


                // Remove active

                buttons.forEach(
                    function(btn) {

                        btn.classList.remove(
                            "active"
                        );

                    }
                );


                // Add active

                button.classList.add(
                    "active"
                );


                // Selected year

                const year =
                    button.dataset.year;


                // Load map

                loadRaster(
                    year
                );

            }
        );

    }
);


// ============================================================
// 12. LAYER CONTROL
// ============================================================

const baseMaps = {

    "Street Map":
        streetMap,

    "Satellite":
        satellite

};


const overlays = {

    "Rajapeta Boundary":
        boundaryLayer

};


L.control.layers(

    baseMaps,

    overlays,

    {

        collapsed: false

    }

).addTo(map);


// ============================================================
// 13. SCALE
// ============================================================

L.control.scale({

    imperial: false

}).addTo(map);


// ============================================================
// 14. LEGEND
// ============================================================

function updateLegend(
    type,
    year
) {


    const legend =
        document.getElementById(
            "legend"
        );


    // --------------------------------------------------------
    // NDVI
    // --------------------------------------------------------

    if (type === "ndvi") {


        if (year === "change") {

            legend.innerHTML = `

                <div class="legend-title">
                    NDVI Change
                </div>

                <div class="gradient gradient-ndvi-change">
                </div>

                <div class="legend-labels">
                    <span>Loss</span>
                    <span>0</span>
                    <span>Gain</span>
                </div>

            `;

        }

        else {

            legend.innerHTML = `

                <div class="legend-title">
                    NDVI
                </div>

                <div class="gradient gradient-ndvi">
                </div>

                <div class="legend-labels">
                    <span>Low</span>
                    <span>High</span>
                </div>

                <div style="margin-top:6px">
                    Year: ${year}
                </div>

            `;

        }

    }


    // --------------------------------------------------------
    // LST
    // --------------------------------------------------------

    if (type === "lst") {


        if (year === "change") {

            legend.innerHTML = `

                <div class="legend-title">
                    LST Change (°C)
                </div>

                <div class="gradient gradient-lst-change">
                </div>

                <div class="legend-labels">
                    <span>Decrease</span>
                    <span>0</span>
                    <span>Increase</span>
                </div>

            `;

        }

        else {

            legend.innerHTML = `

                <div class="legend-title">
                    LST (°C)
                </div>

                <div class="gradient gradient-lst">
                </div>

                <div class="legend-labels">
                    <span>Low</span>
                    <span>High</span>
                </div>

                <div style="margin-top:6px">
                    Year: ${year}
                </div>

            `;

        }

    }


    // --------------------------------------------------------
    // LULC
    // --------------------------------------------------------

    if (type === "lulc") {

        legend.innerHTML = `

            <div class="legend-title">
                LULC Classes
            </div>

            <div class="lulc-item">
                <span
                    class="lulc-color"
                    style="background:#006400">
                </span>
                Vegetation
            </div>

            <div class="lulc-item">
                <span
                    class="lulc-color"
                    style="background:#7fcdbb">
                </span>
                Water
            </div>

            <div class="lulc-item">
                <span
                    class="lulc-color"
                    style="background:#ffff33">
                </span>
                Cropland
            </div>

            <div class="lulc-item">
                <span
                    class="lulc-color"
                    style="background:#d95f02">
                </span>
                Built-up
            </div>

            <div class="lulc-item">
                <span
                    class="lulc-color"
                    style="background:#999999">
                </span>
                Barren
            </div>

        `;

    }

}


// ============================================================
// 15. INITIAL LOAD
// ============================================================

loadRaster("2004");