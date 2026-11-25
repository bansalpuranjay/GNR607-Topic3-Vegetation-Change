import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from PIL import Image
from rasterio.warp import transform
from rasterio.windows import from_bounds

YEARS = range(2014, 2027)

BOX = (274845, 2113125, 286545, 2122125)
LONLAT = [72.86, 19.10, 72.97, 19.18]
IITB = (72.9133, 19.1334)
AIRPORT = (291495, 2097465, 300045, 2104215)
PATH, ROW = "148", "047"
TILE = "N18E072"
UTM = "EPSG:32643"
SITES = {"Aarey Metro depot": 629105307, "Kanjur landfill": 257455760}
IST = timedelta(hours=5, minutes=30)
SUNRISE, SUNSET = 7.25, 18.33

BANDS = ["blue", "green", "red", "nir08", "swir16", "qa_pixel"]
NAMES = ["blue", "green", "red", "nir", "swir1", "qa"]

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
WORLDCOVER = {
    2020: f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{TILE}_Map.tif",
    2021: f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{TILE}_Map.tif",
}
RAIN = (f"https://archive-api.open-meteo.com/v1/archive?latitude={IITB[1]}&longitude={IITB[0]}"
        f"&start_date={YEARS[0] - 1}-06-01&end_date={YEARS[-1]}-01-31&daily=precipitation_sum&timezone=Asia%2FKolkata")
OVERPASS = "https://overpass-api.de/api/interpreter?data="
AGENT = {"User-Agent": "GNR607 course project (IIT Bombay)"}
WAYBACK = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
TILES = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/"
         "MapServer/tile/{release}/{z}/{y}/{x}")
POINT_RELEASES = {2014: ["2015-06-24", "2017-06-14"],
                  2026: ["2026-11-18"]}
SITE_RELEASES = ["2015-06-24", "2017-06-14", "2017-11-16", "2020-06-10", "2021-01-13", "2023-01-11", "2024-01-18",
                 "2025-01-30", "2026-11-18"]
POINT_SIZE, POINT_PIXELS = 150, 250
SITE_PIXELS = 600
CLASSES = {10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"),
           40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare / sparse", "#b4b4b4"),
           80: ("Water", "#0064c8"), 90: ("Wetland", "#0096a0"), 95: ("Mangroves", "#00cf75")}

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"


def get_json(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def find_scenes(year):
    body = {"collections": ["landsat-c2-l2"], "bbox": LONLAT,
            "datetime": f"{year - 1}-12-11/{year}-02-21", "limit": 100}
    scenes = []
    for item in get_json(STAC, body)["features"]:
        p = item["properties"]
        if (p["platform"] == "landsat-8" and p["landsat:wrs_path"] == PATH
                and p["landsat:wrs_row"] == ROW and p["landsat:collection_category"] == "T1"):
            scenes.append(item)
    return scenes


def read_box(url, token, box=BOX):
    with rasterio.open(url + "?" + token) as src:
        win = from_bounds(*box, transform=src.transform)
        return src.read(1, window=win), src.window_transform(win), src.crs


def cloud_percent(qa):
    bad = (qa & 0b11111) != 0
    return 100 * bad.mean()


def haze_percent(aerosol):
    level = (aerosol >> 6) & 3
    return 100 * (level == 3).mean()


def pick_best(scenes):
    clear = [s for s in scenes if s["cloud"] <= 1]
    if not clear:
        return min(scenes, key=lambda s: s["cloud"])
    january = [s for s in clear if s["date"].month == 1]
    if january:
        clear = january
    return min(clear, key=lambda s: s["haze"])


def save_year(year, item, token, box=BOX, folder=DATA):
    bands = []
    for key in BANDS:
        band, tf, crs = read_box(item["assets"][key]["href"], token, box)
        bands.append(band)
    stack = np.stack(bands)
    with rasterio.open(folder / f"{year}.tif", "w", driver="GTiff", width=stack.shape[2],
                       height=stack.shape[1], count=len(bands), dtype="uint16", crs=crs,
                       transform=tf, compress="deflate") as dst:
        dst.write(stack)
        for i, name in enumerate(NAMES, start=1):
            dst.set_band_description(i, name)
        dst.update_tags(scene=item["id"], date=item["properties"]["datetime"][:10])
    return stack


def get_worldcover():
    for year, url in WORLDCOVER.items():
        with rasterio.open(url) as src:
            win = from_bounds(round(LONLAT[0] - 0.01, 4), round(LONLAT[1] - 0.01, 4), round(LONLAT[2] + 0.01, 4),
                              round(LONLAT[3] + 0.01, 4), transform=src.transform)
            classes = src.read(1, window=win)
            with rasterio.open(DATA / f"worldcover_{year}.tif", "w", driver="GTiff",
                               width=classes.shape[1], height=classes.shape[0], count=1,
                               dtype="uint8", crs=src.crs, transform=src.window_transform(win),
                               compress="deflate") as dst:
                dst.write(classes, 1)


def get_rainfall():
    days = get_json(RAIN)["daily"]
    with open(DATA / "rainfall.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "rain_mm"])
        writer.writerows(zip(days["time"], days["precipitation_sum"]))


def get_outlines():
    rows = []
    for name, way in SITES.items():
        query = urllib.parse.quote(f"[out:json];way({way});out geom;")
        req = urllib.request.Request(OVERPASS + query, headers={"User-Agent": "GNR607 course project (IIT Bombay)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            for point in json.load(r)["elements"][0]["geometry"]:
                rows.append([name, point["lat"], point["lon"]])
    with open(DATA / "sites.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site", "lat", "lon"])
        writer.writerows(rows)


def get_second_area():
    folder = DATA / "airport"
    folder.mkdir(exist_ok=True)
    chosen = {int(r["year"]): r["date"] for r in csv.DictReader(open(DATA / "scene_log.csv"))}
    token = get_json(TOKEN)["token"]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for year in YEARS:
            item = [s for s in find_scenes(year) if s["properties"]["datetime"][:10] == chosen[year]][0]
            stack = save_year(year, item, token, AIRPORT, folder)
            print(year, chosen[year], f"cloud in the airport box {cloud_percent(stack[5]):.2f} %")


def world_pixel(lon, lat, z):
    size = 256 * 2 ** z
    x = (lon + 180) / 360 * size
    y = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * size
    return x, y


def world_lonlat(x, y, z):
    size = 256 * 2 ** z
    return x / size * 360 - 180, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / size))))


def fetch(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=60) as r:
                return r.read()
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def photo_date(metadata_url, lon, lat):
    for layer, zoom in [(5, 18), (6, 17), (7, 16)]:
        query = urllib.parse.urlencode({"f": "json", "where": "1=1", "returnGeometry": "false",
                                        "outFields": "SRC_DATE2,SRC_DESC,SRC_RES,NICE_DESC",
                                        "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint",
                                        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects"})
        found = json.loads(fetch(f"{metadata_url}/{layer}/query?{query}")).get("features", [])
        if found:
            a = found[0]["attributes"]
            taken = datetime.fromtimestamp(a["SRC_DATE2"] / 1000, tz=timezone.utc).date()
            return {"capture_date": str(taken), "satellite": a["SRC_DESC"], "provider": a["NICE_DESC"],
                    "resolution_m": a["SRC_RES"], "max_zoom": zoom}
    return None


def get_tile(release, z, x, y, cache):
    key = (release, z, x, y)
    if key not in cache:
        cache[key] = Image.open(io.BytesIO(fetch(TILES.format(release=release, z=z, x=x, y=y)))).convert("RGB")
    return cache[key]


def tile_zoom(lat, metres_per_pixel, max_zoom):
    return min(max_zoom, int(np.ceil(np.log2(156543.03392 * np.cos(np.radians(lat)) / metres_per_pixel))))


def save_photo(path, release, centre, size_m, pixels, zoom, cache):
    step = size_m / pixels
    left, top = centre[0] - size_m / 2, centre[1] + size_m / 2
    east = left + (np.arange(pixels) + 0.5) * step
    north = top - (np.arange(pixels) + 0.5) * step
    xs, ys = np.meshgrid(east, north)
    lon, lat = transform(UTM, "EPSG:4326", xs.ravel(), ys.ravel())
    px, py = world_pixel(np.array(lon), np.array(lat), zoom)
    x0, y0 = int(px.min()) // 256, int(py.min()) // 256
    x1, y1 = int(px.max()) // 256, int(py.max()) // 256
    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3), dtype=np.uint8)
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tile = np.asarray(get_tile(release, zoom, tx, ty, cache))
            mosaic[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = tile
    rows = (py - y0 * 256).astype(int)
    cols = (px - x0 * 256).astype(int)
    Image.fromarray(mosaic[rows, cols].reshape(pixels, pixels, 3)).save(path, quality=92)
    corner_lon, corner_lat = transform(UTM, "EPSG:4326", [left, left + size_m], [top, top - size_m])
    return left, top, step, corner_lon[0], corner_lat[1], corner_lon[1], corner_lat[0]


def get_photos():
    releases = {}
    for value in get_json(WAYBACK).values():
        day = value["itemTitle"].split("Wayback ")[1].strip(")")
        releases[day] = (value["itemURL"].split("/tile/")[1].split("/")[0], value["metadataLayerUrl"])
    folder = DATA / "photos"
    index = folder / "photos.csv"
    header = ["kind", "year", "name", "file", "release", "capture_date", "satellite", "provider", "resolution_m",
              "zoom", "pixels", "metres_per_pixel", "left", "top", "west", "south", "east", "north"]
    before = {}
    if index.exists():
        for row in csv.DictReader(open(index, encoding="utf-8")):
            if set(header) <= set(row) and (folder / row["file"]).exists():
                before[(row["kind"], row["year"], row["name"], row["release"])] = [row[h] for h in header]
    cache, rows, failed = {}, [], []
    pool = ThreadPoolExecutor(8)

    def add(kind, year, name, file, release_day, lon, lat, centre, size_m, pixels, info=None):
        old = before.get((kind, str(year), name, release_day))
        if old:
            rows.append(old)
            return old[5]
        try:
            release, metadata = releases[release_day]
            info = info or photo_date(metadata, lon, lat)
            if info is None:
                return None
            zoom = tile_zoom(lat, size_m / pixels, info["max_zoom"])
            path = folder / file.format(date=info["capture_date"])
            path.parent.mkdir(parents=True, exist_ok=True)
            left, top, step, west, south, east, north = save_photo(path, release, centre, size_m, pixels, zoom, cache)
        except OSError as error:
            failed.append(f"{name} ({release_day} release): {error}")
            return None
        rows.append([kind, year, name, path.relative_to(folder).as_posix(), release_day, info["capture_date"],
                     info["satellite"], info["provider"], info["resolution_m"], zoom, pixels, round(step, 4),
                     round(left, 2), round(top, 2), round(west, 7), round(south, 7), round(east, 7), round(north, 7)])
        return info["capture_date"]

    def site_date(name, day, lon, lat):
        old = before.get(("site", "", name, day))
        if old:
            return {"capture_date": old[5]}
        try:
            return photo_date(releases[day][1], lon, lat)
        except OSError as error:
            failed.append(f"{name} ({day} release): {error}")
            return None

    for year, days in POINT_RELEASES.items():
        points = list(csv.DictReader(open(DATA / f"points_{year}.csv")))
        for day in days:
            jobs = [pool.submit(add, "point", year, p["point"], f"points_{year}/{p['point']}_{{date}}.jpg", day,
                                float(p["lon"]), float(p["lat"]),
                                (BOX[0] + (int(p["col"]) + 0.5) * 30, BOX[3] - (int(p["row"]) + 0.5) * 30),
                                POINT_SIZE, POINT_PIXELS) for p in points]
            dates = sorted({job.result() for job in jobs if job.result()})
            print(f"points {year}, {day} release: {len(jobs)} photos taken on {', '.join(dates)}")

    outlines = {}
    for row in csv.DictReader(open(DATA / "sites.csv")):
        outlines.setdefault(row["site"], []).append((float(row["lon"]), float(row["lat"])))
    for site in csv.DictReader(open(DATA / "hotspot_checks.csv")):
        name = site["site"]
        if name in outlines:
            x, y = transform("EPSG:4326", UTM, [p[0] for p in outlines[name]], [p[1] for p in outlines[name]])
            centre = ((min(x) + max(x)) / 2, (min(y) + max(y)) / 2)
            size = round(1.3 * max(max(x) - min(x), max(y) - min(y)), -1)
        else:
            x, y = transform("EPSG:4326", UTM, [float(site["lon"])], [float(site["lat"])])
            centre, size = (x[0], y[0]), 450
        (lon,), (lat,) = transform(UTM, "EPSG:4326", [centre[0]], [centre[1]])
        slug = name.lower().replace(" ", "_")
        infos = list(pool.map(lambda day: site_date(name, day, lon, lat), SITE_RELEASES))
        jobs, taken = [], set()
        for day, info in zip(SITE_RELEASES, infos):
            if info is None or info["capture_date"] in taken:
                continue
            taken.add(info["capture_date"])
            jobs.append(pool.submit(add, "site", "", name, f"sites/{slug}_{{date}}.jpg", day, lon, lat, centre, size,
                                    SITE_PIXELS, info if len(info) > 1 else None))
        for job in jobs:
            job.result()
        print(f"{name}: {len(taken)} photos ({', '.join(sorted(taken))})")
    pool.shutdown()
    rows.sort(key=lambda r: (r[0], str(r[1]), r[2], r[5]))

    with open(index, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{len(rows)} photos in {folder} ({len(rows) - len(before)} new), {len(cache)} map tiles downloaded")
    for gap in failed:
        print("  not downloaded, run download.py photos again:", gap)


def box_outline():
    xs = [BOX[0], BOX[2], BOX[2], BOX[0], BOX[0]]
    ys = [BOX[1], BOX[1], BOX[3], BOX[3], BOX[1]]
    return transform(UTM, "EPSG:4326", xs, ys)


def make_kml():
    lons, lats = box_outline()
    ring = " ".join(f"{lon:.6f},{lat:.6f},0" for lon, lat in zip(lons, lats))
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>GNR 607 Topic 3 study area</name>
  <Placemark>
    <name>Study box (Landsat crop, 390 x 300 pixels)</name>
    <Style><LineStyle><color>ff00ffff</color><width>3</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing></outerBoundaryIs></Polygon>
  </Placemark>
  <Placemark>
    <name>IIT Bombay</name>
    <Point><coordinates>{IITB[0]},{IITB[1]},0</coordinates></Point>
  </Placemark>
</Document>
</kml>
"""
    (DATA / "study_area.kml").write_text(kml)


def make_fcc(stack, limits):
    order = [3, 2, 1]
    return np.dstack([np.clip((stack[b] - lo) / (hi - lo), 0, 1) for b, (lo, hi) in zip(order, limits)])


def show_fcc(stacks, dates):
    limits = [np.percentile(np.concatenate([s[b].ravel() for s in stacks.values()]), [2, 98])
              for b in [3, 2, 1]]

    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, year in zip(axes.flat, stacks):
        ax.imshow(make_fcc(stacks[year], limits))
        ax.set_title(f"{year} ({dates[year]})", fontsize=11)
    h, w = next(iter(stacks.values())).shape[1:]
    ax = axes.flat[0]
    ax.plot([12, 12 + 2000 / 30], [h - 15, h - 15], color="white", lw=3)
    ax.text(12, h - 22, "2 km", color="white", fontsize=8)
    ax.annotate("N", xy=(w - 18, 18), xytext=(w - 18, 60), color="white", fontsize=8, ha="center", weight="bold",
                arrowprops=dict(arrowstyle="-|>", color="white"))
    fig.suptitle("Landsat 8 false colour (NIR, red, green) - same stretch for every year", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "day1_fcc_all_years.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    last = max(stacks)
    x, y = transform("EPSG:4326", UTM, [IITB[0]], [IITB[1]])
    col, row = (x[0] - BOX[0]) / 30, (BOX[3] - y[0]) / 30
    fig, ax = plt.subplots(figsize=(10, 7.9))
    ax.imshow(make_fcc(stacks[last], limits))
    ax.plot(col, row, "o", ms=10, mfc="none", mec="yellow", mew=2)
    ax.annotate("IIT Bombay", (col, row), xytext=(10, -12), textcoords="offset points", color="yellow", fontsize=12)
    ax.plot([12, 12 + 2000 / 30], [h - 12, h - 12], color="white", lw=4)
    ax.text(12, h - 18, "2 km", color="white", fontsize=11)
    ax.set_title(f"{last} ({dates[last]}), false colour, north is up", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "day1_fcc_latest.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_times(times):
    years = list(times)
    hours = [times[y][0] for y in years]
    sun = [times[y][1] for y in years]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))

    ax = axes[0]
    ax.axhspan(0, SUNRISE, color="#1f2a44", alpha=0.85, label="night")
    ax.axhspan(SUNSET, 24, color="#1f2a44", alpha=0.85)
    ax.axhspan(SUNRISE, SUNSET, color="#ffe9a8", alpha=0.7, label="daylight (sunrise to sunset)")
    ax.plot(years, hours, "o", color="tab:red", ms=7, label="picture taken")
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 3))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_xticks(years[::2])
    ax.set_ylabel("time of day in India (IST)")
    clock = int(np.mean(hours)), round(np.mean(hours) % 1 * 60)
    ax.set_title(f"Every picture was taken at about {clock[0]:02d}:{clock[1]:02d} IST (late morning)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.bar(years, sun, color="orange")
    for y, s in zip(years, sun):
        ax.text(y, s + 1.5, f"{s:.0f}°", ha="center", fontsize=8)
    ax.set_ylim(0, 90)
    ax.set_xticks(years[::2])
    ax.set_ylabel("sun elevation (degrees above the horizon)")
    ax.set_title("Height of the sun at that moment")
    fig.tight_layout()
    fig.savefig(OUT / "day1_time_of_day.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_worldcover():
    with rasterio.open(DATA / "worldcover_2021.tif") as src:
        wc = src.read(1)
        tf = src.transform

    rows, cols = np.indices(wc.shape)
    x, y = transform("EPSG:4326", UTM, (tf.c + (cols + 0.5) * tf.a).ravel(),
                     (tf.f + (rows + 0.5) * tf.e).ravel())
    x, y = np.reshape(x, wc.shape), np.reshape(y, wc.shape)
    inside = (x >= BOX[0]) & (x <= BOX[2]) & (y >= BOX[1]) & (y <= BOX[3])

    rgb = np.ones(wc.shape + (3,))
    legend = []
    for value, (name, colour) in CLASSES.items():
        rgb[wc == value] = to_rgb(colour)
        share = 100 * np.mean(wc[inside] == value)
        if share >= 0.1:
            legend.append(Patch(color=colour, label=f"{name}  {share:.1f} %"))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.imshow(rgb, extent=(tf.c, tf.c + tf.a * wc.shape[1], tf.f + tf.e * wc.shape[0], tf.f))
    lons, lats = box_outline()
    ax.plot(lons, lats, color="black", lw=2)
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1), title="Share inside the box")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("ESA WorldCover 2021 (10 m) - study box in black")
    fig.tight_layout()
    fig.savefig(OUT / "day1_worldcover_2021.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def redraw():
    stacks, dates, times = {}, {}, {}
    for row in csv.DictReader(open(DATA / "scene_log.csv")):
        year = int(row["year"])
        with rasterio.open(DATA / f"{year}.tif") as src:
            stacks[year] = src.read()
        dates[year] = date.fromisoformat(row["date"]).strftime("%d %b")
        hour, minute = row["time_ist"].split(":")
        times[year] = (int(hour) + int(minute) / 60, float(row["sun_elevation"]))
    show_fcc(stacks, dates)
    show_times(times)
    show_worldcover()
    print("Day 1 pictures redrawn in", OUT)


def covers(ring, lon, lat):
    hit = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def new_place(west, south, east, north):
    zone = int(((west + east) / 2 + 180) // 6) + 1
    utm = f"EPSG:{32600 + zone}"
    x, y = transform("EPSG:4326", utm, [west, east, west, east], [south, south, north, north])
    left = np.floor((min(x) - 15) / 30) * 30 + 15
    right = np.ceil((max(x) - 15) / 30) * 30 + 15
    bottom = np.floor((min(y) - 15) / 30) * 30 + 15
    top = np.ceil((max(y) - 15) / 30) * 30 + 15
    print("put these in download.py:")
    print(f"BOX = ({left:.0f}, {bottom:.0f}, {right:.0f}, {top:.0f})")
    print(f"LONLAT = [{west}, {south}, {east}, {north}]")
    print(f'TILE = "N{int(south // 3 * 3):02d}E{int(west // 3 * 3):03d}"')
    print(f'UTM = "{utm}"')
    print(f"(the box is {(right - left) / 30:.0f} x {(top - bottom) / 30:.0f} pixels of 30 m)")
    body = {"collections": ["landsat-c2-l2"], "bbox": [west, south, east, north],
            "datetime": f"{YEARS[-1] - 1}-12-01/{YEARS[-1]}-02-28", "limit": 100}
    found = {}
    for item in get_json(STAC, body)["features"]:
        p = item["properties"]
        if p["platform"] != "landsat-8":
            continue
        g = item["geometry"]
        ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
        whole = all(covers(ring, a, b) for a, b in [(west, south), (east, south), (west, north), (east, north)])
        found[(p["landsat:wrs_path"], p["landsat:wrs_row"])] = whole
    print("Landsat 8 path and row over this box:")
    for (path, row), whole in sorted(found.items()):
        print(f'  PATH, ROW = "{path}", "{row}"    whole box inside the scene: {"yes" if whole else "no"}')
    if zone != 43:
        print(f"this place is in UTM zone {zone}, so also put UTM = \"{utm}\" at the top of analysis.py")


def main():
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    token = get_json(TOKEN)["token"]
    log, checked, stacks, dates, times = [], [], {}, {}, {}

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for year in YEARS:
            scenes = []
            for item in find_scenes(year):
                qa, _, _ = read_box(item["assets"]["qa_pixel"]["href"], token)
                aerosol, _, _ = read_box(item["assets"]["qa_aerosol"]["href"], token)
                taken = datetime.fromisoformat(item["properties"]["datetime"])
                scenes.append({"item": item, "date": taken.date(), "utc": taken, "ist": taken + IST,
                               "sun": item["properties"]["view:sun_elevation"],
                               "azimuth": item["properties"]["view:sun_azimuth"],
                               "cloud": cloud_percent(qa), "haze": haze_percent(aerosol)})
            if not scenes:
                print(year, ": no Landsat 8 scene found")
                continue

            best = pick_best(scenes)
            item = best["item"]
            stacks[year] = save_year(year, item, token)
            dates[year] = best["date"].strftime("%d %b")
            times[year] = (best["ist"].hour + best["ist"].minute / 60, round(best["sun"], 1))

            for s in sorted(scenes, key=lambda s: s["date"]):
                checked.append([year, s["date"], s["ist"].strftime("%H:%M"), round(s["sun"], 1),
                                round(s["cloud"], 2), round(s["haze"], 1), "yes" if s is best else ""])
            notes = []
            if best["date"].month != 1:
                notes.append("fallback date outside January")
            if best["cloud"] > 1:
                notes.append("some cloud in the box")
            product = item["assets"]["red"]["href"].split("/")[-2]
            log.append([year, best["date"], best["date"].timetuple().tm_yday, best["utc"].strftime("%H:%M"),
                        best["ist"].strftime("%H:%M"), round(best["sun"], 1), round(best["azimuth"], 1),
                        product, item["properties"]["eo:cloud_cover"], round(best["cloud"], 2),
                        round(best["haze"], 1), len(scenes), "; ".join(notes)])
            print(year, best["date"], best["ist"].strftime("%H:%M IST"), f"sun {best['sun']:.1f} deg,",
                  f"cloud {best['cloud']:.2f} %, haze {best['haze']:.1f} %", f"({len(scenes)} scenes checked)")

        get_worldcover()

    with open(DATA / "scene_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "date", "day_of_year", "time_utc", "time_ist", "sun_elevation", "sun_azimuth",
                         "product_id", "scene_cloud_pct", "box_cloud_pct", "box_haze_pct", "scenes_checked",
                         "notes"])
        writer.writerows(log)
    with open(DATA / "scenes_checked.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "date", "time_ist", "sun_elevation", "box_cloud_pct", "box_haze_pct", "chosen"])
        writer.writerows(checked)

    make_kml()
    show_fcc(stacks, dates)
    show_times(times)
    show_worldcover()
    get_rainfall()
    get_outlines()
    get_second_area()
    if (DATA / "points_2014.csv").exists():
        get_photos()
    print("done - data in", DATA, "and pictures in", OUT)


if __name__ == "__main__":
    if sys.argv[1:] == ["day4"]:
        get_rainfall()
        get_outlines()
        print("saved", DATA / "rainfall.csv", "and", DATA / "sites.csv")
    elif sys.argv[1:] == ["airport"]:
        get_second_area()
    elif sys.argv[1:] == ["photos"]:
        get_photos()
    elif sys.argv[1:] == ["figures"]:
        redraw()
    elif sys.argv[1:2] == ["box"]:
        if len(sys.argv) == 6:
            new_place(*[float(a) for a in sys.argv[2:6]])
        else:
            print("write it like this: python download.py box west south east north")
    else:
        main()
