import csv
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.warp import transform

YEARS = list(range(2014, 2027))
BASE = 2018
NAMES = ["blue", "green", "red", "nir", "swir1"]

VIHAR = (slice(40, 170), slice(130, 220))
FOREST = (slice(0, 40), slice(110, 230))
ZOOM = (slice(110, 240), slice(125, 255))

PIXEL_KM2 = 30 * 30 / 1e6
SENSITIVITY = [-0.10, -0.05, 0.0, 0.05, 0.10]
WORLDCOVER = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
              60: "Bare / sparse", 80: "Water", 90: "Wetland", 95: "Mangroves"}
TRAINING = {c: WORLDCOVER[c] for c in [10, 95, 30, 40, 50, 60]}
IITB = (72.9133, 19.1334)
UTM = "EPSG:32643"
CONFIRM = 3
PERIODS = [(2015, 2018), (2019, 2022), (2023, 2026)]
REPLACED = ["built-up or bare", "still partly green", "dark (shadow?)", "water"]
REFERENCE = {10: 1, 20: 1, 95: 1, 50: 0, 60: 0}
CHECK_YEARS = [2020, 2021]
POINT_YEARS = [2014, 2026]
PER_CLASS = 50
SEED = 607

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"


def to_reflectance(dn):
    refl = dn * 0.0000275 - 0.2
    refl[dn == 0] = np.nan
    return refl.astype(np.float32)


def cloudy(qa):
    return (qa & 0b11111) != 0


def norm_diff(a, b):
    total = a + b
    return np.where(total != 0, (a - b) / np.where(total != 0, total, 1), np.nan)


def neighbours(mask):
    p = np.pad(mask, 1, mode="edge")
    rows, cols = mask.shape
    return np.stack([p[r:r + rows, c:c + cols] for r in range(3) for c in range(3)])


def erode(mask):
    return neighbours(mask).all(axis=0)


def dilate(mask):
    return neighbours(mask).any(axis=0)


def opening(mask):
    return dilate(erode(mask))


def closing(mask):
    return erode(dilate(mask))


def majority(mask, valid):
    votes = neighbours(mask & valid).sum(axis=0)
    voters = neighbours(valid).sum(axis=0)
    return (2 * votes > voters) & valid


def histogram(values, bins, lo, hi):
    k = np.clip(((values - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    return np.array([(k == b).sum() for b in range(bins)])


def otsu(values, lo=-0.2, hi=1.0, bins=200):
    p = histogram(values, bins, lo, hi) / values.size
    centre = lo + (np.arange(bins) + 0.5) * (hi - lo) / bins
    score = np.zeros(bins)
    for k in range(1, bins):
        w0, w1 = p[:k].sum(), p[k:].sum()
        if w0 > 0 and w1 > 0:
            m0 = (p[:k] * centre[:k]).sum() / w0
            m1 = (p[k:] * centre[k:]).sum() / w1
            score[k] = w0 * w1 * (m0 - m1) ** 2
    return lo + np.argmax(score) * (hi - lo) / bins, score


def vegetation(ndvi, cut, valid):
    return np.array([majority(year > cut, valid) for year in ndvi])


def year_of_loss(veg):
    out = np.zeros(veg.shape[1:], dtype=int)
    for k in range(1, len(YEARS)):
        recent = veg[max(0, k - 3):k].sum(axis=0) >= min(2, k)
        mostly = 2 * veg[:k].sum(axis=0) >= k
        gone = ~veg[k:].any(axis=0)
        out[(out == 0) & recent & mostly & gone] = YEARS[k]
    return out


def patches(mask):
    label = np.zeros(mask.shape, dtype=int)
    left = mask.copy()
    while left.any():
        patch = np.zeros_like(mask)
        patch[tuple(np.argwhere(left)[0])] = True
        while True:
            grown = dilate(patch) & mask
            if (grown == patch).all():
                break
            patch = grown
        label[patch] = label.max() + 1
        left &= ~patch
    return label


def outline(mask):
    return mask & ~erode(mask)


def ndvi_change(before, after, valid):
    diff = after - before
    mean = diff[valid].mean()
    sd = np.sqrt(((diff[valid] - mean) ** 2).mean())
    return diff, mean, sd, (diff < mean - 2 * sd) & valid, (diff > mean + 2 * sd) & valid


def around(stack, when):
    before, after = np.full(when.shape, np.nan), np.full(when.shape, np.nan)
    for k, year in enumerate(YEARS[1:], start=1):
        m = when == year
        before[m] = stack[max(0, k - 3):k][:, m].mean(axis=0)
        after[m] = stack[k:][:, m].mean(axis=0)
    return before, after


def moved(mask, dr, dc):
    out = np.zeros_like(mask)
    rows, cols = mask.shape
    out[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)] = \
        mask[max(0, dr):rows - max(0, -dr), max(0, dc):cols - max(0, -dc)]
    return out


def correlation(x, y):
    x, y = x - x.mean(), y - y.mean()
    return (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())


def inside(corners, profile):
    tf = profile["transform"]
    x, y = transform("EPSG:4326", UTM, [p[1] for p in corners], [p[0] for p in corners])
    pc = (np.array(x) - tf.c) / tf.a
    pr = (np.array(y) - tf.f) / tf.e
    r, c = np.indices((profile["height"], profile["width"])) + 0.5
    result = np.zeros(r.shape, dtype=bool)
    for k in range(len(pr) - 1):
        r1, c1, r2, c2 = pr[k], pc[k], pr[k + 1], pc[k + 1]
        if r1 != r2:
            result ^= ((r1 > r) != (r2 > r)) & (c < c1 + (r - r1) * (c2 - c1) / (r2 - r1))
    return result


def compass(dx, dy):
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int((np.degrees(np.arctan2(dx, dy)) + 22.5) % 360 // 45)]


def rain_between(rain, first, last):
    return sum(mm for day, mm in rain.items() if first <= day <= last)


def fit_line(x, y, weight):
    xm = (weight * x).sum() / weight.sum()
    ym = (weight * y).sum() / weight.sum()
    a = (weight * (x - xm) * (y - ym)).sum() / (weight * (x - xm) ** 2).sum()
    b = ym - a * xm
    r2 = 1 - (weight * (y - a * x - b) ** 2).sum() / (weight * (y - ym) ** 2).sum()
    return a, b, r2


def find_shift(ref, img, reach=2):
    rows, cols = ref.shape
    b = img[reach:rows - reach, reach:cols - reach]
    b = b - b.mean()
    best = (0, 0, -1.0)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            a = ref[reach + dy:rows - reach + dy, reach + dx:cols - reach + dx]
            a = a - a.mean()
            r = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
            if r > best[2]:
                best = (dy, dx, r)
    return best


def box_sum(img, half):
    total = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    total[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    w = 2 * half + 1
    out = np.full(img.shape, np.nan)
    out[half:img.shape[0] - half, half:img.shape[1] - half] = \
        total[w:, w:] - total[:-w, w:] - total[w:, :-w] + total[:-w, :-w]
    return out


def window_correlation(a, b, half):
    n = (2 * half + 1) ** 2
    sa, sb = box_sum(a, half), box_sum(b, half)
    va = box_sum(a * a, half) - sa * sa / n
    vb = box_sum(b * b, half) - sb * sb / n
    cov = box_sum(a * b, half) - sa * sb / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(va * vb), np.sqrt(np.maximum(va, 0) / n)


def cornerness(img, half):
    gx, gy = np.zeros_like(img), np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    gy[1:-1] = (img[2:] - img[:-2]) / 2
    sxx, syy, sxy = box_sum(gx * gx, half), box_sum(gy * gy, half), box_sum(gx * gy, half)
    return (sxx + syy) / 2 - np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)


def pick_gcps(stack, base, half, reach, apart):
    steady = np.ones(stack.shape[1:], dtype=bool)
    worst = np.ones(stack.shape[1:])
    for img in stack:
        corr = window_correlation(stack[base], img, half)[0]
        steady &= corr >= 0.9
        worst = np.fmin(worst, corr)
    corner = cornerness(stack[base], half)
    score = np.where(steady, corner, 0)
    edge = half + reach + 6
    score[:edge], score[-edge:], score[:, :edge], score[:, -edge:] = 0, 0, 0, 0
    backup = np.where(np.nan_to_num(corner) > 0, np.nan_to_num(worst, nan=-1.0), -1.0)
    backup[:edge], backup[-edge:], backup[:, :edge], backup[:, -edge:] = -np.inf, -np.inf, -np.inf, -np.inf
    rows, cols = score.shape
    parts = [(slice(0, rows // 2), slice(0, cols // 2)), (slice(0, rows // 2), slice(cols // 2, cols)),
             (slice(rows // 2, rows), slice(0, cols // 2)), (slice(rows // 2, rows), slice(cols // 2, cols)),
             (slice(rows // 4, 3 * rows // 4), slice(cols // 4, 3 * cols // 4))]
    points = []
    for part in parts:
        s = np.zeros_like(score)
        s[part] = score[part]
        for r, c in points:
            s[max(0, r - apart):r + apart + 1, max(0, c - apart):c + apart + 1] = 0
        if not np.nanmax(s) > 0:
            s = np.full_like(backup, -np.inf)
            s[part] = backup[part]
            for r, c in points:
                s[max(0, r - apart):r + apart + 1, max(0, c - apart):c + apart + 1] = -np.inf
        r, c = np.unravel_index(np.nanargmax(s), s.shape)
        points.append((int(r), int(c)))
    return points


def best_match(ref, img, row, col, half, reach):
    a = ref[row - half:row + half + 1, col - half:col + half + 1]
    a = a - a.mean()
    size = 2 * reach + 1
    score = np.full((size, size), -1.0)
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            b = img[row + dr - half:row + dr + half + 1, col + dc - half:col + dc + half + 1]
            b = b - b.mean()
            bottom = np.sqrt((a * a).sum() * (b * b).sum())
            if bottom > 0:
                score[dr + reach, dc + reach] = (a * b).sum() / bottom
    i, j = np.unravel_index(np.argmax(score), score.shape)

    def vertex(before, peak, after):
        bend = before - 2 * peak + after
        return 0.5 * (before - after) / bend if bend < 0 else 0.0

    di = vertex(score[i - 1, j], score[i, j], score[i + 1, j]) if 0 < i < size - 1 else 0.0
    dj = vertex(score[i, j - 1], score[i, j], score[i, j + 1]) if 0 < j < size - 1 else 0.0
    return i - reach + di, j - reach + dj, score[i, j]


def worldcover_on_grid(year, profile):
    rows, cols = np.indices((profile["height"], profile["width"]))
    tf = profile["transform"]
    lon, lat = transform(UTM, "EPSG:4326", (tf.c + (cols + 0.5) * tf.a).ravel(),
                         (tf.f + (rows + 0.5) * tf.e).ravel())
    with rasterio.open(DATA / f"worldcover_{year}.tif") as src:
        classes, wt = src.read(1), src.transform
    r = ((np.array(lat) - wt.f) / wt.e).astype(int)
    c = ((np.array(lon) - wt.c) / wt.a).astype(int)
    return classes[r, c].reshape(rows.shape)


def confusion(mapped, truth):
    cm = np.zeros((2, 2))
    for i in (0, 1):
        for j in (0, 1):
            cm[i, j] = ((mapped == i) & (truth == j)).sum()
    return cm


def kappa(cm):
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / n ** 2
    return (po - pe) / (1 - pe)


def divide(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.where(b != 0, a / np.where(b != 0, b, 1), np.nan)


def accuracy(cm):
    right = np.diag(cm)
    return {"overall": right.sum() / cm.sum(), "producer": divide(right, cm.sum(axis=0)),
            "user": divide(right, cm.sum(axis=1)), "kappa": kappa(cm)}


def weighted(cm, share):
    return cm / cm.sum(axis=1, keepdims=True) * np.array(share)[:, None]


def pick_points(classes, per_class, seed):
    rng = np.random.default_rng(seed)
    picked = []
    for c in (1, 0):
        rows, cols = np.nonzero(classes == c)
        for k in rng.choice(rows.size, per_class, replace=False):
            picked.append((int(rows[k]), int(cols[k])))
    return [picked[k] for k in rng.permutation(len(picked))]


def pixel_square(row, col, profile):
    tf = profile["transform"]
    xs = [tf.c + (col + dc) * tf.a for dc in (0, 1, 1, 0, 0)]
    ys = [tf.f + (row + dr) * tf.e for dr in (0, 0, 1, 1, 0)]
    lon, lat = transform(UTM, "EPSG:4326", xs, ys)
    return list(zip(lat, lon))


def on_photo(lon, lat, photo, width, height):
    x, y = transform("EPSG:4326", UTM, np.ravel(lon), np.ravel(lat))
    size = int(photo["pixels"]) * float(photo["metres_per_pixel"])
    col = (np.array(x) - float(photo["left"])) / size * width
    row = (float(photo["top"]) - np.array(y)) / size * height
    return col.reshape(np.shape(lon)), row.reshape(np.shape(lat))


def read_label(text):
    text = text.strip().lower()
    if text in ("v", "veg", "vegetation", "1", "y", "yes", "green"):
        return 1
    if text in ("n", "none", "no", "0", "non-vegetation", "not", "bare", "built"):
        return 0
    return None


def components(stack, valid):
    x = stack[:, valid].T
    x = x - x.mean(axis=0)
    cov = x.T @ x / (x.shape[0] - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    years = np.arange(stack.shape[0]) - (stack.shape[0] - 1) / 2
    for k in range(len(values)):
        flip = vectors[:, k].sum() < 0 if k == 0 else (vectors[:, k] * years).sum() < 0
        if flip:
            vectors[:, k] = -vectors[:, k]
    return values, vectors, x @ vectors


def load_scenes(folder=DATA):
    dn, qa = [], []
    for year in YEARS:
        with rasterio.open(folder / f"{year}.tif") as src:
            img = src.read()
            profile = src.profile
        dn.append(img[:5])
        qa.append(img[5])
    log = {int(r["year"]): r for r in csv.DictReader(open(DATA / "scene_log.csv"))}
    return to_reflectance(np.stack(dn)), np.stack(qa), log, profile


def read_photos():
    path = DATA / "photos" / "photos.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def save_tif(path, bands, names, profile, dtype):
    p = profile.copy()
    p.update(count=len(bands), dtype=dtype, nodata=None)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.stack(bands).astype(dtype))
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, name)


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_kml(path, pins, outlines, title="GNR 607 Topic 3 - Day 4 loss patches"):
    parts = []
    for name, text, lat, lon in pins:
        parts.append(f"  <Placemark><name>{name}</name><description>{text}</description>"
                     f"<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>")
    for name, corners in outlines.items():
        ring = " ".join(f"{lon:.6f},{lat:.6f},0" for lat, lon in corners)
        parts.append(f"  <Placemark><name>{name}</name><Style><LineStyle><color>ff0000ff</color><width>2</width>"
                     f"</LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style><Polygon><outerBoundaryIs>"
                     f"<LinearRing><coordinates>{ring}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
                     f"</Placemark>")
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n'
                    f"<Document>\n  <name>{title}</name>\n" + "\n".join(parts) +
                    "\n</Document>\n</kml>\n", encoding="utf-8")


def prepare():
    refl, qa, log, profile = load_scenes()
    base = YEARS.index(BASE)
    green, red, nir, swir = refl[:, 1], refl[:, 2], refl[:, 3], refl[:, 4]

    cloud = cloudy(qa)
    ndvi_raw = norm_diff(nir, red)
    mndwi = norm_diff(green, swir)
    ndbi = norm_diff(swir, nir)
    water = (mndwi > 0).sum(axis=0) >= 3
    valid = ~cloud.any(axis=0) & ~water

    wc20, wc21 = worldcover_on_grid(2020, profile), worldcover_on_grid(2021, profile)
    deep = erode(erode((mndwi > 0).all(axis=0)))
    dark = np.zeros_like(deep)
    dark[VIHAR] = deep[VIHAR]
    dark &= ~cloud.any(axis=0)
    roofs = (wc20 == 50) & (wc21 == 50) & (ndvi_raw < 0.2).all(axis=0) & valid
    mean_nir = nir.mean(axis=0)
    bright = roofs & (mean_nir >= np.percentile(mean_nir[roofs], 75))
    weight = dark / dark.sum() + bright / bright.sum()
    pif = weight > 0

    norm = np.empty_like(refl)
    fits = []
    for i, year in enumerate(YEARS):
        for b, name in enumerate(NAMES):
            a, off, r2 = fit_line(refl[i, b][pif], refl[base, b][pif], weight[pif])
            norm[i, b] = a * refl[i, b] + off
            fits.append([year, name, round(a, 4), round(off, 5), round(r2, 3)])
    ndvi_pif = norm_diff(norm[:, 3], norm[:, 2])

    shifts = [find_shift(nir[base], nir[i]) for i in range(len(YEARS))]

    forest = np.zeros_like(valid)
    forest[FOREST] = True
    forest &= (wc20 == 10) & (wc21 == 10) & valid
    f_raw = np.array([np.median(ndvi_raw[i][forest]) for i in range(len(YEARS))])
    f_pif = np.array([np.median(ndvi_pif[i][forest]) for i in range(len(YEARS))])
    scale = f_raw.mean() / f_raw
    ndvi = ndvi_raw * scale[:, None, None]

    rest = valid & ~forest
    box_raw = np.array([ndvi_raw[i][rest].mean() for i in range(len(YEARS))])
    box_new = np.array([ndvi[i][rest].mean() for i in range(len(YEARS))])
    days = np.array([int(log[y]["day_of_year"]) for y in YEARS], dtype=float)
    sun = np.array([float(log[y]["sun_elevation"]) for y in YEARS])
    ones = np.ones(len(YEARS))
    date_forest = fit_line(days, f_raw, ones)
    date_before = fit_line(days, box_raw, ones)
    date_after = fit_line(days, box_new, ones)
    sun_vs_day = fit_line(days, sun, ones)
    shadow = 1 / np.tan(np.radians(sun))

    save_tif(DATA / "ndvi.tif", list(ndvi), [str(y) for y in YEARS], profile, "float32")
    save_tif(DATA / "masks.tif", [water, valid, dark, bright, forest],
             ["water", "valid", "pif_dark", "pif_bright", "forest_control"], profile, "uint8")
    save_csv(OUT / "day2_pif_fits.csv", ["year", "band", "gain", "offset", "r2"], fits)
    rows = []
    for i, y in enumerate(YEARS):
        rows.append([y, log[y]["date"], int(days[i]), log[y]["time_ist"], log[y]["sun_elevation"],
                     round(shadow[i], 2), log[y]["box_haze_pct"], shifts[i][0], shifts[i][1],
                     round(shifts[i][2], 3), round(f_raw[i], 3), round(f_pif[i], 3), round(scale[i], 3),
                     round(box_raw[i], 3), round(box_new[i], 3)])
    save_csv(OUT / "day2_checks.csv", ["year", "date", "day_of_year", "time_ist", "sun_elevation",
                                       "shadow_per_m_height", "haze_pct", "shift_rows", "shift_cols", "shift_r",
                                       "forest_ndvi_raw", "forest_ndvi_pif", "scale", "rest_ndvi_raw",
                                       "rest_ndvi_scaled"], rows)

    show_ndvi(ndvi, ~valid, log)
    show_masks(nir[base], water, cloud.any(axis=0), dark, bright, forest, valid)
    show_pif(refl, fits, dark, bright, base)
    show_forest(days, f_raw, f_pif, date_forest, box_raw, box_new, date_before, date_after, sun_vs_day)

    print(f"valid area {100 * valid.mean():.1f} % of the box | water {100 * water.mean():.1f} % | "
          f"cloud in some year {100 * cloud.any(axis=0).mean():.1f} %")
    print(f"PIFs: {dark.sum()} dark (Vihar Lake), {bright.sum()} bright (roofs) | forest control {forest.sum()} px")
    print("shifts against", BASE, ":", sorted({(s[0], s[1]) for s in shifts}),
          f"| lowest correlation {min(s[2] for s in shifts):.3f}")
    print(f"forest NDVI vs day of January: {date_forest[0]:+.4f}/day (r2 {date_forest[2]:.2f}) | "
          f"scale factors {scale.min():.3f} to {scale.max():.3f}")
    print(f"rest of the box vs day of January: {date_before[0]:+.4f}/day (r2 {date_before[2]:.2f}) before, "
          f"{date_after[0]:+.4f}/day (r2 {date_after[2]:.2f}) after scaling")
    print(f"time of day: {sorted({log[y]['time_ist'] for y in YEARS})} IST | sun {sun.min():.1f}-{sun.max():.1f} "
          f"deg, moves with the date (r2 {sun_vs_day[2]:.2f}) | shadows {shadow.min():.2f}-{shadow.max():.2f} "
          f"x building height, towards the north-west")
    return {"ndvi": ndvi, "ndvi_raw": ndvi_raw, "ndbi": ndbi, "mndwi": mndwi, "refl": refl, "valid": valid,
            "water": water, "forest": forest, "scale": scale, "days": days, "sun": sun, "wc20": wc20,
            "wc21": wc21, "log": log, "profile": profile}


def nice_date(text):
    return date.fromisoformat(text).strftime("%d %b")


def show_ndvi(ndvi, hidden, log):
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("#b9d6f2")
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5))
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, year) in enumerate(zip(axes.flat, YEARS)):
        im = ax.imshow(np.where(hidden, np.nan, ndvi[i]), cmap=cmap, vmin=-0.2, vmax=0.9)
        ax.set_title(f"{year} ({nice_date(log[year]['date'])})", fontsize=11)
    map_marks(axes.flat[0])
    fig.subplots_adjust(top=0.93, hspace=0.22, wspace=0.08)
    fig.colorbar(im, ax=axes.flat[13:], orientation="horizontal", fraction=0.5, label="NDVI")
    fig.suptitle("NDVI after forest scaling, same colour scale for every year (water and cloud in blue)",
                 fontsize=14, y=0.98)
    fig.savefig(OUT / "day2_ndvi_all_years.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_masks(nir, water, cloud, dark, bright, forest, valid):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    grey = np.clip(nir / np.percentile(nir, 98), 0, 1)

    img = np.dstack([grey * 0.6 + 0.3] * 3)
    img[water] = (0.2, 0.45, 0.85)
    img[cloud] = (0.9, 0.1, 0.1)
    axes[0].imshow(img)
    axes[0].set_title(f"Water in 3+ years ({100 * water.mean():.1f} %) and\ncloud in any year "
                      f"({100 * cloud.mean():.1f} %, red)")

    img = np.dstack([grey] * 3)
    img[forest] = (0.1, 0.65, 0.2)
    img[dark] = (0.1, 0.3, 0.9)
    img[bright] = (1.0, 0.55, 0.0)
    axes[1].imshow(img)
    axes[1].set_title(f"Reference areas: deep water ({dark.sum()} px, blue),\nold roofs ({bright.sum()} px, "
                      f"orange), forest control ({forest.sum()} px, green)")

    axes[2].imshow(valid, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Valid area used in every year\n({100 * valid.mean():.1f} % of the box, white)")
    for ax in axes:
        ax.axis("off")
        map_marks(ax, colour="#d03b3b")
    fig.tight_layout()
    fig.savefig(OUT / "day2_masks.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_pif(refl, fits, dark, bright, base):
    gain = np.array([f[2] for f in fits]).reshape(len(YEARS), 5)
    offset = np.array([f[3] for f in fits]).reshape(len(YEARS), 5)
    worst = int(np.argmax(np.abs(gain[:, 2] - 1)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for b, colour in [(2, "tab:red"), (3, "tab:purple")]:
        x, y = refl[worst, b], refl[base, b]
        ax.scatter(x[dark], y[dark], s=3, color=colour, alpha=0.25)
        ax.scatter(x[bright], y[bright], s=3, color=colour, alpha=0.25)
        line = np.array([0, 0.35])
        ax.plot(line, gain[worst, b] * line + offset[worst, b], color=colour,
                label=f"{NAMES[b]}: gain {gain[worst, b]:.2f}, offset {offset[worst, b]:+.3f}")
    ax.plot([0, 0.35], [0, 0.35], "k--", lw=1, label="1:1 line")
    ax.set_xlabel(f"reflectance {YEARS[worst]}")
    ax.set_ylabel(f"reflectance {BASE}")
    ax.set_title(f"PIFs, {YEARS[worst]} against {BASE}\n(left cloud: deep water, right cloud: roofs)")
    ax.legend(fontsize=8)

    for ax, values, label in [(axes[1], gain, "Gain a"), (axes[2], offset, "Offset b")]:
        for b, name in enumerate(NAMES):
            ax.plot(YEARS, values[:, b], marker="o", ms=4, label=name)
        ax.axhline(1 if label == "Gain a" else 0, color="k", lw=1, ls="--")
        ax.set_title(f"{label} per band and year")
        ax.set_xticks(YEARS[::2])
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Radiometric check on PIFs: reflectance_{BASE} = a x reflectance_year + b", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "day2_pif_check.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_forest(days, f_raw, f_pif, date_forest, box_raw, box_new, date_before, date_after, sun_vs_day):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    line = np.array([1, 31])

    ax = axes[0]
    ax.plot(YEARS, f_raw, "o-", color="grey", label="as measured")
    ax.plot(YEARS, f_pif, "o--", color="tab:orange", label="after PIF normalisation (not used)")
    ax.axhline(f_raw.mean(), color="tab:green", lw=2, label="after forest scaling (13-year average)")
    for y, v, d in zip(YEARS, f_raw, days):
        ax.annotate(f"{int(d)} Jan", (y, v), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(YEARS[::2])
    ax.set_ylabel("median NDVI of the forest control")
    ax.set_title("Forest control year by year (labels: date of the image)", pad=30)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(days, f_raw, color="grey")
    ax.plot(line, date_forest[0] * line + date_forest[1], color="grey",
            label=f"{date_forest[0]:+.4f} NDVI per day (r² {date_forest[2]:.2f})")
    ax.set_xlabel("day of January")
    ax.set_ylabel("median NDVI of the forest control")
    ax.set_title("Forest: the later in January, the less green", pad=30)
    ax.legend(fontsize=8)
    a, b = sun_vs_day[0], sun_vs_day[1]
    top = ax.secondary_xaxis("top", functions=(lambda d: a * d + b, lambda e: (e - b) / a))
    top.set_xlabel("sun elevation at 11:04 IST (degrees)", fontsize=9)

    ax = axes[2]
    ax.scatter(days, box_raw, color="grey")
    ax.plot(line, date_before[0] * line + date_before[1], color="grey",
            label=f"before: {date_before[0]:+.4f} per day (r² {date_before[2]:.2f})")
    ax.scatter(days, box_new, color="tab:green")
    ax.plot(line, date_after[0] * line + date_after[1], color="tab:green",
            label=f"after scaling: {date_after[0]:+.4f} per day (r² {date_after[2]:.2f})")
    ax.set_xlabel("day of January")
    ax.set_ylabel("mean NDVI, rest of the box")
    ax.set_title("Test on the rest of the box (not used for the scale)", pad=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "day2_forest_check.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def map_vegetation(d):
    ndvi, valid, log = d["ndvi"], d["valid"], d["log"]
    land = valid.sum() * PIXEL_KM2

    pooled = ndvi[:, valid].ravel()
    t, score = otsu(pooled)
    t2, _ = otsu(pooled[pooled > t], lo=t)
    own_t = [otsu(year[valid])[0] for year in ndvi]

    veg = vegetation(ndvi, t, valid)
    dense = veg & (ndvi > t2)
    classes = np.where(valid, 1 + veg + dense, 0)

    area = veg.sum(axis=(1, 2)) * PIXEL_KM2
    share = 100 * area / land
    dense_km2 = dense.sum(axis=(1, 2)) * PIXEL_KM2
    change = np.diff(area)
    change_pct = 100 * change / area[:-1]
    mean_ndvi = [year[valid].mean() for year in ndvi]

    tests = np.array([vegetation(ndvi, t + s, valid).sum(axis=(1, 2)) for s in SENSITIVITY]) * PIXEL_KM2
    steady = (np.sign(np.diff(tests)) == np.sign(change)).all(axis=0)
    unscaled = vegetation(d["ndvi_raw"], t, valid).sum(axis=(1, 2)) * PIXEL_KM2
    plain = ((ndvi > t) & valid).sum(axis=(1, 2)) * PIXEL_KM2
    opened = np.array([closing(opening(year > t)) & valid for year in ndvi]).sum(axis=(1, 2)) * PIXEL_KM2
    ones = np.ones(len(YEARS))
    date_before = fit_line(d["days"], 100 * unscaled / land, ones)
    date_after = fit_line(d["days"], share, ones)
    sun_after = fit_line(d["sun"], share, ones)
    sun_vs_day = fit_line(d["days"], d["sun"], ones)
    trend = fit_line(np.array(YEARS, dtype=float), area, ones)
    jan21 = ndvi[YEARS.index(2021)]
    pure = {c: erode((d["wc20"] == c) & (d["wc21"] == c)) & valid for c in TRAINING}
    training = []
    for c, name in TRAINING.items():
        v = jan21[pure[c]]
        if v.size == 0:
            continue
        training.append([name, v.size, np.median(v), np.percentile(v, 10), np.percentile(v, 90),
                         100 * (v > t).mean()])
    green, grey = jan21[pure[10] | pure[95]], jan21[pure[50] | pure[60]]
    cuts = np.arange(0.2, 0.7, 0.01)
    misses = [((green <= c).mean() + (grey > c).mean()) / 2 for c in cuts]
    best_cut = cuts[np.argmin(misses)]
    start, end = veg[:3].sum(axis=0) >= 2, veg[-3:].sum(axis=0) >= 2
    lost, gained = (start & ~end).sum() * PIXEL_KM2, (end & ~start).sum() * PIXEL_KM2
    by_class = {name: (veg & (d["wc21"] == c)).sum(axis=(1, 2)) * PIXEL_KM2 for c, name in TRAINING.items()}
    swing = {name: np.abs(np.diff(a)).mean() for name, a in by_class.items()}

    save_tif(DATA / "vegetation.tif", list(classes), [str(y) for y in YEARS], d["profile"], "uint8")
    rows = []
    for i, y in enumerate(YEARS):
        row = [y, log[y]["date"], round(land, 2), round(area[i], 2), round(share[i], 2), round(dense_km2[i], 2),
               round(area[i] - dense_km2[i], 2), round(mean_ndvi[i], 3), round(own_t[i], 3)]
        if i == 0:
            row += ["", "", "", ""]
        else:
            row += [round(change[i - 1], 2), round(change_pct[i - 1], 2), "up" if change[i - 1] > 0 else "down",
                    "yes" if steady[i - 1] else "no"]
        rows.append(row)
    save_csv(OUT / "day3_vegetation_area.csv", ["year", "date", "land_km2", "vegetation_km2", "vegetation_pct",
                                                "dense_km2", "sparse_km2", "mean_ndvi", "otsu_this_year_alone",
                                                "change_km2", "change_pct", "direction", "same_at_all_thresholds"],
             rows)
    rows = [[y] + [round(100 * a / land, 2) for a in tests[:, i]] +
            [round(100 * x[i] / land, 2) for x in (unscaled, plain, opened)] for i, y in enumerate(YEARS)]
    save_csv(OUT / "day3_sensitivity.csv", ["year"] + [f"above_{t + s:.2f}" for s in SENSITIVITY] +
             ["unscaled_ndvi", "not_cleaned", "opening_closing"], rows)
    save_csv(OUT / "day3_training.csv", ["class", "pixels", "median_ndvi", "p10", "p90", "pct_above_threshold"],
             [[r[0], r[1], round(r[2], 3), round(r[3], 3), round(r[4], 3), round(r[5], 1)] for r in training])
    rows = [[y, round(area[i], 2)] + [round(a[i], 2) for a in by_class.values()] for i, y in enumerate(YEARS)]
    rows.append(["mean yearly swing", round(np.abs(change).mean(), 2)] + [round(s, 2) for s in swing.values()])
    save_csv(OUT / "day3_area_by_class.csv", ["year", "all (km2)"] + [f"{n} (km2)" for n in by_class], rows)

    show_threshold(pooled, score, t, t2, training, best_cut)
    show_cleaning(ndvi[-1], t, valid, YEARS[-1])
    show_maps(classes, share, log)
    show_main_graph(area, land, change_pct, steady, trend, t)
    show_density(dense_km2, area - dense_km2, land, t, t2)
    show_checks(100 * tests / land, steady, t, d["days"], 100 * unscaled / land, share, date_before, date_after,
                sun_vs_day)

    print(f"threshold {t:.3f} (Otsu, 13 years pooled) | sparse/dense cut {t2:.3f} | "
          f"training areas alone would cut at {best_cut:.2f} | each year alone {min(own_t):.3f}-{max(own_t):.3f}")
    print(f"vegetation {share.min():.1f}-{share.max():.1f} % of the {land:.1f} km2 land area "
          f"({area.min():.1f}-{area.max():.1f} km2) | trend {trend[0]:+.3f} km2/year (r2 {trend[2]:.2f})")
    print(f"{(change > 0).sum()} rises, {(change < 0).sum()} falls | same direction at all 5 thresholds: "
          f"{steady.sum()} of {len(change)}")
    print(f"area vs day of January: {date_before[0]:+.3f} %/day (r2 {date_before[2]:.2f}) unscaled, "
          f"{date_after[0]:+.3f} %/day (r2 {date_after[2]:.2f}) scaled | vs sun height r2 {sun_after[2]:.2f}")
    print(f"cleaning changes the area by {100 * (area / plain - 1).mean():+.1f} % (majority vote) and "
          f"{100 * (opened / plain - 1).mean():+.1f} % (opening + closing)")
    print(f"2014-16 -> 2024-26: {lost:.2f} km2 lost, {gained:.2f} km2 gained")
    print(f"mean yearly swing {np.abs(change).mean():.2f} km2, of which " +
          ", ".join(f"{n} {s:.2f}" for n, s in swing.items()))
    return {"veg": veg, "classes": classes, "t": t, "t2": t2, "area": area, "land": land}


VEG_COLOURS = ["#b9d6f2", "#c9c6bc", "#84bf72", "#2e7d32"]
UP, DOWN = "#0ca30c", "#d03b3b"


def tidy(ax):
    ax.grid(color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def map_marks(ax, colour="#0b0b0b"):
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    width, height = right - left, bottom - top
    km = 2 if width * 0.03 > 8 else 1 if width * 0.03 > 3 else 0.5
    bar = km * 1000 / 30
    edge = "#0b0b0b" if colour == "#ffffff" else "#ffffff"
    halo = [patheffects.withStroke(linewidth=2.5, foreground=edge)]
    x, y = left + 0.04 * width, bottom - 0.05 * height
    ax.plot([x, x + bar], [y, y], color=colour, lw=3, solid_capstyle="butt",
            path_effects=[patheffects.withStroke(linewidth=4.5, foreground=edge)])
    ax.text(x + bar / 2, y - 0.025 * height, f"{km:g} km", ha="center", va="bottom", fontsize=8, color=colour,
            path_effects=halo)
    x, y = right - 0.05 * width, top + 0.05 * height
    ax.annotate("", xy=(x, y + 0.04 * height), xytext=(x, y + 0.16 * height),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.5))
    ax.text(x, y + 0.025 * height, "N", ha="center", va="bottom", fontsize=9, weight="bold", color=colour,
            path_effects=halo)


def show_threshold(pooled, score, t, t2, training, best_cut):
    bins, lo, hi = 200, -0.2, 1.0
    left = lo + np.arange(bins) * (hi - lo) / bins
    count = histogram(pooled, bins, lo, hi)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    colour = np.where(left < t - 1e-6, VEG_COLOURS[1], np.where(left < t2 - 1e-6, VEG_COLOURS[2], VEG_COLOURS[3]))
    ax.bar(left, count / 1000, width=(hi - lo) / bins, align="edge", color=colour)
    for x, label, side in [(t, f"T = {t:.2f} ", "right"), (t2, f" {t2:.2f}", "left")]:
        ax.axvline(x, color="#0b0b0b", lw=1)
        ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(0, -12), textcoords="offset points",
                    ha=side, fontsize=9)
    ax.legend(handles=[Patch(color=c, label=n) for c, n in
                       zip(VEG_COLOURS[1:], ["no vegetation", "sparse", "dense"])], fontsize=8, loc="upper left")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("pixels (thousands)")
    ax.set_title(f"One histogram of all 13 years\n({pooled.size / 1e6:.2f} million pixel values)")
    tidy(ax)

    ax = axes[1]
    ax.plot(left, score / score.max(), color="#2a78d6", lw=2)
    ax.axvline(t, color="#0b0b0b", lw=1)
    ax.set_xlabel("NDVI cut")
    ax.set_ylabel("between-class variance (1 = best cut)")
    ax.set_title(f"Otsu's method: every cut tried,\nthe best one is at {t:.2f}")
    tidy(ax)

    ax = axes[2]
    for k, (name, n, median, p10, p90, above) in enumerate(training):
        ax.plot([p10, p90], [-k, -k], color="#52514e", lw=2.5, solid_capstyle="round")
        ax.plot(median, -k, "o", color="#0b0b0b", ms=6)
        ax.text(0.75, -k, f"{above:.0f} % above T", va="center", fontsize=8.5, color="#52514e")
    ax.set_yticks([-k for k in range(len(training))], [f"{r[0]}\n{r[1]:,} px" for r in training], fontsize=8.5)
    ax.axvline(t, color="#0b0b0b", lw=1)
    ax.axvline(best_cut, color="#0b0b0b", lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("NDVI in January 2021 (dot: median, bar: 10th to 90th percentile)")
    ax.set_title(f"Training areas (WorldCover, same class in 2020 and 2021)\nline: Otsu T = {t:.2f}, "
                 f"dashed: best cut for these areas = {best_cut:.2f}", fontsize=11)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day3_threshold.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_cleaning(ndvi, t, valid, year):
    plain = (ndvi > t) & valid
    versions = [(f"NDVI > {t:.2f}, not cleaned", plain),
                ("Opening + closing, 3 x 3 box (planned)", closing(opening(plain)) & valid),
                ("3 x 3 majority vote (used)", majority(plain, valid))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, (title, m) in zip(axes, versions):
        img = np.zeros(m.shape + (3,))
        img[:] = matplotlib.colors.to_rgb(VEG_COLOURS[1])
        img[m] = matplotlib.colors.to_rgb(VEG_COLOURS[3])
        img[plain & ~m] = matplotlib.colors.to_rgb(DOWN)
        img[m & ~plain] = matplotlib.colors.to_rgb("#2a78d6")
        img[~valid] = matplotlib.colors.to_rgb(VEG_COLOURS[0])
        ax.imshow(img[ZOOM], interpolation="nearest")
        if m is plain:
            title += f"\n{plain.sum():,} vegetation pixels in the whole box"
        else:
            title += f"\nwhole box: {(plain & ~m).sum():,} pixels removed, {(m & ~plain).sum():,} added"
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        map_marks(ax)
    names = ["vegetation", "no vegetation", "water or cloud (not used)", "removed by the cleaning",
             "added by the cleaning"]
    colours = [VEG_COLOURS[3], VEG_COLOURS[1], VEG_COLOURS[0], DOWN, "#2a78d6"]
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(colours, names)], loc="lower center", ncol=5,
               fontsize=10, frameon=False)
    fig.suptitle(f"Cleaning the vegetation map, January {year} (zoom on IIT Bombay, Powai Lake and Vihar Lake)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "day3_cleaning.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_maps(classes, share, log):
    cmap = matplotlib.colors.ListedColormap(VEG_COLOURS)
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5))
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, year) in enumerate(zip(axes.flat, YEARS)):
        ax.imshow(classes[i], cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.set_title(f"{year} ({nice_date(log[year]['date'])}): {share[i]:.1f} %", fontsize=11)
    map_marks(axes.flat[0])
    names = ["water or cloud (not used)", "no vegetation", "sparse vegetation", "dense vegetation"]
    axes.flat[14].legend(handles=[Patch(color=c, label=n) for c, n in zip(VEG_COLOURS, names)], loc="center",
                         fontsize=11, frameon=False)
    fig.subplots_adjust(top=0.93, hspace=0.22, wspace=0.08)
    fig.suptitle("Vegetation maps, same threshold every year (title: share of the land area that is vegetation)",
                 fontsize=14, y=0.98)
    fig.savefig(OUT / "day3_vegetation_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_main_graph(area, land, change_pct, steady, trend, t):
    fig, ax = plt.subplots(figsize=(12, 5.8))
    years = np.array(YEARS, dtype=float)
    ax.plot(years, trend[0] * years + trend[1], "--", color="#898781", lw=1.2)
    ax.plot(YEARS, area, "o-", color="#2e7d32", lw=2, ms=7, zorder=3)
    gap = 0.08 * (area.max() - area.min())
    for i, pct in enumerate(change_pct):
        up = pct > 0
        colour = UP if up else DOWN
        x, y = YEARS[i] + 0.5, (area[i] + area[i + 1]) / 2 + (gap if up else -gap)
        ax.plot(x, y, "^" if up else "v", ms=10, color=colour, mfc=colour if steady[i] else "white", mew=1.5)
        ax.annotate(f"{pct:+.1f} %".replace("-", "−"), (x, y), xytext=(0, 9 if up else -16),
                    textcoords="offset points", ha="center", fontsize=9, color="#52514e")
    ax.set_xticks(YEARS)
    ax.set_ylim(area.min() - 4 * gap, area.max() + 5 * gap)
    ax.set_ylabel("vegetation area (km²)")
    right = ax.secondary_yaxis("right", functions=(lambda a: 100 * a / land, lambda p: p * land / 100))
    right.set_ylabel(f"% of the {land:.1f} km² land area")
    handles = [Line2D([], [], color="#2e7d32", marker="o", lw=2, label=f"vegetation (NDVI above {t:.2f})"),
               Line2D([], [], color="#898781", ls="--", lw=1.2, label=f"straight-line trend: {trend[0]:+.2f} km² per year"),
               Line2D([], [], color=UP, marker="^", ls="none", ms=9, label="rise from the year before"),
               Line2D([], [], color=DOWN, marker="v", ls="none", ms=9, label="fall from the year before"),
               Line2D([], [], color="#52514e", marker="^", mfc="white", ls="none", ms=9,
                      label="hollow: direction changes at some threshold")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", ncol=2)
    ax.set_title("Vegetation cover around IIT Bombay, every January from 2014 to 2026", fontsize=13)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day3_main_graph.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_density(dense_km2, sparse_km2, land, t, t2):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.bar(YEARS, dense_km2, width=0.72, color=VEG_COLOURS[3], edgecolor="white", lw=1.5,
           label=f"dense (NDVI above {t2:.2f})")
    ax.bar(YEARS, sparse_km2, bottom=dense_km2, width=0.72, color=VEG_COLOURS[2], edgecolor="white", lw=1.5,
           label=f"sparse (NDVI {t:.2f} to {t2:.2f})")
    for i in (0, len(YEARS) - 1):
        ax.annotate(f"{dense_km2[i]:.1f}", (YEARS[i], dense_km2[i] / 2), ha="center", va="center", color="white",
                    fontsize=9)
        ax.annotate(f"{sparse_km2[i]:.1f}", (YEARS[i], dense_km2[i] + sparse_km2[i] / 2), ha="center",
                    va="center", color="#0b0b0b", fontsize=9)
    ax.set_xticks(YEARS)
    ax.set_ylabel("km²")
    ax.set_title(f"Dense and sparse vegetation each year (the rest of the {land:.1f} km² land area has no vegetation)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.set_ylim(0, (dense_km2 + sparse_km2).max() * 1.15)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day3_density_classes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_checks(tests, steady, t, days, unscaled, share, date_before, date_after, sun_vs_day):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    ax = axes[0]
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    for s, row, colour in zip(SENSITIVITY, tests, ramp):
        main = s == 0
        ax.plot(YEARS, row, "o-" if main else "-", color=colour, lw=2.5 if main else 1.5, ms=5,
                label=f"NDVI above {t + s:.2f}" + (" (Otsu)" if main else ""))
    ax.set_xticks(YEARS[::2])
    ax.set_ylim(tests.min() - 3, tests.max() + 12)
    ax.set_ylabel("vegetation, % of the land area")
    ax.set_title(f"Threshold moved by up to 0.10 both ways:\n{steady.sum()} of {len(steady)} rises and falls "
                 "keep their direction", pad=30)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    tidy(ax)

    ax = axes[1]
    line = np.array([1, 31])
    ax.scatter(days, unscaled, color="#898781")
    ax.plot(line, date_before[0] * line + date_before[1], color="#898781",
            label=f"before forest scaling: {date_before[0]:+.2f} % per day (r² {date_before[2]:.2f})")
    ax.scatter(days, share, color="#2e7d32")
    ax.plot(line, date_after[0] * line + date_after[1], color="#2e7d32",
            label=f"after forest scaling: {date_after[0]:+.2f} % per day (r² {date_after[2]:.2f})")
    ax.set_xlabel("day of January")
    ax.set_ylabel("vegetation, % of the land area")
    ax.set_title("The vegetation area no longer follows the date\n(or the sun height, which moves with it)", pad=30)
    a, b = sun_vs_day[0], sun_vs_day[1]
    top = ax.secondary_xaxis("top", functions=(lambda x: a * x + b, lambda e: (e - b) / a))
    top.set_xlabel("sun elevation at 11:04 IST (degrees)", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day3_checks.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def find_changes(d, v):
    ndvi, ndbi, valid, log, profile = d["ndvi"], d["ndbi"], d["valid"], d["log"], d["profile"]
    veg, area = v["veg"], v["area"]
    n = len(YEARS)

    loss = np.array([veg[k - 1] & ~veg[k] for k in range(1, n)])
    gain = np.array([~veg[k - 1] & veg[k] & valid for k in range(1, n)])
    loss_km2, gain_km2 = loss.sum(axis=(1, 2)) * PIXEL_KM2, gain.sum(axis=(1, 2)) * PIXEL_KM2
    change = gain_km2 - loss_km2

    lost_in = year_of_loss(veg)
    gained_in = year_of_loss(~veg & valid)
    lost = lost_in > 0
    certain = np.array([YEARS[1] < y <= YEARS[-CONFIRM] for y in YEARS[1:]])
    lasting_loss = np.array([(lost_in == y).sum() for y in YEARS[1:]]) * PIXEL_KM2
    lasting_gain = np.array([(gained_in == y).sum() for y in YEARS[1:]]) * PIXEL_KM2
    temporary = 1 - (lasting_loss.sum() + lasting_gain.sum()) / (loss_km2.sum() + gain_km2.sum())

    shift, drops, rises = np.zeros(n - 1), np.zeros(n - 1), np.zeros(n - 1)
    sudden_loss, sudden_gain = 0, 0
    for k in range(1, n):
        _, shift[k - 1], _, drop, rise = ndvi_change(ndvi[k - 1], ndvi[k], valid)
        drops[k - 1], rises[k - 1] = drop.sum() * PIXEL_KM2, rise.sum() * PIXEL_KM2
        sudden_loss += (drop & (lost_in == YEARS[k])).sum()
        sudden_gain += (rise & (gained_in == YEARS[k])).sum()
    diff, mean, sd, drop_all, rise_all = ndvi_change(ndvi[0], ndvi[-1], valid)

    nir, swir = d["refl"][:, 3], d["refl"][:, 4]
    old = (d["wc20"] == 50) & (d["wc21"] == 50) & valid & ~veg.any(axis=0)
    built_level = np.percentile(ndbi[:, old].mean(axis=0), 25)
    ndbi_before, ndbi_after = around(ndbi, lost_in)
    ndvi_before, ndvi_after = around(ndvi, lost_in)
    wet = lost & (around(d["mndwi"], lost_in)[1] > 0)
    dark = lost & ~wet & (around(nir, lost_in)[1] < np.percentile(nir[:, old].mean(axis=0), 10)) & \
        (around(swir, lost_in)[1] < np.percentile(swir[:, old].mean(axis=0), 10))
    built = lost & ~wet & ~dark & (ndbi_after >= built_level)
    replaced = 1 * built + 2 * (lost & ~wet & ~dark & ~built) + 3 * dark + 4 * wet
    end_ndbi = ndbi[-3:].mean(axis=0)
    new_built = (end_ndbi - ndbi[:3].mean(axis=0) > 0.05) & (end_ndbi >= built_level) & valid
    sun = np.radians(np.mean([float(log[y]["sun_azimuth"]) for y in YEARS]))
    near_tower = np.zeros_like(valid)
    for step in range(1, 5):
        near_tower |= moved(new_built, int(round(-np.cos(sun) * step)), int(round(np.sin(sun) * step)))
    shadow = dark & near_tower

    label = patches(opening(lost))
    tf = profile["transform"]
    home = transform("EPSG:4326", UTM, [IITB[0]], [IITB[1]])
    rows, cols = np.indices(valid.shape)
    spots = []
    for p in range(1, label.max() + 1):
        m = label == p
        year = YEARS[int(np.argmax([(lost_in[m] == y).sum() for y in YEARS]))]
        east, north = tf.c + (cols[m].mean() + 0.5) * tf.a, tf.f + (rows[m].mean() + 0.5) * tf.e
        lon, lat = transform(UTM, "EPSG:4326", [east], [north])
        dx, dy = east - home[0][0], north - home[1][0]
        spots.append({"patch": p, "pixels": int(m.sum()), "year": year, "certain": certain[YEARS.index(year) - 1],
                      "lat": lat[0], "lon": lon[0], "km": np.hypot(dx, dy) / 1000, "towards": compass(dx, dy),
                      "cover": WORLDCOVER[max(WORLDCOVER, key=lambda c: (d["wc20"][m] == c).sum())],
                      "ndvi": (np.nanmean(ndvi_before[m]), np.nanmean(ndvi_after[m])),
                      "ndbi": (np.nanmean(ndbi_before[m]), np.nanmean(ndbi_after[m])),
                      "after": REPLACED[max(range(4), key=lambda a: (replaced[m] == a + 1).sum())]})
    spots.sort(key=lambda s: -s["pixels"])

    city = (d["wc20"] == 10) & (d["wc21"] == 10) & valid & ~d["forest"]
    city_raw = np.array([np.median(year[city]) for year in d["ndvi_raw"]])
    forest_raw = np.array([np.median(year[d["forest"]]) for year in d["ndvi_raw"]])
    ratio = city_raw / forest_raw
    rain = {row["date"]: float(row["rain_mm"]) for row in csv.DictReader(open(DATA / "rainfall.csv"))}
    taken = [date.fromisoformat(log[y]["date"]) for y in YEARS]
    monsoon = np.array([rain_between(rain, f"{y - 1}-06-01", f"{y - 1}-09-30") for y in YEARS])
    after_monsoon = np.array([rain_between(rain, f"{y - 1}-10-01", f"{y - 1}-12-31") for y in YEARS])
    last_90 = np.array([rain_between(rain, str(day - timedelta(days=90)), str(day)) for day in taken])
    r_values = {"city trees / forest": correlation(area, ratio), "rain Jun-Sep": correlation(area, monsoon),
                "rain Oct-Dec": correlation(area, after_monsoon), "rain in the 90 days before": correlation(area, last_90)}
    same_sign = (np.sign(shift) == np.sign(change)).sum()
    mangrove = ((d["wc20"] == 95) | (d["wc21"] == 95)) & valid
    near = dilate(dilate(dilate(mangrove))) & ~mangrove
    belt = {"ndvi": [np.median(year[mangrove]) for year in ndvi],
            "nir": [np.median(year[mangrove]) for year in nir],
            "red": [np.median(year[mangrove]) for year in d["refl"][:, 2]],
            "swir": [np.median(year[mangrove]) for year in swir],
            "vegetation": 100 * veg[:, mangrove].mean(axis=1),
            "water": [((year > 0) & near).sum() * PIXEL_KM2 for year in d["mndwi"]]}

    corners = {}
    for row in csv.DictReader(open(DATA / "sites.csv")):
        corners.setdefault(row["site"], []).append((float(row["lat"]), float(row["lon"])))
    site_veg = {}
    if "Aarey Metro depot" in corners and "Kanjur landfill" in corners:
        depot = inside(corners["Aarey Metro depot"], profile)
        landfill = inside(corners["Kanjur landfill"], profile)
        ring = landfill.copy()
        for _ in range(10):
            ring = dilate(ring)
        ring &= ~landfill
        site_veg = {name: veg[:, m].sum(axis=1) * PIXEL_KM2 * 100 for name, m in
                    [("Aarey Metro depot", depot), ("Kanjur landfill", landfill), ("300 m around the landfill", ring)]}

    save_tif(DATA / "change.tif", [lost_in, gained_in, label, rise_all.astype(int) - drop_all.astype(int)],
             ["year_of_loss", "year_of_gain", "loss_patch", "significant_change_2014_2026"], profile, "int16")
    rows = []
    for k in range(1, n):
        i = k - 1
        rows.append([f"{YEARS[i]}-{YEARS[k]}", round(loss_km2[i], 2), round(lasting_loss[i], 2), round(gain_km2[i], 2),
                     round(lasting_gain[i], 2), round(change[i], 2), round(lasting_loss[:k].sum(), 2),
                     round(lasting_gain[:k].sum(), 2), round(shift[i], 4), round(drops[i], 2), round(rises[i], 2),
                     "yes" if certain[i] else "not yet"])
    for name, pick in [("total 2016-2024 (certain)", certain), ("total 2014-2026", certain | ~certain)]:
        rows.append([name, round(loss_km2[pick].sum(), 2), round(lasting_loss[pick].sum(), 2),
                     round(gain_km2[pick].sum(), 2), round(lasting_gain[pick].sum(), 2), round(change[pick].sum(), 2)] +
                    [""] * 6)
    save_csv(OUT / "day4_change_by_interval.csv",
             ["interval", "loss_km2", "lasting_loss_km2", "gain_km2", "lasting_gain_km2", "net_km2",
              "lasting_loss_so_far_km2", "lasting_gain_so_far_km2", "mean_ndvi_change", "significant_drop_km2",
              "significant_rise_km2", "lasting_part_certain"], rows)
    save_csv(OUT / "day4_loss_patches.csv",
             ["rank", "pixels", "area_ha", "year_of_loss", "certain", "lat", "lon", "km_from_iit_bombay", "direction",
              "worldcover_2020", "ndvi_before", "ndvi_after", "ndbi_before", "ndbi_after", "replaced_by"],
             [[i + 1, s["pixels"], round(s["pixels"] * PIXEL_KM2 * 100, 1), s["year"],
               "yes" if s["certain"] else "not yet", round(s["lat"], 5), round(s["lon"], 5), round(s["km"], 1),
               s["towards"], s["cover"], round(s["ndvi"][0], 2), round(s["ndvi"][1], 2), round(s["ndbi"][0], 3),
               round(s["ndbi"][1], 3), s["after"]] for i, s in enumerate(spots)])
    rows = [[y, log[y]["date"], round(area[i], 2), round(city_raw[i], 3), round(forest_raw[i], 3), round(ratio[i], 3),
             round(monsoon[i]), round(after_monsoon[i]), round(last_90[i]), round(belt["ndvi"][i], 3),
             round(belt["nir"][i], 3), round(belt["red"][i], 3), round(belt["swir"][i], 3),
             round(belt["vegetation"][i], 1), round(belt["water"][i], 2)] +
            [round(ha[i], 1) for ha in site_veg.values()] for i, y in enumerate(YEARS)]
    save_csv(OUT / "day4_yearly_checks.csv",
             ["year", "date", "vegetation_km2", "city_trees_ndvi_raw", "forest_ndvi_raw", "city_to_forest",
              "rain_jun_sep_mm", "rain_oct_dec_mm", "rain_90_days_mm", "mangrove_ndvi", "mangrove_nir",
              "mangrove_red", "mangrove_swir", "mangrove_vegetation_pct", "water_next_to_mangroves_km2"] +
             [f"{name} vegetation (ha)" for name in site_veg], rows)
    km2 = [(replaced == a + 1).sum() * PIXEL_KM2 for a in range(4)]
    save_csv(OUT / "day4_after_loss.csv", ["replaced_by", "pixels", "km2", "pct_of_lasting_loss"],
             [[name, (replaced == a + 1).sum(), round(km2[a], 2), round(100 * km2[a] / (lost.sum() * PIXEL_KM2), 1)]
              for a, name in enumerate(REPLACED)] +
             [["dark and next to a new building (possible shadow)", shadow.sum(), round(shadow.sum() * PIXEL_KM2, 2),
               round(100 * shadow.sum() / lost.sum(), 1)]])
    pins = [(f"Loss {i + 1}: {s['year']}, {s['pixels'] * PIXEL_KM2 * 100:.1f} ha",
             f"year of loss {s['year']} ({'certain' if s['certain'] else 'not yet certain'}); NDVI {s['ndvi'][0]:.2f} "
             f"-> {s['ndvi'][1]:.2f}; NDBI {s['ndbi'][0]:+.2f} -> {s['ndbi'][1]:+.2f}; now {s['after']}",
             s["lat"], s["lon"]) for i, s in enumerate(spots[:10])]
    save_kml(OUT / "day4_hotspots.kml", pins, corners)

    show_change_maps(veg, valid)
    show_gain_loss(loss_km2, lasting_loss, gain_km2, lasting_gain, change, certain)
    show_year_of_loss(lost_in, gained_in, label, spots, veg[0], valid, lasting_loss, lasting_gain, certain)
    show_difference(diff, mean, sd, drop_all, rise_all, valid)
    trees = (d["wc20"] == 10) & (d["wc21"] == 10) & valid & veg.all(axis=0)
    show_after_loss(ndbi, trees, old, lost, ndbi_before, ndbi_after, replaced, shadow, built_level)
    hot = []
    if site_veg:
        hot = [(f"Aarey Metro depot (OpenStreetMap outline, {depot.sum() * PIXEL_KM2 * 100:.0f} ha)", depot,
                {"inside the depot": site_veg["Aarey Metro depot"]}),
               (f"Kanjur landfill (outline, {landfill.sum() * PIXEL_KM2 * 100:.0f} ha) and 300 m around it", landfill,
                {"inside the landfill": site_veg["Kanjur landfill"],
                 "300 m around it": site_veg["300 m around the landfill"]})]
    for s in [s for s in spots if s["certain"]][:2]:
        m = label == s["patch"]
        hot.append((f"Loss patch {spots.index(s) + 1}: lost in {s['year']}, {s['pixels'] * PIXEL_KM2 * 100:.1f} ha, "
                    f"{s['km']:.1f} km {s['towards']} of IIT Bombay", m,
                    {"inside the patch": veg[:, m].sum(axis=1) * PIXEL_KM2 * 100}))
    show_hotspots(hot, d["refl"], valid)
    show_swings(shift, change, area, ratio, after_monsoon, r_values, belt, same_sign)

    print(f"each interval: {loss_km2.mean():.2f} km2 lost and {gain_km2.mean():.2f} km2 gained on average; "
          f"{100 * temporary:.0f} % of it did not last")
    print(f"lasting loss {lasting_loss[certain].sum():.2f} km2 in 2016-2024 (+{lasting_loss[~certain].sum():.2f} not yet "
          f"certain) | lasting gain {lasting_gain[certain].sum():.2f} km2 (+{lasting_gain[~certain].sum():.2f})")
    print(f"loss patches of 3 x 3 pixels or more: {label.max()}, holding {100 * (label > 0).sum() / lost.sum():.0f} % "
          f"of the lasting loss | largest {spots[0]['pixels'] * PIXEL_KM2 * 100:.1f} ha, lost in {spots[0]['year']}")
    print(f"NDVI 2014 -> 2026: mean change {mean:+.3f}, 2 sigma {2 * sd:.3f} -> {drop_all.sum() * PIXEL_KM2:.2f} km2 "
          f"significant drop, {rise_all.sum() * PIXEL_KM2:.2f} km2 rise | in its own year, {100 * sudden_loss / lost.sum():.0f} % "
          f"of the lasting loss was a significant drop and {100 * sudden_gain / (gained_in > 0).sum():.0f} % of the "
          "lasting gain a significant rise")
    print("after the loss: " + ", ".join(f"{name} {100 * x / (lost.sum() * PIXEL_KM2):.0f} %"
                                         for name, x in zip(REPLACED, km2)) +
          f" | dark next to a new building (possible shadow): {shadow.sum()} pixels")
    print(f"swings: the area moves with the whole-box NDVI shift in {same_sign} of {n - 1} intervals | r with area: " +
          ", ".join(f"{k} {x:+.2f}" for k, x in r_values.items()))
    if site_veg:
        print("vegetation (ha) 2014 -> 2026: " + ", ".join(f"{name} {ha[0]:.1f} -> {ha[-1]:.1f}"
                                                            for name, ha in site_veg.items()) +
              f" | outlines: depot {depot.sum() * PIXEL_KM2 * 100:.0f} ha, landfill {landfill.sum() * PIXEL_KM2 * 100:.0f} ha")
    return {"lost_in": lost_in, "gained_in": gained_in, "label": label, "spots": spots}


LOSS, GAIN = "#d03b3b", "#2a78d6"
MAP_BACK = ["#ffffff", "#e3e1da", "#cfe5c4"]
PERIOD_COLOURS = ["#7a5cc4", "#cc4a6e", "#e0a030"]


def background(valid, veg):
    img = np.zeros(valid.shape + (3,))
    img[:] = matplotlib.colors.to_rgb(MAP_BACK[0])
    img[valid] = matplotlib.colors.to_rgb(MAP_BACK[1])
    img[veg] = matplotlib.colors.to_rgb(MAP_BACK[2])
    return img


def false_colour(bands, limits):
    return np.dstack([np.clip((b - lo) / (hi - lo), 0, 1) for b, (lo, hi) in zip(bands, limits)])


def label_points(ax, xs, ys, names):
    done = []
    for x, y, name in zip(xs, ys, names):
        close = any(abs(x - a) < 0.04 * np.ptp(xs) and abs(y - b) < 0.04 * np.ptp(ys) for a, b in done)
        ax.annotate(name, (x, y), xytext=(5, -11 if close else 4), textcoords="offset points", fontsize=8,
                    color="#52514e")
        done.append((x, y))


def show_change_maps(veg, valid):
    pairs = [(k - 1, k) for k in range(1, len(YEARS))] + [(0, len(YEARS) - 1)]
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (a, b) in zip(axes.flat, pairs):
        lost, won = veg[a] & ~veg[b], ~veg[a] & veg[b] & valid
        img = background(valid, veg[a] & veg[b])
        img[lost] = matplotlib.colors.to_rgb(LOSS)
        img[won] = matplotlib.colors.to_rgb(GAIN)
        ax.imshow(img, interpolation="nearest")
        name = f"{YEARS[a]} → {YEARS[b]}" + (", whole period" if b - a > 1 else "")
        ax.set_title(f"{name}\nloss {lost.sum() * PIXEL_KM2:.1f} km², gain {won.sum() * PIXEL_KM2:.1f} km²", fontsize=10)
    map_marks(axes.flat[0])
    names = ["loss: vegetation → none", "gain: none → vegetation", "vegetation in both years",
             "no vegetation in both years", "water or cloud (not used)"]
    colours = [LOSS, GAIN, MAP_BACK[2], MAP_BACK[1], MAP_BACK[0]]
    axes.flat[14].legend(handles=[Patch(facecolor=c, edgecolor="#898781", label=s) for c, s in zip(colours, names)],
                         loc="center", fontsize=10, frameon=False)
    fig.subplots_adjust(top=0.92, hspace=0.32, wspace=0.08)
    fig.suptitle("Change maps: each January against the January before, and 2014 against 2026", fontsize=14, y=0.98)
    fig.savefig(OUT / "day4_change_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_gain_loss(loss_km2, lasting_loss, gain_km2, lasting_gain, change, certain):
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = np.arange(len(change))
    parts = [(lasting_gain, 0, "#2e7d32", True), (gain_km2 - lasting_gain, lasting_gain, "#84bf72", False),
             (-lasting_loss, 0, "#a32626", True), (lasting_loss - loss_km2, -lasting_loss, "#e57373", False)]
    for height, bottom, colour, lasting in parts:
        bars = ax.bar(x, height, bottom=bottom, width=0.7, color=colour, edgecolor="white", lw=1)
        for bar, sure in zip(bars, certain):
            if lasting and not sure:
                bar.set_hatch("////")
    ax.plot(x, change, "D", color="#0b0b0b", ms=6, zorder=3)
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    ax.set_xticks(x, [f"{YEARS[k]}→{YEARS[k + 1] % 100:02d}" for k in x])
    ax.set_ylim(-loss_km2.max() * 1.15, gain_km2.max() * 1.45)
    ax.set_ylabel("km² in each interval (gain up, loss down)")
    handles = [Patch(color="#2e7d32", label="lasting gain"),
               Patch(color="#84bf72", label="gain that did not last"),
               Patch(color="#a32626", label="lasting loss"),
               Patch(color="#e57373", label="loss that did not last (grew back)"),
               Patch(facecolor="#898781", edgecolor="white", hatch="////", label="hatched: not yet certain"),
               Line2D([], [], color="#0b0b0b", marker="D", ls="none", label="net change = the Day 3 rise or fall")]
    ax.legend(handles=handles, fontsize=8.5, ncol=3, loc="upper left")
    ax.set_title("Gain and loss in every interval, split into lasting change and change that did not last", fontsize=12)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day4_gain_loss.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_year_of_loss(lost_in, gained_in, label, spots, first, valid, lasting_loss, lasting_gain, certain):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1, 1, 0.85]})
    for ax, when, name in [(axes[0], lost_in, "loss"), (axes[1], gained_in, "gain")]:
        img = background(valid, first)
        for (a, b), colour in zip(PERIODS, PERIOD_COLOURS):
            img[(when >= a) & (when <= b)] = matplotlib.colors.to_rgb(colour)
        if name == "loss":
            for s in spots[:8]:
                img[outline(dilate(label == s["patch"]))] = (0, 0, 0)
        ax.imshow(img, interpolation="nearest")
        if name == "loss":
            for i, s in enumerate(spots[:8]):
                r, c = np.argwhere(label == s["patch"]).mean(axis=0)
                ax.annotate(str(i + 1), (c, r), xytext=(7, 5), textcoords="offset points", fontsize=10, weight="bold",
                            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
        ax.set_title(f"Year of lasting {name}: {(when > 0).sum() * PIXEL_KM2:.1f} km² in all", fontsize=12)
        ax.axis("off")
        map_marks(ax)
    names = [f"{a}–{b}" for a, b in PERIODS] + ["vegetation in 2014", "no vegetation in 2014", "water or cloud"]
    colours = PERIOD_COLOURS + MAP_BACK[::-1]
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#898781", label=s) for c, s in zip(colours, names)],
               loc="lower center", bbox_to_anchor=(0.36, -0.01), ncol=6, fontsize=10, frameon=False)

    ax = axes[2]
    years = np.array(YEARS[1:])
    for values, colour, name in [(lasting_loss, LOSS, "lasting loss"), (lasting_gain, GAIN, "lasting gain")]:
        total = np.cumsum(values)
        ax.plot(years, total, color=colour, lw=2)
        ax.plot(years[certain], total[certain], "o", color=colour, ms=6)
        ax.plot(years[~certain], total[~certain], "o", color=colour, mfc="white", mew=1.5, ms=6)
        ax.annotate(f"{name}\n{total[-1]:.1f} km²", (years[-1], total[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set_xticks(years[1::2])
    ax.set_xlim(years[0] - 0.5, years[-1] + 2.8)
    ax.set_ylabel("km² since 2014 (running total)")
    ax.legend(handles=[Line2D([], [], color="#52514e", marker="o", ls="none", label="certain (2016–2024)"),
                       Line2D([], [], color="#52514e", marker="o", mfc="white", ls="none",
                              label="not yet certain (2015, 2025, 2026)")], fontsize=8.5, loc="upper left")
    ax.set_title("Running total of lasting loss and gain", fontsize=12)
    tidy(ax)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "day4_year_of_loss.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_difference(diff, mean, sd, drop, rise, valid):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.12, 1, 0.9]})
    cmap = matplotlib.colormaps["BrBG"].copy()
    cmap.set_bad("#ffffff")
    im = axes[0].imshow(np.where(valid, diff, np.nan), cmap=cmap, vmin=-0.3, vmax=0.3)
    fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02, label="NDVI 2026 − NDVI 2014")
    axes[0].set_title("NDVI difference image, 2026 − 2014", fontsize=12)
    img = background(valid, np.zeros_like(valid))
    img[drop] = matplotlib.colors.to_rgb(LOSS)
    img[rise] = matplotlib.colors.to_rgb(GAIN)
    axes[1].imshow(img, interpolation="nearest")
    axes[1].set_title(f"2-sigma rule: significant drop {drop.sum() * PIXEL_KM2:.1f} km² (red),\n"
                      f"significant rise {rise.sum() * PIXEL_KM2:.1f} km² (blue)", fontsize=12)
    for ax in axes[:2]:
        ax.axis("off")
        map_marks(ax)
    ax = axes[2]
    bins, lo, hi = 120, -0.6, 0.6
    width = (hi - lo) / bins
    left = lo + np.arange(bins) * width
    count = histogram(diff[valid], bins, lo, hi)
    colour = np.where(left + width <= mean - 2 * sd, LOSS, np.where(left >= mean + 2 * sd, GAIN, "#c9c6bc"))
    ax.bar(left, count / 1000, width=width, align="edge", color=colour)
    for x, style in [(mean, "-"), (mean - 2 * sd, "--"), (mean + 2 * sd, "--")]:
        ax.axvline(x, color="#0b0b0b", lw=1, ls=style)
    ax.set_xlabel("NDVI change, 2014 → 2026")
    ax.set_ylabel("pixels (thousands)")
    ax.set_title(f"Histogram of the change: mean {mean:+.3f} (line),\nmean ± 2σ = ± {2 * sd:.3f} (dashed)", fontsize=12)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day4_ndvi_difference.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_after_loss(ndbi, trees, old, lost, ndbi_before, ndbi_after, replaced, shadow, built_level):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    groups = [("tree cover, vegetation every year", ndbi[:, trees].mean(axis=0)),
              ("lasting-loss pixels, before", ndbi_before[lost]),
              ("lasting-loss pixels, after", ndbi_after[lost]),
              ("old built-up land", ndbi[:, old].mean(axis=0))]
    for k, (name, values) in enumerate(groups):
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        ax.plot([p10, p90], [-k, -k], color="#52514e", lw=2.5, solid_capstyle="round")
        ax.plot(p50, -k, "o", color="#0b0b0b", ms=6)
    ax.axvline(built_level, color="#0b0b0b", lw=1, ls="--")
    ax.set_yticks([-k for k in range(len(groups))], [g[0] for g in groups])
    ax.set_xlabel("NDBI (dot: median, bar: 10th to 90th percentile)")
    ax.set_title(f"NDBI before and after the loss (dashed: built-up level, {built_level:+.3f})", fontsize=12)
    tidy(ax)

    ax = axes[1]
    for k, name in enumerate(REPLACED):
        share = 100 * (replaced == k + 1).sum() / lost.sum()
        ax.barh(-k, share, color="#52514e", height=0.6)
        ax.annotate(f"{share:.0f} %", (share, -k), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    ax.barh(-2, 100 * shadow.sum() / lost.sum(), color="#52514e", height=0.6, hatch="////", edgecolor="white")
    ax.set_yticks([-k for k in range(len(REPLACED))], REPLACED)
    ax.set_xlabel("% of the lasting loss")
    ax.set_title(f"What replaced the lost vegetation\n(hatched: dark and next to a new building, {shadow.sum()} pixels)",
                 fontsize=12)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day4_after_loss.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_hotspots(rows, refl, valid):
    order = [3, 2, 1]
    limits = [np.percentile(refl[:, b][:, valid], [2, 98]) for b in order]
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 3.9 * len(rows)), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    for (title, mask, series), ax_row in zip(rows, axes):
        rr, cc = np.nonzero(mask)
        pad = max(10, (rr.max() - rr.min()) // 2, (cc.max() - cc.min()) // 3)
        win = (slice(max(0, rr.min() - pad), rr.max() + pad + 1), slice(max(0, cc.min() - pad), cc.max() + pad + 1))
        edge = outline(dilate(mask))[win]
        for ax, i in zip(ax_row[:2], (0, len(YEARS) - 1)):
            img = false_colour([refl[i, b][win] for b in order], limits)
            img[edge] = (1, 1, 0)
            ax.imshow(img, interpolation="nearest")
            ax.set_title(f"{YEARS[i]} ({img.shape[1] * 0.03:.1f} km × {img.shape[0] * 0.03:.1f} km)", fontsize=10)
            ax.axis("off")
            map_marks(ax, colour="#ffffff")
        ax = ax_row[2]
        for (name, ha), colour in zip(series.items(), ["#2e7d32", "#898781"]):
            ax.plot(YEARS, ha, "o-", color=colour, lw=2, ms=5, label=name)
        ax.set_ylim(bottom=0)
        ax.set_xticks(YEARS[::2])
        ax.set_ylabel("vegetation (ha)")
        ax.legend(fontsize=8.5, loc="lower left")
        ax.set_title(title, fontsize=11)
        tidy(ax)
    fig.suptitle("Hotspots: false colour (vegetation red) in 2014 and 2026, outline in yellow, and the vegetation "
                 "inside every January", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "day4_hotspots.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_swings(shift, change, area, ratio, after_monsoon, r_values, belt, same_sign):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    ax = axes[0, 0]
    for i in range(len(change)):
        up = change[i] > 0
        ax.plot(shift[i], change[i], "^" if up else "v", color=UP if up else DOWN, ms=9)
        ax.annotate(f"{YEARS[i]}→{YEARS[i + 1] % 100:02d}", (shift[i], change[i]), xytext=(6, 3),
                    textcoords="offset points", fontsize=8, color="#52514e")
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_xlabel("mean NDVI change of the whole box (the shift)")
    ax.set_ylabel("change in vegetation area (km²)")
    ax.set_title(f"The area rises when the whole box turns greener\n(same direction in {same_sign} of {len(change)} "
                 "intervals)", fontsize=12)
    tidy(ax)

    panels = [(axes[0, 1], ratio, "raw NDVI of the city's trees ÷ raw NDVI of the park forest", "city trees / forest",
               "The area follows how green the city's trees are\nnext to the park forest"),
              (axes[1, 0], after_monsoon, "rain from October to December before the image (mm, ERA5)", "rain Oct-Dec",
               "Rain after the monsoon explains only part of it")]
    for ax, x, xlabel, key, title in panels:
        fit = fit_line(x, area, np.ones(len(x)))
        line = np.array([x.min(), x.max()])
        ax.plot(line, fit[0] * line + fit[1], "--", color="#898781", lw=1.2)
        ax.plot(x, area, "o", color="#2e7d32", ms=7)
        label_points(ax, x, area, [str(year) for year in YEARS])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("vegetation area (km²)")
        ax.set_title(f"{title} (r = {r_values[key]:.2f})", fontsize=12)
        tidy(ax)
    axes[1, 0].text(0.98, 0.04, "r with the area:\n" + "\n".join(f"{k}: {x:+.2f}" for k, x in r_values.items()
                                                                if k.startswith("rain")),
                    transform=axes[1, 0].transAxes, ha="right", va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="#c9c6bc"))

    ax = axes[1, 1]
    for key, colour, name in [("nir", "#2a78d6", "near-infrared"), ("red", "#d03b3b", "red"),
                              ("swir", "#e0a030", "short-wave infrared")]:
        ax.plot(YEARS, belt[key], "o-", color=colour, lw=2, ms=5, label=name)
    for year in (2014, 2021, 2022):
        ax.axvspan(year - 0.4, year + 0.4, color="#e3e1da", zorder=0)
    ax.set_xticks(YEARS[::2])
    if np.isfinite(belt["nir"]).any():
        ax.set_ylim(0, np.nanmax(belt["nir"]) * 1.25)
    ax.set_ylabel("median reflectance, mangrove belt")
    ax.legend(fontsize=8.5, ncol=3, loc="upper center")
    ax.set_title("Mangrove belt: in the dip years (grey) NIR falls, SWIR hardly moves", fontsize=12)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day4_swings.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def accuracy_row(name, cm):
    a = accuracy(cm)
    return [name, int(round(cm.sum())), round(100 * a["overall"], 1), round(100 * a["producer"][1], 1),
            round(100 * a["producer"][0], 1), round(100 * a["user"][1], 1), round(100 * a["user"][0], 1),
            round(a["kappa"], 3)]


def check_accuracy(d, v, c):
    ndvi, valid, veg, profile = d["ndvi"], d["valid"], v["veg"], d["profile"]
    table, matrices = [], []

    wc = {2020: d["wc20"], 2021: d["wc21"]}
    reference, results = {}, {}
    for year in CHECK_YEARS:
        ref = np.full(valid.shape, -1)
        for cls, value in REFERENCE.items():
            ref[wc[year] == cls] = value
        ref[~valid] = -1
        pure = (erode(ref == 1) | erode(ref == 0)) & valid
        ours = veg[YEARS.index(year)]
        for name, use in [("all pixels", ref >= 0), ("away from class edges", pure)]:
            cm = confusion(ours[use], ref[use])
            label = f"WorldCover {year} vs our January {year} map, {name}"
            table.append(accuracy_row(label, cm))
            matrices.append((label, cm))
            results[(year, name)] = cm
        reference[year] = ref

    k21 = YEARS.index(2021)
    use = reference[2021] >= 0
    cuts = np.round(np.arange(0.20, 0.71, 0.02), 2)
    kappas = [kappa(confusion(majority(ndvi[k21] > t, valid)[use], reference[2021][use])) for t in cuts]
    best = cuts[int(np.argmax(kappas))]

    by_class = []
    for cls, name in WORLDCOVER.items():
        rows = [name]
        for year in CHECK_YEARS:
            m = (wc[year] == cls) & valid
            rows += [int(m.sum()), round(100 * veg[YEARS.index(year)][m].mean(), 1) if m.any() else ""]
        by_class.append(rows)

    photos = {}
    for r in read_photos():
        if r["kind"] == "point":
            photos.setdefault((r["year"], r["name"]), []).append(r)
    scored = {}
    for year in POINT_YEARS:
        path = DATA / f"points_{year}.csv"
        k = YEARS.index(year)
        mapped = np.where(valid, veg[k], -1)
        if not path.exists():
            tf = profile["transform"]
            rows = []
            for i, (r, col) in enumerate(pick_points(mapped, PER_CLASS, SEED + year), start=1):
                lon, lat = transform(UTM, "EPSG:4326", [tf.c + (col + 0.5) * tf.a], [tf.f + (r + 0.5) * tf.e])
                rows.append([f"P{i:03d}", round(lat[0], 6), round(lon[0], 6), r, col, "", ""])
            save_csv(path, ["point", "lat", "lon", "row", "col", "label", "note"], rows)
        points = list(csv.DictReader(open(path)))
        save_kml(OUT / f"day5_points_{year}.kml",
                 [(p["point"], f"Is the 30 m square mostly green vegetation (trees, shrubs, mangroves)? Write v or n "
                               f"in the label column of data/points_{year}.csv", float(p["lat"]), float(p["lon"]))
                  for p in points],
                 {p["point"]: pixel_square(int(p["row"]), int(p["col"]), profile) for p in points},
                 title=f"GNR 607 Topic 3 - Day 5 reference points {year}")
        prow, pcol = np.array([int(p["row"]) for p in points]), np.array([int(p["col"]) for p in points])
        labels = [read_label(p["label"]) for p in points]
        done = np.array([lab is not None for lab in labels])
        truth = np.array([lab if lab is not None else -1 for lab in labels])
        ours = mapped[prow, pcol]
        share = [(mapped == 0).sum() / (mapped >= 0).sum(), (mapped == 1).sum() / (mapped >= 0).sum()]
        gaps = [(date.fromisoformat(r["capture_date"]) - date.fromisoformat(d["log"][year]["date"])).days
                for p in points for r in photos.get((str(year), p["point"]), [])]
        scored[year] = {"rows": prow, "cols": pcol, "ours": ours, "truth": truth, "done": done, "share": share,
                        "points": points, "gaps": gaps}
        if done.sum() >= 10 and len(set(truth[done])) == 2 and len(set(ours[done])) == 2:
            cm = confusion(ours[done], truth[done])
            table.append(accuracy_row(f"Reference points {year}, counts", cm))
            matrices.append((f"Reference points {year}", cm))
            est = weighted(cm, share)
            row = accuracy_row(f"Reference points {year}, weighted by map area", est)
            row[1] = int(done.sum())
            table.append(row)

    path = DATA / "hotspot_checks.csv"
    if not path.exists():
        rows = [[f"Loss patch {i + 1}", round(s["lat"], 5), round(s["lon"], 5), s["year"],
                 round(s["pixels"] * PIXEL_KM2 * 100, 1), "", "", "", ""] for i, s in enumerate(c["spots"][:10])]
        for name in ("Aarey Metro depot", "Kanjur landfill"):
            pts = [(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(DATA / "sites.csv"))
                   if r["site"] == name]
            if pts:
                rows.append([name, round(np.mean([p[0] for p in pts]), 5), round(np.mean([p[1] for p in pts]), 5),
                             "", "", "", "", "", ""])
        save_csv(path, ["site", "lat", "lon", "our_year_of_loss", "area_ha", "photo_before", "photo_after",
                        "what_the_photos_show", "confirmed"], rows)
    checks = list(csv.DictReader(open(path)))
    verdicts = [r["confirmed"].strip().lower() for r in checks if r["confirmed"].strip()]

    save_csv(OUT / "day5_accuracy.csv", ["comparison", "pixels_or_points", "overall_pct", "producer_vegetation_pct",
                                         "producer_none_pct", "user_vegetation_pct", "user_none_pct", "kappa"], table)
    rows = []
    for name, cm in matrices:
        rows += [[name, "our map: vegetation", int(cm[1, 1]), int(cm[1, 0])],
                 [name, "our map: none", int(cm[0, 1]), int(cm[0, 0])]]
    save_csv(OUT / "day5_confusion_matrices.csv", ["comparison", "row", "reference: vegetation", "reference: none"],
             rows)
    save_csv(OUT / "day5_worldcover_classes.csv",
             ["worldcover_class"] + [f"{h}_{y}" for y in CHECK_YEARS for h in ("pixels", "pct_vegetation_on_our_map")],
             by_class)
    save_csv(OUT / "day5_threshold_vs_kappa.csv", ["threshold", "kappa_vs_worldcover_2021"],
             [[t, round(k, 3)] for t, k in zip(cuts, kappas)])
    names = {1: "v", 0: "n", -1: ""}
    taken = {key: [r["capture_date"] for r in found] for key, found in photos.items()}
    rows = []
    for year in POINT_YEARS:
        s = scored[year]
        for i, p in enumerate(s["points"]):
            rows.append([year, p["point"], p["lat"], p["lon"], names[int(s["ours"][i])], p["label"],
                         "" if not s["done"][i] else ("yes" if s["ours"][i] == s["truth"][i] else "no"),
                         " ".join(sorted(taken.get((str(year), p["point"]), []))), p["note"]])
    save_csv(OUT / "day5_point_results.csv", ["year", "point", "lat", "lon", "our_map", "photo_label", "agree",
                                              "photo_dates", "note"], rows)

    show_agreement(veg[k21], reference[2021], d["wc21"], valid, results, cuts, kappas, v["t"])
    show_points(veg, valid, scored)
    show_point_photos(scored, read_photos(), profile)
    show_site_photos(checks, read_photos(), c, profile, d["log"])

    for row in table:
        print(f"{row[0]}: {row[1]} | overall {row[2]} %, producer's veg/none {row[3]}/{row[4]} %, "
              f"user's veg/none {row[5]}/{row[6]} %, kappa {row[7]}")
    print(f"kappa against WorldCover 2021 is highest at NDVI {best:.2f} ({max(kappas):.3f}); at our T = {v['t']:.2f} "
          f"it is {kappa(results[(2021, 'all pixels')]):.3f}")
    print("WorldCover class -> % vegetation on our 2021 map: " +
          ", ".join(f"{r[0]} {r[4]}" for r in by_class if r[3]))
    for year in POINT_YEARS:
        gaps = scored[year]["gaps"]
        print(f"reference points {year}: {scored[year]['done'].sum()} of {len(scored[year]['done'])} labelled "
              f"(data/points_{year}.csv)" + (f"; photos taken {min(gaps)} to {max(gaps)} days from the Landsat date"
                                             if gaps else "; no photos yet (download.py photos)"))
    print(f"hotspot checks: {len(verdicts)} of {len(checks)} filled in (data/hotspot_checks.csv)" +
          (": " + ", ".join(f"{w} {verdicts.count(w)}" for w in sorted(set(verdicts))) if verdicts else ""))


def find_components(d, v, c):
    ndvi, valid = d["ndvi"], d["valid"]
    values, vectors, scores = components(ndvi, valid)
    pct = 100 * values / values.sum()
    years = np.array(YEARS, dtype=float)
    lost, gained = c["lost_in"] > 0, c["gained_in"] > 0
    stable = valid & ~lost & ~gained
    rows, notes = [], []
    for k in range(len(YEARS)):
        image = np.full(valid.shape, np.nan)
        image[valid] = scores[:, k]
        r_year = correlation(years, vectors[:, k])
        r_area = correlation(v["area"], vectors[:, k])
        mean = [np.nanmean(image[m]) for m in (lost, gained, stable)]
        rows.append([f"PC{k + 1}", round(values[k], 5), round(pct[k], 2), round(pct[:k + 1].sum(), 2),
                     round(r_year, 2), round(r_area, 2)] + [round(x, 3) for x in mean] +
                    [round(x, 3) for x in vectors[:, k]])
        if k < 4:
            notes.append(f"PC{k + 1} {pct[k]:.1f} % (r with year {r_year:+.2f}, with area {r_area:+.2f}; "
                         f"mean score lost/gained/stable {mean[0]:+.2f}/{mean[1]:+.2f}/{mean[2]:+.2f})")
    save_csv(OUT / "day5_pca.csv", ["component", "eigenvalue", "pct_variance", "cumulative_pct", "r_loadings_vs_year",
                                    "r_loadings_vs_area", "mean_score_lasting_loss", "mean_score_lasting_gain",
                                    "mean_score_stable"] + [f"loading_{y}" for y in YEARS], rows)
    images = []
    for k in range(3):
        image = np.full(valid.shape, np.nan)
        image[valid] = scores[:, k]
        images.append(image)
    show_components(pct, vectors, images, lost, gained)
    print("PCA of the 13 NDVI images: " + "; ".join(notes))


def second_area(d, v):
    folder = DATA / "airport"
    if not (folder / f"{YEARS[-1]}.tif").exists():
        print("second area skipped: run  download.py airport  first")
        return
    refl, qa, log, profile = load_scenes(folder)
    green, red, nir, swir = refl[:, 1], refl[:, 2], refl[:, 3], refl[:, 4]
    cloud = cloudy(qa)
    mndwi = norm_diff(green, swir)
    water = (mndwi > 0).sum(axis=0) >= 3
    valid = ~cloud.any(axis=0) & ~water
    ndvi_raw = norm_diff(nir, red)
    ndvi = ndvi_raw * d["scale"][:, None, None]
    veg = vegetation(ndvi, v["t"], valid)
    land = valid.sum() * PIXEL_KM2
    area = veg.sum(axis=(1, 2)) * PIXEL_KM2
    unscaled = vegetation(ndvi_raw, v["t"], valid).sum(axis=(1, 2)) * PIXEL_KM2
    open_water = (((mndwi > 0) & ~cloud).sum(axis=(1, 2))) * PIXEL_KM2
    change = np.diff(area)
    lost_in, gained_in = year_of_loss(veg), year_of_loss(~veg & valid)
    certain = np.array([YEARS[1] < y <= YEARS[-CONFIRM] for y in YEARS[1:]])
    lasting_loss = np.array([(lost_in == y).sum() for y in YEARS[1:]]) * PIXEL_KM2
    lasting_gain = np.array([(gained_in == y).sum() for y in YEARS[1:]]) * PIXEL_KM2
    main_share = 100 * v["area"] / v["land"]
    trend = fit_line(np.array(YEARS, dtype=float), area, np.ones(len(YEARS)))

    save_tif(DATA / "airport_vegetation.tif", [np.where(valid, 1 + veg[i], 0) for i in range(len(YEARS))] +
             [lost_in, gained_in], [str(y) for y in YEARS] + ["year_of_loss", "year_of_gain"], profile, "int16")
    rows = []
    for i, y in enumerate(YEARS):
        rows.append([y, log[y]["date"], round(land, 2), round(area[i], 2), round(100 * area[i] / land, 2),
                     round(100 * unscaled[i] / land, 2), round(main_share[i], 2), round(open_water[i], 2),
                     "" if i == 0 else round(change[i - 1], 2), "" if i == 0 else round(lasting_loss[i - 1], 2),
                     "" if i == 0 else round(lasting_gain[i - 1], 2)])
    save_csv(OUT / "day5_airport_area.csv", ["year", "date", "land_km2", "vegetation_km2", "vegetation_pct",
                                             "vegetation_pct_unscaled", "main_box_vegetation_pct", "open_water_km2",
                                             "change_km2", "lasting_loss_km2", "lasting_gain_km2"], rows)
    show_second_area(refl, valid, veg, area, land, main_share, lost_in, open_water, trend)

    print(f"airport box ({profile['width']} x {profile['height']} px, land {land:.1f} km2): vegetation "
          f"{100 * area[0] / land:.1f} % (2014) -> {100 * area[-1] / land:.1f} % (2026), trend {trend[0]:+.2f} km2/year "
          f"(r2 {trend[2]:.2f}); {(change < 0).sum()} falls, {(change > 0).sum()} rises")
    print(f"airport box lasting loss {lasting_loss[certain].sum():.2f} km2 in 2016-2024 "
          f"(+{lasting_loss[~certain].sum():.2f} not yet certain), lasting gain {lasting_gain[certain].sum():.2f} km2 | "
          f"largest yearly loss {lasting_loss.max():.2f} km2 in {YEARS[1 + int(np.argmax(lasting_loss))]} | "
          f"open water {open_water[0]:.2f} -> {open_water[-1]:.2f} km2 | area follows the main box: "
          f"r = {correlation(area, v['area']):+.2f}")


def draw_matrix(ax, cm, title):
    a = accuracy(cm)
    shown = cm[::-1, ::-1]
    share = shown / shown.sum()
    ax.imshow(share, cmap="Greens", vmin=0, vmax=share.max() * 1.6)
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, f"{shown[i, j]:,.0f}\n({100 * share[i, j]:.1f} %)", ha="center", va="center", fontsize=9)
    ax.set_xticks([0, 1], ["vegetation", "none"])
    ax.set_yticks([0, 1], ["vegetation", "none"])
    ax.set_xlabel("WorldCover (reference)")
    ax.set_ylabel("our map")
    ax.set_title(f"{title}\noverall {100 * a['overall']:.1f} %, kappa {a['kappa']:.2f}", fontsize=10.5)


def show_agreement(ours, ref, wc, valid, results, cuts, kappas, t):
    fig = plt.figure(figsize=(17, 9.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    colours = {"both vegetation": "#2e7d32", "both none": "#d6d3ca", "only our map: vegetation": GAIN,
               "only WorldCover: vegetation": LOSS, "WorldCover grassland or cropland": "#e8d36a",
               "water, wetland or cloud (not used)": "#ffffff"}
    img = np.ones(valid.shape + (3,))
    masks = [ours & (ref == 1), ~ours & (ref == 0), ours & (ref == 0), ~ours & (ref == 1),
             valid & ((wc == 30) | (wc == 40)), ~valid | ((ref < 0) & (wc != 30) & (wc != 40))]
    for m, colour in zip(masks, colours.values()):
        img[m] = matplotlib.colors.to_rgb(colour)
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    map_marks(ax)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#898781", label=n) for n, c in colours.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9.5, frameon=False)
    ax.set_title("Our January 2021 map against ESA WorldCover 2021", fontsize=12)

    draw_matrix(fig.add_subplot(grid[0, 1]), results[(2021, "all pixels")], "2021, all pixels")
    draw_matrix(fig.add_subplot(grid[0, 2]), results[(2021, "away from class edges")], "2021, away from class edges")
    draw_matrix(fig.add_subplot(grid[1, 1]), results[(2020, "all pixels")], "2020, all pixels")
    ax = fig.add_subplot(grid[1, 2])
    ax.plot(cuts, kappas, "o-", color="#2a78d6", ms=4, lw=2)
    ax.axvline(t, color="#0b0b0b", lw=1)
    ax.annotate(f" Otsu T = {t:.2f}", (t, min(kappas)), fontsize=9)
    ax.set_xlabel("NDVI threshold")
    ax.set_ylabel("kappa against WorldCover 2021")
    ax.set_title(f"Agreement for every threshold\n(best at {cuts[int(np.argmax(kappas))]:.2f})", fontsize=10.5)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "day5_accuracy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_points(veg, valid, scored):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, year in zip(axes, POINT_YEARS):
        s = scored[year]
        ax.imshow(background(valid, veg[YEARS.index(year)]), interpolation="nearest")
        right = s["done"] & (s["ours"] == s["truth"])
        wrong = s["done"] & (s["ours"] != s["truth"])
        todo = ~s["done"]
        for m, marker, colour, name in [(todo & (s["ours"] == 1), "o", "#2e7d32", "vegetation on our map, not labelled"),
                                        (todo & (s["ours"] == 0), "s", "#52514e", "none on our map, not labelled"),
                                        (right, "o", GAIN, "photo agrees with our map"),
                                        (wrong, "X", LOSS, "photo disagrees")]:
            if m.any():
                ax.plot(s["cols"][m], s["rows"][m], marker, color=colour, mfc="white" if m is not wrong else colour,
                        mew=1.5, ms=6, ls="none", label=f"{name} ({m.sum()})")
        ax.axis("off")
        map_marks(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9, frameon=False)
        ax.set_title(f"January {year}: {len(s['done'])} random points, {PER_CLASS} per class of our map", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "day5_points.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_components(pct, vectors, images, lost, gained):
    fig = plt.figure(figsize=(17, 10))
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, 0])
    ax.bar(range(1, 9), pct[:8], color="#52514e")
    for k in range(4):
        ax.annotate(f"{pct[k]:.1f} %", (k + 1, pct[k]), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9)
    ax.set_xticks(range(1, 9), [f"PC{k}" for k in range(1, 9)])
    ax.set_ylabel("% of the variance")
    ax.set_title("Share of the variance in each component", fontsize=12)
    tidy(ax)
    ax = fig.add_subplot(grid[0, 1:])
    for k, colour in zip(range(3), ["#0b0b0b", "#2a78d6", "#e0a030"]):
        ax.plot(YEARS, vectors[:, k], "o-", color=colour, lw=2, ms=5, label=f"PC{k + 1} ({pct[k]:.1f} %)")
    ax.axhline(0, color="#898781", lw=0.8)
    ax.set_xticks(YEARS)
    ax.set_ylabel("weight of each year (eigenvector)")
    ax.legend(fontsize=9)
    ax.set_title("What each component is made of: PC1 weighs all years alike (average greenness); "
                 "the next ones contrast years", fontsize=12)
    tidy(ax)
    cmap = matplotlib.colormaps["BrBG"].copy()
    cmap.set_bad("#ffffff")
    for k, image in enumerate(images):
        ax = fig.add_subplot(grid[1, k])
        lim = np.nanpercentile(np.abs(image), 98)
        im = ax.imshow(image, cmap=cmap if k else "Greens", vmin=-lim if k else np.nanpercentile(image, 2), vmax=lim)
        if k:
            overlay = np.zeros(lost.shape + (4,))
            overlay[outline(lost)] = matplotlib.colors.to_rgba(LOSS)
            ax.imshow(overlay, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=f"PC{k + 1} score")
        ax.axis("off")
        map_marks(ax)
        ax.set_title(f"PC{k + 1}" + (" (red outline: lasting loss)" if k else " (high = green every year)"),
                     fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "day5_pca.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def show_second_area(refl, valid, veg, area, land, main_share, lost_in, open_water, trend):
    order = [3, 2, 1]
    limits = [np.percentile(refl[:, b][:, valid], [2, 98]) for b in order]
    fig = plt.figure(figsize=(17, 10))
    grid = fig.add_gridspec(2, 3)
    for k, i in enumerate((0, YEARS.index(2019), len(YEARS) - 1)):
        ax = fig.add_subplot(grid[0, k])
        ax.imshow(false_colour([refl[i, b] for b in order], limits))
        ax.axis("off")
        map_marks(ax, colour="#ffffff")
        ax.set_title(f"January {YEARS[i]}, false colour (vegetation red): {100 * area[i] / land:.0f} % vegetation",
                     fontsize=11)
    ax = fig.add_subplot(grid[1, 0])
    img = background(valid, veg[0])
    for (a, b), colour in zip(PERIODS, PERIOD_COLOURS):
        img[(lost_in >= a) & (lost_in <= b)] = matplotlib.colors.to_rgb(colour)
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    map_marks(ax)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#898781", label=f"lost {a}–{b}") for (a, b), c in
                       zip(PERIODS, PERIOD_COLOURS)], loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
              fontsize=9, frameon=False)
    ax.set_title(f"Year of lasting loss: {(lost_in > 0).sum() * PIXEL_KM2:.1f} km² in all", fontsize=11)

    ax = fig.add_subplot(grid[1, 1:])
    share = 100 * area / land
    ax.plot(YEARS, share, "o-", color="#2e7d32", lw=2, ms=7, label="airport box (Ulwe)")
    ax.plot(YEARS, main_share, "o--", color="#898781", lw=1.5, ms=5, label="main box (Powai–Aarey–Kanjurmarg)")
    years = np.array(YEARS, dtype=float)
    ax.plot(years, 100 * (trend[0] * years + trend[1]) / land, ":", color="#2e7d32", lw=1.2,
            label=f"trend, airport box: {trend[0]:+.2f} km² per year")
    ax.set_xticks(YEARS)
    ax.set_ylabel("vegetation, % of the land area")
    right = ax.twinx()
    right.bar(YEARS, open_water, color="#2a78d6", alpha=0.18, width=0.6)
    right.set_ylabel("open water in the airport box (km², blue bars)")
    right.spines[["top"]].set_visible(False)
    ax.set_zorder(right.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Same scenes, same steps, same threshold: vegetation every January", fontsize=12)
    tidy(ax)
    fig.suptitle("Second area: the new Navi Mumbai airport at Ulwe (73.02–73.10° E, 18.96–19.02° N)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "day5_airport.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def pixel_edges(mask, profile):
    tf = profile["transform"]
    ends = []
    for r, c in np.argwhere(mask):
        for dr, dc, side in [(-1, 0, ((0, 0), (0, 1))), (1, 0, ((1, 0), (1, 1))), (0, -1, ((0, 0), (1, 0))),
                             (0, 1, ((0, 1), (1, 1)))]:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]):
                ends += [(r + side[0][0], c + side[0][1]), (r + side[1][0], c + side[1][1])]
    ends = np.array(ends, dtype=float)
    lon, lat = transform(UTM, "EPSG:4326", tf.c + ends[:, 1] * tf.a, tf.f + ends[:, 0] * tf.e)
    return np.array(lon).reshape(-1, 2), np.array(lat).reshape(-1, 2)


def draw_on_photo(ax, lons, lats, photo, image, colour, lw=1.4):
    x, y = on_photo(lons, lats, photo, image.shape[1], image.shape[0])
    ax.plot(x, y, color=colour, lw=lw)


def photo_frame(ax, colour, width):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(colour)
        spine.set_linewidth(width)


def show_point_photos(scored, photos, profile):
    names = {1: "v", 0: "n"}
    for year in POINT_YEARS:
        s = scored[year]
        series = {}
        for row in photos:
            if row["kind"] == "point" and row["year"] == str(year):
                series.setdefault(row["name"], []).append(row)
        if not series:
            continue
        per_point = max(len(v) for v in series.values())
        across = 4 if per_point > 1 else 5
        per_sheet = across * 5
        for sheet in range(int(np.ceil(len(s["points"]) / per_sheet))):
            fig, axes = plt.subplots(5, across * per_point, figsize=(2.05 * across * per_point, 11.4))
            for ax in axes.flat:
                ax.axis("off")
            for k, p in enumerate(s["points"][sheet * per_sheet:(sheet + 1) * per_sheet]):
                i = sheet * per_sheet + k
                corners = pixel_square(int(p["row"]), int(p["col"]), profile)
                lats, lons = [c[0] for c in corners], [c[1] for c in corners]
                done = s["done"][i]
                colour = "#c9c6bc" if not done else (GAIN if s["truth"][i] == s["ours"][i] else LOSS)
                for j, photo in enumerate(sorted(series.get(p["point"], []), key=lambda r: r["capture_date"])):
                    ax = axes.flat[(k // across) * across * per_point + (k % across) * per_point + j]
                    ax.axis("on")
                    image = plt.imread(DATA / "photos" / photo["file"])
                    ax.imshow(image)
                    draw_on_photo(ax, lons, lats, photo, image, "#ffd400")
                    photo_frame(ax, colour, 3 if done else 1)
                    title = p["point"] if j == 0 else ""
                    if j == 0 and done:
                        title += f"  photo {names[s['truth'][i]]} / map {names[s['ours'][i]]}"
                    ax.set_title(title, fontsize=9, loc="left")
                    ax.text(0.02, 0.03, photo["capture_date"], transform=ax.transAxes, fontsize=7.5, color="white",
                            path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
            labelled = s["done"].sum()
            fig.suptitle(f"Reference points {year} (sheet {sheet + 1}): the 30 m Landsat pixel in yellow" +
                         ("; frame blue = photo agrees with our map, red = it does not" if labelled else
                          "; not labelled yet") +
                         "\nPhotos: Esri World Imagery Wayback (WorldView, GeoEye: Maxar / Vantor; Pleiades: CNES / "
                         "Airbus); 250 x 250 pixels of 0.6 m", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(OUT / f"day5_photos_{year}_{sheet + 1}.png", dpi=100)
            plt.close(fig)


def pick_photos(series, year, log):
    start, end = log[year - 1]["date"], log[year]["date"]
    before = [r for r in series if r["capture_date"] <= start]
    after = [r for r in series if r["capture_date"] >= end] or [r for r in series if r["capture_date"] > start]
    chosen = [series[0]] + before[-1:] + after[:1] + [series[-1]]
    picked = []
    for r in chosen:
        if r not in picked:
            picked.append(r)
    return sorted(picked, key=lambda r: r["capture_date"])[:4]


def show_site_photos(checks, photos, c, profile, log):
    series = {}
    for row in photos:
        if row["kind"] == "site":
            series.setdefault(row["name"], []).append(row)
    if not series:
        return
    for rows in series.values():
        rows.sort(key=lambda r: r["capture_date"])
    outlines = {}
    for row in csv.DictReader(open(DATA / "sites.csv")):
        outlines.setdefault(row["site"], []).append((float(row["lon"]), float(row["lat"])))

    named = [n for n in ("Aarey Metro depot", "Kanjur landfill") if n in series]
    if named:
        across = max(len(series[n]) for n in named)
        fig, axes = plt.subplots(len(named), across, figsize=(2.3 * across, 2.75 * len(named) + 0.6))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")
        for k, name in enumerate(named):
            lons, lats = zip(*outlines[name])
            for j, photo in enumerate(series[name]):
                ax = axes[k, j]
                ax.axis("on")
                image = plt.imread(DATA / "photos" / photo["file"])
                ax.imshow(image)
                draw_on_photo(ax, lons, lats, photo, image, "#ffd400", lw=1.2)
                photo_frame(ax, "#c9c6bc", 1)
                ax.set_title(photo["capture_date"], fontsize=9)
            axes[k, 0].set_ylabel(name, fontsize=10)
        fig.suptitle("Named sites in the high-resolution photo archive (OpenStreetMap outline in yellow)\n"
                     "Photos: Esri World Imagery Wayback (Maxar / Vantor, CNES / Airbus); title = date the photo was taken",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(OUT / "day5_photos_sites.png", dpi=110)
        plt.close(fig)

    patches_ = [r for r in checks if r["site"].startswith("Loss patch") and r["site"] in series]
    fig, axes = plt.subplots(len(patches_), 4, figsize=(10.4, 2.55 * len(patches_) + 0.6))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, check in enumerate(patches_):
        number = int(check["site"].split()[-1])
        spot = c["spots"][number - 1]
        lons, lats = pixel_edges(c["label"] == spot["patch"], profile)
        chosen = pick_photos(series[check["site"]], spot["year"], log)
        for j, photo in enumerate(chosen):
            ax = axes[k, j]
            ax.axis("on")
            image = plt.imread(DATA / "photos" / photo["file"])
            ax.imshow(image)
            for a, b in zip(lons, lats):
                draw_on_photo(ax, a, b, photo, image, "#ffd400", lw=1.1)
            photo_frame(ax, "#c9c6bc", 1)
            ax.set_title(photo["capture_date"], fontsize=9)
        axes[k, 0].set_ylabel(f"Patch {number}: lost {spot['year']}\n{spot['pixels'] * PIXEL_KM2 * 100:.1f} ha, "
                              f"{'certain' if spot['certain'] else 'not yet certain'}", fontsize=8.5)
    fig.suptitle("Loss patches (outline in yellow): first photo, last photo before the loss, first photo after it, "
                 "latest photo\nPhotos: Esri World Imagery Wayback (Maxar / Vantor, CNES / Airbus); title = date taken",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "day5_photos_patches.png", dpi=110)
    plt.close(fig)


def image_sizes():
    rows = []
    for box, folder in [("main box", DATA), ("airport box", DATA / "airport")]:
        for year in YEARS:
            with rasterio.open(folder / f"{year}.tif") as src:
                tf, w, h = src.transform, src.width, src.height
                left, top = tf.c, tf.f
                right, bottom = left + w * tf.a, top + h * tf.e
                lon, lat = transform(UTM, "EPSG:4326", [left, right, left, right], [top, top, bottom, bottom])
                rows.append([box, year, w, h, src.count, src.dtypes[0], f"EPSG:{src.crs.to_epsg()}", tf.a, -tf.e,
                             left, top, right, bottom, round(w * h * tf.a * -tf.e / 1e6, 4)] +
                            [f"{la:.5f} N {lo:.5f} E" for lo, la in zip(lon, lat)])
    return rows


def ground_control(stack, base, profile):
    half, reach = 4, 3
    points = pick_gcps(stack, base, half, reach, apart=20)
    tf = profile["transform"]
    rows = []
    for g, (r, c) in enumerate(points, start=1):
        east, north = tf.c + (c + 0.5) * tf.a, tf.f + (r + 0.5) * tf.e
        (lon,), (lat,) = transform(UTM, "EPSG:4326", [east], [north])
        for k, year in enumerate(YEARS):
            dr, dc, corr = best_match(stack[base], stack[k], r, c, half, reach)
            rows.append([f"G{g}", r, c, round(east, 1), round(north, 1), round(lat, 6), round(lon, 6), year,
                         round(dr, 3), round(dc, 3), round(30 * np.hypot(dr, dc), 1), round(corr, 3)])
    return points, rows


def photo_series_control(photos):
    half, reach = 15, 25
    series = {}
    for p in photos:
        if p["kind"] == "site":
            series.setdefault(p["name"], []).append(p)
    rows = []
    for name, found in series.items():
        found.sort(key=lambda p: p["capture_date"])
        stack = np.array([plt.imread(DATA / "photos" / p["file"])[:, :, :3].astype(float).mean(axis=2) for p in found])
        step = float(found[0]["metres_per_pixel"])
        for k in range(1, len(found)):
            for g, (r, c) in enumerate(pick_gcps(stack[k - 1:k], 0, half, reach, apart=60), start=1):
                dr, dc, corr = best_match(stack[k - 1], stack[k], r, c, half, reach)
                rows.append([name, found[k - 1]["capture_date"], found[k]["capture_date"], f"G{g}", r, c,
                             round(dc * step, 2), round(-dr * step, 2), round(np.hypot(dr, dc) * step, 2),
                             round(corr, 3), "yes" if corr >= 0.6 else "no (ground changed?)"])
    return rows


def check_geometry(d):
    sizes = image_sizes()
    main = [r for r in sizes if r[0] == "main box"]
    air = [r for r in sizes if r[0] == "airport box"]
    same_main = all(r[2:] == main[0][2:] for r in main)
    same_air = all(r[2:] == air[0][2:] for r in air)
    save_csv(OUT / "day7_image_sizes.csv", ["box", "year", "width_px", "height_px", "bands", "type", "projection",
                                            "pixel_width_m", "pixel_height_m", "left_m", "top_m", "right_m", "bottom_m",
                                            "area_km2", "top_left", "top_right", "bottom_left", "bottom_right"], sizes)
    show_footprints(d["refl"], sizes)

    base = YEARS.index(BASE)
    points, gcp_rows = ground_control(d["refl"][:, 3], base, d["profile"])
    refl_air, _, _, profile_air = load_scenes(DATA / "airport")
    points_air, gcp_air = ground_control(refl_air[:, 3], base, profile_air)
    save_csv(OUT / "day7_gcps.csv", ["box", "gcp", "row", "col", "easting_m", "northing_m", "lat", "lon", "year",
                                     "shift_rows", "shift_cols", "shift_m", "correlation"],
             [["main box"] + r for r in gcp_rows] + [["airport box"] + r for r in gcp_air])

    photos = read_photos()
    shapes, ground = {}, {}
    for p in photos:
        key = (p["kind"], p["year"] if p["kind"] == "point" else p["name"])
        shapes.setdefault(key, set()).add((p["pixels"], p["metres_per_pixel"]))
        ground.setdefault((p["kind"], p["year"], p["name"]), set()).add((p["left"], p["top"], p["pixels"],
                                                                        p["metres_per_pixel"]))
    same_ground = all(len(v) == 1 for v in ground.values())
    photo_rows = photo_series_control(photos)
    save_csv(OUT / "day7_photo_gcps.csv", ["site", "earlier_photo", "photo", "gcp", "row", "col", "shift_east_m",
                                           "shift_north_m", "shift_m", "correlation", "matched"], photo_rows)
    show_gcps(d["refl"], points, gcp_rows, base)
    calls = library_calls()
    save_csv(OUT / "day7_library_calls.csv", ["library", "call", "times", "where (file: line numbers)"], calls)

    def rms(rows, year):
        shifts = [r[10] for r in rows if r[7] == year]
        return np.sqrt(np.mean(np.square(shifts)))
    worst = max(rms(gcp_rows, y) for y in YEARS)
    worst_air = max(rms(gcp_air, y) for y in YEARS)
    print(f"main box: {len(main)} images, all {main[0][2]} x {main[0][3]} pixels of {main[0][7]:g} m, {main[0][4]} "
          f"bands, corners ({main[0][9]:.0f}, {main[0][10]:.0f}) to ({main[0][11]:.0f}, {main[0][12]:.0f}) {main[0][6]}, "
          f"{main[0][13]:g} km2 - identical: {same_main} | airport box: {len(air)} images, {air[0][2]} x {air[0][3]} "
          f"pixels, {air[0][13]:g} km2 - identical: {same_air}")
    print(f"every photo of the same point or site covers the same ground (same corner, pixels and pixel size): "
          f"{same_ground} ({len(ground)} points and sites)")
    largest = max(r[10] for r in gcp_rows)
    largest_air = max(r[10] for r in gcp_air)
    print(f"ground control points: 5 per image; largest shift {largest:.1f} m ({largest / 30:.2f} pixel) in the main "
          f"box, {largest_air:.1f} m ({largest_air / 30:.2f} pixel) in the airport box - every point within one "
          f"pixel: {largest < 30 and largest_air < 30} | RMS error per image at most {worst:.1f} m (main box), "
          f"{worst_air:.1f} m (airport box); lowest correlation {min(r[11] for r in gcp_rows + gcp_air):.2f}")
    print("photos: " + "; ".join(f"{k[1]}: {len(v)} size(s) " +
                                 ", ".join(f"{px} x {px} px of {float(m):g} m" for px, m in sorted(v))
                                 for k, v in sorted(shapes.items())))
    print(f"library calls in the three code files: {len(calls)} different calls, from " +
          ", ".join(sorted({c[0] for c in calls})) + " (output/day7_library_calls.csv)")
    matched = [r for r in photo_rows if r[10] == "yes"]
    if matched:
        shifts = np.array([r[8] for r in matched])
        print(f"photo series: {len(photo_rows)} control-point matches, {len(matched)} found again; shift between "
              f"photos of the same place: median {np.median(shifts):.1f} m, 90 % within {np.percentile(shifts, 90):.1f} m")


def library_calls():
    import ast
    libraries = {"np": "numpy", "numpy": "numpy", "plt": "matplotlib", "matplotlib": "matplotlib",
                 "rasterio": "rasterio", "transform": "rasterio", "from_bounds": "rasterio", "Image": "Pillow",
                 "ImageGrab": "Pillow", "Patch": "matplotlib", "Line2D": "matplotlib", "patheffects": "matplotlib",
                 "to_rgb": "matplotlib", "csv": "csv", "json": "json", "urllib": "urllib", "tk": "tkinter",
                 "ttk": "tkinter", "messagebox": "tkinter", "ScrolledText": "tkinter", "FigureCanvasTkAgg": "matplotlib",
                 "NavigationToolbar2Tk": "matplotlib", "Figure": "matplotlib", "LineCollection": "matplotlib",
                 "ListedColormap": "matplotlib", "BoundaryNorm": "matplotlib", "ScalarMappable": "matplotlib",
                 "Rectangle": "matplotlib", "ThreadPoolExecutor": "concurrent.futures", "subprocess": "subprocess",
                 "webbrowser": "webbrowser", "threading": "threading", "queue": "queue"}
    found = {}
    for name in ("download.py", "analysis.py", "app.py"):
        path = HERE / name
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            parts, f = [], node.func
            while isinstance(f, ast.Attribute):
                parts.append(f.attr)
                f = f.value
            if isinstance(f, ast.Name) and f.id in libraries:
                call = ".".join([f.id] + parts[::-1])
                found.setdefault((libraries[f.id], call), {}).setdefault(name, []).append(node.lineno)
    return [[lib, call, sum(len(v) for v in where.values()),
             "; ".join(f"{n}: " + ", ".join(map(str, sorted(lines))) for n, lines in sorted(where.items()))]
            for (lib, call), where in sorted(found.items())]


def show_footprints(refl, sizes):
    order = [3, 2, 1]
    valid = np.isfinite(refl[-1, 3])
    limits = [np.percentile(refl[:, b][:, valid], [2, 98]) for b in order]
    main = [r for r in sizes if r[0] == "main box"]
    left, top, right, bottom = main[0][9], main[0][10], main[0][11], main[0][12]
    fig, ax = plt.subplots(figsize=(12, 8.4))
    ax.imshow(false_colour([refl[-1, b] for b in order], limits), extent=(left / 1000, right / 1000, bottom / 1000,
                                                                         top / 1000))
    colours = matplotlib.colormaps["viridis"](np.linspace(0, 1, len(main)))
    for i, (r, colour) in enumerate(zip(main, colours)):
        xs = np.array([r[9], r[11], r[11], r[9], r[9]]) / 1000
        ys = np.array([r[10], r[10], r[12], r[12], r[10]]) / 1000
        ax.plot(xs, ys, color=colour, lw=9 - 0.6 * i, solid_joinstyle="miter", label=str(r[1]))
    for (x, y), corner, ha, va in [((left, top), main[0][14], "left", "bottom"), ((right, top), main[0][15], "right",
                                                                                  "bottom"),
                                   ((left, bottom), main[0][16], "left", "top"), ((right, bottom), main[0][17], "right",
                                                                                  "top")]:
        ax.annotate(f"E {x:,.0f} m, N {y:,.0f} m\n{corner}", (x / 1000, y / 1000),
                    xytext=(0, 10 if va == "bottom" else -10), textcoords="offset points", ha=ha, va=va, fontsize=9)
    ax.set_xlim(left / 1000 - 0.6, right / 1000 + 0.6)
    ax.set_ylim(bottom / 1000 - 1.1, top / 1000 + 1.1)
    ax.set_xlabel("easting (km, UTM zone 43N)")
    ax.set_ylabel("northing (km)")
    ax.legend(title="outline of the image of", fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"The ground covered by each of the 13 images: all outlines lie on top of each other\n"
                 f"{main[0][2]} x {main[0][3]} pixels of {main[0][7]:g} m = {(right - left) / 1000:.1f} km x "
                 f"{(top - bottom) / 1000:.1f} km = {main[0][13]:g} km² in every year", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "day7_footprints.png", dpi=110)
    plt.close(fig)


def show_gcps(refl, points, rows, base):
    order = [3, 2, 1]
    valid = np.isfinite(refl[base, 3])
    limits = [np.percentile(refl[:, b][:, valid], [2, 98]) for b in order]
    fig = plt.figure(figsize=(18, 12.5))
    grid = fig.add_gridspec(len(points) + 1, len(YEARS) + 1, height_ratios=[3.2] + [1] * len(points))
    ax = fig.add_subplot(grid[0, 3:11])
    ax.imshow(false_colour([refl[base, b] for b in order], limits))
    for g, (r, c) in enumerate(points, start=1):
        ax.plot(c, r, "+", color="yellow", ms=16, mew=2)
        ax.annotate(f"G{g}", (c, r), xytext=(8, 6), textcoords="offset points", color="yellow", fontsize=12,
                    weight="bold", path_effects=[patheffects.withStroke(linewidth=2.5, foreground="black")])
    ax.axis("off")
    map_marks(ax, colour="#ffffff")
    ax.set_title(f"Five ground control points (false colour, January {YEARS[base]})", fontsize=13)
    for g, (r, c) in enumerate(points, start=1):
        label = fig.add_subplot(grid[g, 0])
        label.axis("off")
        label.text(0.5, 0.5, f"G{g}\nrow {r}\ncol {c}", ha="center", va="center", fontsize=10)
        for k, year in enumerate(YEARS):
            ax = fig.add_subplot(grid[g, k + 1])
            win = (slice(r - 10, r + 11), slice(c - 10, c + 11))
            ax.imshow(false_colour([refl[k, b][win] for b in order], limits), interpolation="nearest")
            ax.plot([10, 10], [6, 14], color="yellow", lw=1)
            ax.plot([6, 14], [10, 10], color="yellow", lw=1)
            shift = next(x for x in rows if x[0] == f"G{g}" and x[7] == year)
            ax.set_title(f"{year}" if g == 1 else "", fontsize=9)
            ax.set_xlabel(f"{shift[10]:.0f} m", fontsize=8, labelpad=1)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("The same ground under the cross in all 13 images: 21 x 21 pixels (630 m) around each control "
                 "point; below each window the shift found by matching", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "day7_gcps.png", dpi=110)
    plt.close(fig)


class Log:
    def __init__(self, path):
        self.file, self.screen = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)

    def flush(self):
        self.screen.flush()
        self.file.flush()


def main():
    OUT.mkdir(exist_ok=True)
    sys.stdout = Log(OUT / "run_log.txt")
    start = time.time()
    print("--- Day 2: prepare ---")
    d = prepare()
    print("\n--- Day 3: vegetation maps ---")
    v = map_vegetation(d)
    print("\n--- Day 4: changes ---")
    c = find_changes(d, v)
    print("\n--- Day 5: accuracy ---")
    check_accuracy(d, v, c)
    print("\n--- Day 5: principal components ---")
    find_components(d, v, c)
    print("\n--- Day 5: second area ---")
    second_area(d, v)
    print("\n--- Day 7: image sizes and ground control points ---")
    check_geometry(d)
    print(f"\nall done in {time.time() - start:.0f} s - figures and tables are in the output folder")
    sys.stdout = sys.stdout.screen


if __name__ == "__main__":
    main()
