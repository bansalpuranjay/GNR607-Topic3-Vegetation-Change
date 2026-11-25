import csv
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib
import numpy as np
import rasterio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from rasterio.warp import transform

import analysis
from analysis import DATA, OUT, PIXEL_KM2, YEARS

HERE = analysis.HERE
LAYERS = ["False colour", "NDVI", "Vegetation map"]
CLASS_NAMES = ["water or cloud (not used)", "no vegetation", "sparse vegetation", "dense vegetation"]
CHANGES = ["Loss and gain between two years", "Year of lasting loss", "Year of lasting gain",
           "NDVI difference (2-sigma rule)"]
SITE_ORDER = ["Aarey Metro depot", "Kanjur landfill"] + [f"Loss patch {i}" for i in range(1, 11)]
LABEL_NAMES = {"v": "vegetation", "n": "no vegetation", "?": "can't tell"}


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_bands(folder):
    dn = []
    for year in YEARS:
        with rasterio.open(folder / f"{year}.tif") as src:
            dn.append(src.read([2, 3, 4]))
    return analysis.to_reflectance(np.stack(dn))


def stretch(refl, valid):
    return [np.percentile(refl[:, b][:, valid], [2, 98]) for b in (2, 1, 0)]


def false_colour(refl, limits, k):
    img = analysis.false_colour([refl[k, 2], refl[k, 1], refl[k, 0]], limits)
    return np.nan_to_num(img)


def load_results():
    r = {}
    with rasterio.open(DATA / "ndvi.tif") as src:
        r["ndvi"], r["profile"] = src.read(), src.profile
    with rasterio.open(DATA / "masks.tif") as src:
        r["valid"] = src.read(2).astype(bool)
    with rasterio.open(DATA / "vegetation.tif") as src:
        r["classes"] = src.read()
    with rasterio.open(DATA / "change.tif") as src:
        r["lost_in"], r["gained_in"], r["patch"] = src.read(1), src.read(2), src.read(3)
    r["veg"] = r["classes"] >= 2
    r["refl"] = read_bands(DATA)
    r["limits"] = stretch(r["refl"], r["valid"])
    r["log"] = {int(row["year"]): row for row in read_csv(DATA / "scene_log.csv")}
    r["land"] = r["valid"].sum() * PIXEL_KM2
    r["area"] = r["veg"].sum(axis=(1, 2)) * PIXEL_KM2
    r["pooled"] = r["ndvi"][:, r["valid"]].ravel()
    r["t"], _ = analysis.otsu(r["pooled"])
    for name in ["day3_vegetation_area", "day4_change_by_interval", "day5_accuracy", "day5_threshold_vs_kappa",
                 "day5_airport_area"]:
        r[name] = read_csv(OUT / f"{name}.csv")
    r["photos"] = analysis.read_photos()
    r["checks"] = {row["site"]: row for row in read_csv(DATA / "hotspot_checks.csv")}
    r["outlines"] = {}
    for row in read_csv(DATA / "sites.csv"):
        r["outlines"].setdefault(row["site"], []).append((float(row["lat"]), float(row["lon"])))
    airport = DATA / "airport"
    if (airport / f"{YEARS[-1]}.tif").exists() and (DATA / "airport_vegetation.tif").exists():
        r["air_refl"] = read_bands(airport)
        with rasterio.open(DATA / "airport_vegetation.tif") as src:
            bands = src.read()
        r["air_classes"], r["air_lost_in"] = bands[:len(YEARS)], bands[len(YEARS)]
        r["air_valid"] = r["air_classes"][0] > 0
        r["air_limits"] = stretch(r["air_refl"], r["air_valid"])
    return r


def key_numbers(r):
    lines = []
    rows = r["day3_vegetation_area"]
    if rows:
        years = np.array([int(x["year"]) for x in rows], dtype=float)
        area = np.array([float(x["vegetation_km2"]) for x in rows])
        share = np.array([float(x["vegetation_pct"]) for x in rows])
        trend = analysis.fit_line(years, area, np.ones(len(years)))
        ups = sum(x["direction"] == "up" for x in rows[1:])
        downs = sum(x["direction"] == "down" for x in rows[1:])
        lines += [f"Vegetation (NDVI above T = {r['t']:.2f}, one threshold for all years): "
                  f"{area.min():.1f} to {area.max():.1f} km², i.e. {share.min():.1f} to {share.max():.1f} % of the "
                  f"{float(rows[0]['land_km2']):.1f} km² of land in the box.",
                  f"It rose in {ups} intervals and fell in {downs}; straight-line trend {trend[0]:+.2f} km² per year "
                  f"(r² {trend[2]:.2f}).", ""]
    for row in r["day4_change_by_interval"]:
        if row["interval"].startswith("total 2016-2024"):
            lines += [f"Most of the yearly change does not last. Lasting change in 2016-2024: "
                      f"{float(row['lasting_loss_km2']):.2f} km² lost, "
                      f"{float(row['lasting_gain_km2']):.2f} km² gained.", ""]
    for row in r["day5_accuracy"]:
        if row["comparison"].endswith("all pixels") or "weighted" in row["comparison"]:
            lines.append(f"{row['comparison']}: overall {row['overall_pct']} %, kappa {row['kappa']}")
    air = r["day5_airport_area"]
    if air:
        lines += ["", f"Second area (Navi Mumbai airport): vegetation {float(air[0]['vegetation_pct']):.1f} % in "
                      f"{air[0]['year']}, {float(air[-1]['vegetation_pct']):.1f} % in {air[-1]['year']}."]
    log = r["log"]
    if log:
        times = sorted({row["time_ist"] for row in log.values()})
        days = [int(row["day_of_year"]) for row in log.values()]
        lines += ["", "The images", f"Landsat 8, OLI sensor, Collection 2 Level-2 surface reflectance, Tier 1, WRS-2 "
                      f"path 148 / row 047. One image every January from {min(log)} to {max(log)} "
                      f"(between {min(days)} and {max(days)} January), all taken at {times[0]}-{times[-1]} IST in "
                      f"daylight. 30 m pixels; the study box is 390 x 300 pixels (11.7 km x 9.0 km)."]
    return "\n".join(lines)


def add_figure(parent, toolbar=True):
    fig = Figure(figsize=(9, 6), dpi=100, layout="constrained")
    canvas = FigureCanvasTkAgg(fig, master=parent)
    bar = None
    if toolbar:
        bar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        bar.update()
        bar.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    return fig, canvas, bar


def text_box(parent, height):
    box = tk.Text(parent, height=height, width=40, wrap="word", relief="flat", background="#f4f3ef",
                  font=("Segoe UI", 9), padx=6, pady=4)
    box.pack(fill="both", expand=False, pady=(4, 8))
    box.configure(state="disabled")
    return box


def write(box, text):
    box.configure(state="normal")
    box.delete("1.0", "end")
    box.insert("1.0", text)
    box.configure(state="disabled")


def heading(parent, text):
    ttk.Label(parent, text=text, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 2))


def year_colours(first, last):
    years = list(range(first, last + 1))
    cmap = ListedColormap([matplotlib.colormaps["plasma"](x) for x in np.linspace(0.05, 0.9, len(years))])
    norm = BoundaryNorm(np.arange(first - 0.5, last + 1.5), cmap.N)
    return years, cmap, norm


def certain_year(year):
    return YEARS[1] < year <= YEARS[-analysis.CONFIRM]


def draw_main_graph(ax, rows):
    years = np.array([int(x["year"]) for x in rows])
    area = np.array([float(x["vegetation_km2"]) for x in rows])
    land = float(rows[0]["land_km2"])
    trend = analysis.fit_line(years.astype(float), area, np.ones(len(years)))
    ax.plot(years, trend[0] * years + trend[1], "--", color="#898781", lw=1.2)
    ax.plot(years, area, "o-", color="#2e7d32", lw=2, ms=6, zorder=3)
    gap = 0.08 * np.ptp(area)
    for i, row in enumerate(rows[1:]):
        pct = float(row["change_pct"])
        up = pct > 0
        colour = analysis.UP if up else analysis.DOWN
        x, y = years[i] + 0.5, (area[i] + area[i + 1]) / 2 + (gap if up else -gap)
        steady = row["same_at_all_thresholds"] == "yes"
        ax.plot(x, y, "^" if up else "v", ms=9, color=colour, mfc=colour if steady else "white", mew=1.5)
        ax.annotate(f"{pct:+.1f} %".replace("-", "−"), (x, y), xytext=(0, 8 if up else -14),
                    textcoords="offset points", ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(years)
    ax.set_ylim(area.min() - 4 * gap, area.max() + 5 * gap)
    ax.set_ylabel("vegetation area (km²)")
    right = ax.secondary_yaxis("right", functions=(lambda a: 100 * a / land, lambda p: p * land / 100))
    right.set_ylabel(f"% of the {land:.1f} km² of land")
    handles = [Line2D([], [], color="#2e7d32", marker="o", lw=2, label="vegetation"),
               Line2D([], [], color="#898781", ls="--", lw=1.2, label=f"trend {trend[0]:+.2f} km² per year"),
               Line2D([], [], color=analysis.UP, marker="^", ls="none", ms=8, label="rise from the year before"),
               Line2D([], [], color=analysis.DOWN, marker="v", ls="none", ms=8, label="fall from the year before"),
               Line2D([], [], color="#52514e", marker="^", mfc="white", ls="none", ms=8,
                      label="hollow: direction depends on the threshold")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Vegetation cover around IIT Bombay, every January from 2014 to 2026", fontsize=12)
    analysis.tidy(ax)


def edge_lines(mask, profile, photo, width, height):
    tf = profile["transform"]
    xs, ys = transform("EPSG:4326", analysis.UTM, [float(photo["west"]), float(photo["east"])],
                       [float(photo["north"]), float(photo["south"])])
    c0, c1 = int((xs[0] - tf.c) // tf.a) - 1, int((xs[1] - tf.c) // tf.a) + 2
    r0, r1 = int((ys[0] - tf.f) // tf.e) - 1, int((ys[1] - tf.f) // tf.e) + 2
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, mask.shape[1]), min(r1, mask.shape[0])
    if c1 <= c0 or r1 <= r0:
        return []
    padded = np.pad(mask[r0:r1, c0:c1], 1)
    ends = []
    inner = padded[1:-1, 1:-1]
    sides = [(-1, 0, (0, 0), (0, 1)), (1, 0, (1, 0), (1, 1)), (0, -1, (0, 0), (1, 0)), (0, 1, (0, 1), (1, 1))]
    for dr, dc, a, b in sides:
        open_side = inner & ~padded[1 + dr:padded.shape[0] - 1 + dr, 1 + dc:padded.shape[1] - 1 + dc]
        rr, cc = np.nonzero(open_side)
        for r, c in zip(rr + r0, cc + c0):
            ends.append((r + a[0], c + a[1], r + b[0], c + b[1]))
    if not ends:
        return []
    ends = np.array(ends, dtype=float)
    lines = []
    for k in (0, 2):
        lon, lat = transform(analysis.UTM, "EPSG:4326", list(tf.c + ends[:, k + 1] * tf.a),
                             list(tf.f + ends[:, k] * tf.e))
        lines.append(analysis.on_photo(np.array(lon), np.array(lat), photo, width, height))
    return [[(x0, y0), (x1, y1)] for x0, y0, x1, y1 in zip(lines[0][0], lines[0][1], lines[1][0], lines[1][1])]


class App(tk.Tk):
    def __init__(self):
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass
        super().__init__()
        self.title("GNR 607 Topic 3 - vegetation cover around IIT Bombay, 2014-2026")
        width, height = min(1400, self.winfo_screenwidth() - 40), min(900, self.winfo_screenheight() - 80)
        self.geometry(f"{width}x{height}+20+10")
        self.minsize(1000, 640)
        self.points_dir = DATA
        self.process = None
        self.output = queue.Queue()
        self.photo_cache = {}
        self.fcc_cache = {}
        self.pixel = None

        self.status = tk.StringVar(value="Loading the results ...")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken", padding=(6, 2)).pack(side="bottom",
                                                                                                    fill="x")
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.update()
        self.r = load_results()
        self.make_overview()
        self.make_images()
        self.make_threshold()
        self.make_change()
        self.make_accuracy()
        self.make_airport()
        self.make_run()
        self.bind("<KeyPress>", self.key_pressed)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.tabs.bind("<<NotebookTabChanged>>", lambda _: self.draw_current())
        self.draw_everything()
        self.status.set(f"Results loaded from {DATA} and {OUT}.  Otsu threshold T = {self.r['t']:.3f}")

    def page(self, title, panel_width=310):
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text=f"  {title}  ")
        left = ttk.Frame(frame, width=panel_width, padding=(10, 6))
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(frame)
        right.pack(side="left", fill="both", expand=True)
        return left, right

    def draw_everything(self):
        self.drawn = set()
        self.draw_current()

    def draw_current(self):
        tab = self.tabs.index("current")
        if tab in self.drawn:
            return
        self.drawn.add(tab)
        steps = [self.draw_overview, self.draw_image, self.recompute_threshold, self.draw_change, self.draw_accuracy,
                 self.draw_airport, lambda: None]
        steps[tab]()

    def make_overview(self):
        left, right = self.page("Overview", panel_width=380)
        heading(left, "Key results")
        self.summary_box = text_box(left, 30)
        self.overview_fig, self.overview_canvas, _ = add_figure(right, toolbar=False)

    def draw_overview(self):
        write(self.summary_box, key_numbers(self.r))
        fig = self.overview_fig
        fig.clear()
        if self.r["day3_vegetation_area"]:
            draw_main_graph(fig.add_subplot(), self.r["day3_vegetation_area"])
        self.overview_canvas.draw_idle()

    def make_images(self):
        left, right = self.page("Images")
        heading(left, "Year")
        self.year_var = tk.IntVar(value=YEARS[-1])
        tk.Scale(left, from_=YEARS[0], to=YEARS[-1], orient="horizontal", variable=self.year_var, length=270,
                 command=lambda _: self.draw_image()).pack(anchor="w")
        heading(left, "Show")
        self.layer_var = tk.StringVar(value=LAYERS[0])
        for name in LAYERS:
            ttk.Radiobutton(left, text=name, value=name, variable=self.layer_var,
                            command=self.draw_image).pack(anchor="w")
        heading(left, "The image")
        self.scene_box = text_box(left, 9)
        heading(left, "The pixel you clicked")
        self.pixel_box = text_box(left, 12)
        self.images_fig, self.images_canvas, self.images_bar = add_figure(right)
        self.images_canvas.mpl_connect("button_press_event", self.map_clicked)

    def fcc(self, k):
        if k not in self.fcc_cache:
            self.fcc_cache[k] = false_colour(self.r["refl"], self.r["limits"], k)
        return self.fcc_cache[k]

    def draw_image(self):
        r = self.r
        year = self.year_var.get()
        k = YEARS.index(year)
        layer = self.layer_var.get()
        fig = self.images_fig
        fig.clear()
        grid = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
        ax = fig.add_subplot(grid[0, 0])
        self.map_ax = ax
        colour = "#0b0b0b"
        if layer == "False colour":
            ax.imshow(self.fcc(k), interpolation="nearest")
            colour = "#ffffff"
        elif layer == "NDVI":
            cmap = matplotlib.colormaps["RdYlGn"].copy()
            cmap.set_bad("#b9d6f2")
            im = ax.imshow(np.where(r["valid"], r["ndvi"][k], np.nan), cmap=cmap, vmin=-0.2, vmax=0.9,
                           interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.6, label="NDVI (after forest scaling)")
        else:
            ax.imshow(r["classes"][k], cmap=ListedColormap(analysis.VEG_COLOURS), vmin=-0.5, vmax=3.5,
                      interpolation="nearest")
            ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(analysis.VEG_COLOURS, CLASS_NAMES)],
                      loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False)
        ax.axis("off")
        analysis.map_marks(ax, colour=colour)
        log = r["log"][year]
        title = f"January {year} ({log['date']}), {layer if layer == 'NDVI' else layer.lower()}"
        if layer == LAYERS[2]:
            title += f"\nvegetation {100 * r['area'][k] / r['land']:.1f} % of the land"
        ax.set_title(title, fontsize=11)
        series = fig.add_subplot(grid[0, 1])
        self.draw_series(series)
        if self.pixel:
            row, col = self.pixel
            ax.plot(col, row, "s", ms=11, mfc="none", mec="yellow" if layer == LAYERS[0] else "#0b0b0b", mew=2)
        write(self.scene_box, f"Date: {log['date']} (day {log['day_of_year']} of the year)\n"
                              f"Time: {log['time_ist']} IST ({log['time_utc']} UTC)\n"
                              f"Sun: {log['sun_elevation']}° above the horizon, azimuth {log['sun_azimuth']}°\n"
                              f"Cloud in the box: {log['box_cloud_pct']} %   haze: {log['box_haze_pct']} %\n"
                              f"Landsat 8 OLI, Collection 2 Level-2\nProduct: {log['product_id']}")
        self.images_canvas.draw_idle()

    def draw_series(self, ax):
        r = self.r
        if not self.pixel:
            ax.text(0.5, 0.5, "Click a pixel on the map\nto see its NDVI in all 13 Januaries", ha="center",
                    va="center", fontsize=10, color="#52514e", transform=ax.transAxes)
            ax.axis("off")
            return
        row, col = self.pixel
        values = r["ndvi"][:, row, col]
        classes = r["classes"][:, row, col]
        ax.plot(YEARS, values, "-", color="#898781", lw=1)
        for year, value, c in zip(YEARS, values, classes):
            ax.plot(year, value, "o", ms=7, color=analysis.VEG_COLOURS[c], mec="#52514e")
        ax.axhline(r["t"], color="#0b0b0b", lw=1, ls="--")
        ax.annotate(f"T = {r['t']:.2f}", (YEARS[0], r["t"]), xytext=(0, 4), textcoords="offset points", fontsize=8)
        ax.set_ylim(-0.25, 1.0)
        ax.set_xticks(YEARS[::4])
        ax.set_ylabel("NDVI")
        ax.set_title(f"Pixel at row {row}, column {col}", fontsize=10)
        analysis.tidy(ax)

    def map_clicked(self, event):
        if event.inaxes is not getattr(self, "map_ax", None) or self.images_bar.mode:
            return
        col, row = int(np.floor(event.xdata + 0.5)), int(np.floor(event.ydata + 0.5))
        self.show_pixel(row, col)

    def show_pixel(self, row, col):
        r = self.r
        if not (0 <= row < r["valid"].shape[0] and 0 <= col < r["valid"].shape[1]):
            return
        self.pixel = (row, col)
        if "wc21" not in r:
            self.status.set("Reading WorldCover 2021 for the pixel details ...")
            self.update_idletasks()
            r["wc21"] = analysis.worldcover_on_grid(2021, r["profile"])
        tf = r["profile"]["transform"]
        lon, lat = transform(analysis.UTM, "EPSG:4326", [tf.c + (col + 0.5) * tf.a], [tf.f + (row + 0.5) * tf.e])
        lost, gained, patch = int(r["lost_in"][row, col]), int(r["gained_in"][row, col]), int(r["patch"][row, col])

        def when(year):
            return "never" if year == 0 else f"{year} ({'certain' if certain_year(year) else 'not yet certain'})"
        classes = r["classes"][:, row, col]
        years = ", ".join(f"{y}: {['-', 'none', 'sparse', 'dense'][c]}" for y, c in zip(YEARS, classes))
        write(self.pixel_box, f"Row {row}, column {col}: {lat[0]:.5f} N, {lon[0]:.5f} E\n"
                              f"WorldCover 2021: {analysis.WORLDCOVER.get(int(r['wc21'][row, col]), 'other')}\n"
                              f"Used in the analysis: {'yes' if r['valid'][row, col] else 'no (water or cloud)'}\n"
                              f"Lasting loss: {when(lost)}\nLasting gain: {when(gained)}\n"
                              f"Loss patch: {patch if patch else '-'}\n\n{years}")
        self.status.set(f"Pixel row {row}, column {col}")
        self.draw_image()

    def make_threshold(self):
        left, right = self.page("Threshold")
        heading(left, "NDVI threshold T")
        self.t_var = tk.DoubleVar(value=round(self.r["t"], 2))
        scale = tk.Scale(left, from_=0.20, to=0.70, resolution=0.01, orient="horizontal", variable=self.t_var,
                         length=270)
        scale.pack(anchor="w")
        scale.bind("<ButtonRelease-1>", lambda _: self.recompute_threshold())
        scale.bind("<KeyRelease>", lambda _: self.recompute_threshold())
        buttons = ttk.Frame(left)
        buttons.pack(anchor="w", pady=4)
        ttk.Button(buttons, text="Recompute", command=self.recompute_threshold).pack(side="left")
        ttk.Button(buttons, text="Back to Otsu's T", command=self.back_to_otsu).pack(side="left", padx=6)
        heading(left, "Map of the year")
        self.t_year = tk.StringVar(value=str(YEARS[-1]))
        box = ttk.Combobox(left, textvariable=self.t_year, values=[str(y) for y in YEARS], width=8, state="readonly")
        box.pack(anchor="w")
        box.bind("<<ComboboxSelected>>", lambda _: self.draw_threshold())
        heading(left, "What changes")
        self.t_box = text_box(left, 14)
        self.t_fig, self.t_canvas, _ = add_figure(right)

    def back_to_otsu(self):
        self.t_var.set(round(self.r["t"], 2))
        self.recompute_threshold()

    def recompute_threshold(self):
        r = self.r
        t = float(self.t_var.get())
        self.config(cursor="watch")
        self.update_idletasks()
        self.veg_t = analysis.vegetation(r["ndvi"], t, r["valid"])
        self.area_t = self.veg_t.sum(axis=(1, 2)) * PIXEL_KM2
        self.config(cursor="")
        if "counts" not in r:
            r["counts"] = analysis.histogram(r["pooled"], 200, -0.2, 1.0)
        self.draw_threshold()

    def draw_threshold(self):
        r = self.r
        t = float(self.t_var.get())
        k = YEARS.index(int(self.t_year.get()))
        land = r["land"]
        fig = self.t_fig
        fig.clear()
        grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])

        ax = fig.add_subplot(grid[0, 0])
        left = -0.2 + np.arange(200) * 1.2 / 200
        colour = np.where(left < t - 1e-9, analysis.VEG_COLOURS[1], analysis.VEG_COLOURS[2])
        ax.bar(left, r["counts"] / 1000, width=1.2 / 200, align="edge", color=colour)
        ax.axvline(r["t"], color="#0b0b0b", lw=1)
        ax.axvline(t, color="#2a78d6", lw=1.5, ls="--")
        ax.set_xlabel("NDVI (all 13 years, valid pixels)")
        ax.set_ylabel("pixels (thousands)")
        ax.set_title(f"Histogram: Otsu's T = {r['t']:.2f} (black),\nyour T = {t:.2f} (blue)", fontsize=10)
        analysis.tidy(ax)

        ax = fig.add_subplot(grid[1, 0])
        ax.plot(YEARS, 100 * r["area"] / land, "o-", color="#2e7d32", lw=2, ms=5, label=f"Otsu's T = {r['t']:.2f}")
        ax.plot(YEARS, 100 * self.area_t / land, "o--", color="#2a78d6", lw=2, ms=5, label=f"T = {t:.2f}")
        ax.set_xticks(YEARS[::2])
        ax.set_ylabel("vegetation, % of the land")
        ax.legend(fontsize=8)
        ax.set_title("Vegetation every January", fontsize=10)
        analysis.tidy(ax)

        ax = fig.add_subplot(grid[:, 1])
        img = analysis.background(r["valid"], r["veg"][k] & self.veg_t[k])
        img[self.veg_t[k] & ~r["veg"][k]] = to_rgb(analysis.GAIN)
        img[r["veg"][k] & ~self.veg_t[k]] = to_rgb(analysis.LOSS)
        ax.imshow(img, interpolation="nearest")
        ax.axis("off")
        analysis.map_marks(ax)
        ax.legend(handles=[Patch(facecolor=analysis.MAP_BACK[2], edgecolor="#898781", label="vegetation at both"),
                           Patch(color=analysis.GAIN, label=f"vegetation only at T = {t:.2f}"),
                           Patch(color=analysis.LOSS, label="vegetation only at Otsu's T"),
                           Patch(facecolor=analysis.MAP_BACK[1], edgecolor="#898781", label="no vegetation")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False)
        ax.set_title(f"January {YEARS[k]}: {100 * self.area_t[k] / land:.1f} % at T = {t:.2f},\n"
                     f"{100 * r['area'][k] / land:.1f} % at Otsu's T", fontsize=10)
        self.t_canvas.draw_idle()

        otsu_way, new_way = np.sign(np.diff(r["area"])), np.sign(np.diff(self.area_t))
        share = 100 * self.area_t / land
        write(self.t_box, f"At T = {t:.2f} vegetation is {share.min():.1f} to {share.max():.1f} % of the land "
                          f"(Otsu's T: {100 * r['area'].min() / land:.1f} to {100 * r['area'].max() / land:.1f} %).\n\n"
                          f"Rises: {(new_way > 0).sum()}, falls: {(new_way < 0).sum()}.\n"
                          f"{(new_way == otsu_way).sum()} of {len(otsu_way)} intervals go the same way as with "
                          f"Otsu's T.\n\nThe maps are rebuilt with our own vegetation(): NDVI above T, then the "
                          f"3 x 3 majority vote. The blue curve and map follow your T; the green ones are the "
                          f"results of the project.")

    def make_change(self):
        left, right = self.page("Change")
        heading(left, "From and to")
        row = ttk.Frame(left)
        row.pack(anchor="w")
        self.from_year = tk.StringVar(value=str(YEARS[0]))
        self.to_year = tk.StringVar(value=str(YEARS[-1]))
        for var, text in [(self.from_year, "from"), (self.to_year, "to")]:
            ttk.Label(row, text=text).pack(side="left", padx=(0, 4))
            box = ttk.Combobox(row, textvariable=var, values=[str(y) for y in YEARS], width=6, state="readonly")
            box.pack(side="left", padx=(0, 10))
            box.bind("<<ComboboxSelected>>", lambda _: self.draw_change())
        heading(left, "Show")
        self.change_var = tk.StringVar(value=CHANGES[0])
        for name in CHANGES:
            ttk.Radiobutton(left, text=name, value=name, variable=self.change_var,
                            command=self.draw_change).pack(anchor="w")
        heading(left, "Numbers")
        self.change_box = text_box(left, 26)
        self.change_fig, self.change_canvas, _ = add_figure(right)

    def draw_change(self):
        r = self.r
        a, b = YEARS.index(int(self.from_year.get())), YEARS.index(int(self.to_year.get()))
        mode = self.change_var.get()
        fig = self.change_fig
        fig.clear()
        valid, veg = r["valid"], r["veg"]
        if mode == CHANGES[0]:
            ax = fig.add_subplot()
            lost, won = veg[a] & ~veg[b], ~veg[a] & veg[b] & valid
            img = analysis.background(valid, veg[a] & veg[b])
            img[lost] = to_rgb(analysis.LOSS)
            img[won] = to_rgb(analysis.GAIN)
            ax.imshow(img, interpolation="nearest")
            ax.legend(handles=[Patch(color=analysis.LOSS, label="loss: vegetation -> none"),
                               Patch(color=analysis.GAIN, label="gain: none -> vegetation"),
                               Patch(facecolor=analysis.MAP_BACK[2], edgecolor="#898781", label="vegetation in both"),
                               Patch(facecolor=analysis.MAP_BACK[1], edgecolor="#898781", label="none in both"),
                               Patch(facecolor=analysis.MAP_BACK[0], edgecolor="#898781", label="water or cloud")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8, frameon=False)
            ax.set_title(f"January {YEARS[a]} -> January {YEARS[b]}\nloss {lost.sum() * PIXEL_KM2:.2f} km², "
                         f"gain {won.sum() * PIXEL_KM2:.2f} km²", fontsize=11)
            ax.axis("off")
            analysis.map_marks(ax)
            write(self.change_box, f"Loss: {lost.sum() * PIXEL_KM2:.2f} km²\nGain: {won.sum() * PIXEL_KM2:.2f} km²\n"
                                   f"Net: {(won.sum() - lost.sum()) * PIXEL_KM2:+.2f} km²\n\n"
                                   f"Vegetation {YEARS[a]}: {r['area'][a]:.2f} km², "
                                   f"{YEARS[b]}: {r['area'][b]:.2f} km².\n\n"
                                   "Most year-to-year loss and gain does not last (Day 4): the whole landscape is "
                                   "a little greener or browner in some Januaries. Use 'Year of lasting loss' to "
                                   "see the change that stayed.")
        elif mode in CHANGES[1:3]:
            ax = fig.add_subplot()
            when = r["lost_in"] if mode == CHANGES[1] else r["gained_in"]
            years, cmap, norm = year_colours(YEARS[1], YEARS[-1])
            img = analysis.background(valid, veg[0])
            for i, year in enumerate(years):
                img[when == year] = cmap(i)[:3]
            ax.imshow(img, interpolation="nearest")
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=years, shrink=0.8,
                         label="year (first January without vegetation)" if mode == CHANGES[1] else
                         "year (first January with vegetation)")
            ax.axis("off")
            analysis.map_marks(ax)
            sure = sum((when == y).sum() for y in years if certain_year(y)) * PIXEL_KM2
            ax.set_title(f"{mode}: {(when > 0).sum() * PIXEL_KM2:.2f} km² in all,\n{sure:.2f} km² in the certain "
                         f"years 2016-2024", fontsize=11)
            per_year = "\n".join(f"{y}: {(when == y).sum() * PIXEL_KM2:.2f} km²" +
                                 ("" if certain_year(y) else "  (not yet certain)") for y in years)
            write(self.change_box, f"{mode}, km² per year:\n{per_year}\n\nRule (Day 4): vegetation in at least 2 of "
                                   "the 3 Januaries before and in at least half of all Januaries before, then none in "
                                   "that January and every later one (the opposite for gain). Pale green / grey "
                                   "behind: vegetation / none in 2014.")
        else:
            diff, mean, sd, drop, rise = analysis.ndvi_change(r["ndvi"][a], r["ndvi"][b], valid)
            ax = fig.add_subplot(1, 2, 1)
            cmap = matplotlib.colormaps["BrBG"].copy()
            cmap.set_bad("#ffffff")
            im = ax.imshow(np.where(valid, diff, np.nan), cmap=cmap, vmin=-0.3, vmax=0.3, interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.6, label=f"NDVI {YEARS[b]} - NDVI {YEARS[a]}")
            ax.axis("off")
            analysis.map_marks(ax)
            ax.set_title(f"NDVI difference, {YEARS[b]} - {YEARS[a]}", fontsize=11)
            ax = fig.add_subplot(1, 2, 2)
            img = analysis.background(valid, np.zeros_like(valid))
            img[drop] = to_rgb(analysis.LOSS)
            img[rise] = to_rgb(analysis.GAIN)
            ax.imshow(img, interpolation="nearest")
            ax.axis("off")
            analysis.map_marks(ax)
            ax.set_title(f"2-sigma rule: drop {drop.sum() * PIXEL_KM2:.2f} km² (red),\n"
                         f"rise {rise.sum() * PIXEL_KM2:.2f} km² (blue)", fontsize=11)
            write(self.change_box, f"Mean change of the box: {mean:+.3f}\nStandard deviation: {sd:.3f}\n"
                                   f"Significant drop (below mean - 2 sd): {drop.sum() * PIXEL_KM2:.2f} km²\n"
                                   f"Significant rise (above mean + 2 sd): {rise.sum() * PIXEL_KM2:.2f} km²\n\n"
                                   "Measured from the mean, so a greening or browning of the whole box is left out "
                                   "and only places that changed more than their surroundings remain.")
        self.change_canvas.draw_idle()

    def make_accuracy(self):
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text="  Accuracy  ")
        self.accuracy_tab = frame
        top = ttk.Frame(frame, padding=(10, 6, 10, 0))
        top.pack(side="top", fill="x")
        columns = ["comparison", "n", "overall", "prod_veg", "prod_none", "user_veg", "user_none", "kappa"]
        names = ["Comparison", "Pixels", "Overall %", "Prod. veg %", "Prod. none", "User veg %", "User none",
                 "Kappa"]
        self.table = ttk.Treeview(top, columns=columns, show="headings", height=6)
        for col, name in zip(columns, names):
            self.table.heading(col, text=name)
            wide = col == "comparison"
            self.table.column(col, width=430 if wide else 100, anchor="w" if wide else "e", stretch=wide)
        self.table.pack(fill="x")

        body = ttk.Frame(frame)
        body.pack(side="top", fill="both", expand=True)
        left = ttk.Frame(body, width=330, padding=(10, 6))
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        heading(left, "Reference points")
        row = ttk.Frame(left)
        row.pack(anchor="w")
        self.point_year = tk.IntVar(value=2014)
        for year in analysis.POINT_YEARS:
            ttk.Radiobutton(row, text=str(year), value=year, variable=self.point_year,
                            command=lambda: self.go_to_point(0)).pack(side="left", padx=(0, 8))
        row = ttk.Frame(left)
        row.pack(anchor="w", pady=4)
        ttk.Button(row, text="< Prev", width=7, command=lambda: self.step_point(-1)).pack(side="left")
        self.point_name = tk.StringVar()
        self.point_box = ttk.Combobox(row, textvariable=self.point_name, width=7, state="readonly")
        self.point_box.pack(side="left", padx=4)
        self.point_box.bind("<<ComboboxSelected>>", lambda _: self.draw_point())
        ttk.Button(row, text="Next >", width=7, command=lambda: self.step_point(1)).pack(side="left")
        self.show_class = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Show our map's class (blind if off)", variable=self.show_class,
                        command=self.draw_point).pack(anchor="w")
        ttk.Label(left, text="Label this point:").pack(anchor="w", pady=(4, 0))
        row = ttk.Frame(left)
        row.pack(anchor="w", pady=(0, 4))
        for value, text in [("v", "v: vegetation"), ("n", "n: none"), ("?", "?: unsure")]:
            ttk.Button(row, text=text, width=len(text) + 1, command=lambda x=value: self.set_label(x)).pack(
                side="left", padx=2)
        ttk.Button(left, text="Open this point in Google Earth (web)", command=self.open_earth).pack(anchor="w", pady=2)
        self.point_info = text_box(left, 13)

        heading(left, "Hotspot photo series")
        row = ttk.Frame(left)
        row.pack(anchor="w")
        self.site_name = tk.StringVar()
        self.site_box = ttk.Combobox(row, textvariable=self.site_name, width=22, state="readonly")
        self.site_box.pack(side="left")
        self.site_box.bind("<<ComboboxSelected>>", lambda _: self.draw_site())
        ttk.Button(row, text="Show", command=self.draw_site).pack(side="left", padx=4)
        self.acc_fig, self.acc_canvas, _ = add_figure(right)
        self.fill_point_list()

    def draw_accuracy(self):
        self.fill_accuracy_table()
        self.draw_point()

    def fill_accuracy_table(self):
        self.table.delete(*self.table.get_children())
        for row in self.r["day5_accuracy"]:
            self.table.insert("", "end", values=[row["comparison"], row["pixels_or_points"], row["overall_pct"],
                                                 row["producer_vegetation_pct"], row["producer_none_pct"],
                                                 row["user_vegetation_pct"], row["user_none_pct"], row["kappa"]])
        sites = {p["name"] for p in self.r["photos"] if p["kind"] == "site"}
        self.site_box.configure(values=[s for s in SITE_ORDER if s in sites] + sorted(sites - set(SITE_ORDER)))

    def points_file(self, year):
        return self.points_dir / f"points_{year}.csv"

    def read_points(self, year):
        return read_csv(self.points_file(year))

    def fill_point_list(self):
        names = [p["point"] for p in self.read_points(self.point_year.get())]
        self.point_box.configure(values=names)
        if names and self.point_name.get() not in names:
            self.point_name.set(names[0])

    def go_to_point(self, index):
        self.fill_point_list()
        names = list(self.point_box.cget("values"))
        if names:
            self.point_name.set(names[index])
        self.draw_point()

    def step_point(self, step):
        names = list(self.point_box.cget("values"))
        if not names:
            return
        i = names.index(self.point_name.get()) if self.point_name.get() in names else 0
        self.point_name.set(names[(i + step) % len(names)])
        self.draw_point()

    def current_point(self):
        for p in self.read_points(self.point_year.get()):
            if p["point"] == self.point_name.get():
                return p
        return None

    def point_photos(self, year, name):
        rows = [p for p in self.r["photos"] if p["kind"] == "point" and p["year"] == str(year) and p["name"] == name]
        return sorted(rows, key=lambda p: p["capture_date"])

    def photo_image(self, photo):
        path = DATA / "photos" / photo["file"]
        if not path.exists():
            return None
        if path not in self.photo_cache:
            with Image.open(path) as img:
                self.photo_cache[path] = np.asarray(img.convert("RGB"))
        return self.photo_cache[path]

    def point_accuracy(self, year):
        points = self.read_points(year)
        classes = self.r["classes"][YEARS.index(year)]
        mapped = np.where(classes > 0, (classes >= 2).astype(int), -1)
        if not points:
            return "No point file yet: run analysis.py."
        rows = np.array([int(p["row"]) for p in points])
        cols = np.array([int(p["col"]) for p in points])
        labels = [analysis.read_label(p["label"]) for p in points]
        done = np.array([lab is not None for lab in labels])
        truth = np.array([lab if lab is not None else -1 for lab in labels])
        ours = mapped[rows, cols]
        text = f"{year}: {done.sum()} of {len(points)} points labelled"
        if done.sum() >= 10 and len(set(truth[done])) == 2 and len(set(ours[done])) == 2:
            share = [(mapped == 0).sum() / (mapped >= 0).sum(), (mapped == 1).sum() / (mapped >= 0).sum()]
            cm = analysis.confusion(ours[done], truth[done])
            plain, area = analysis.accuracy(cm), analysis.accuracy(analysis.weighted(cm, share))
            text += (f"\nAgree {int(np.trace(cm))}, disagree {int(cm.sum() - np.trace(cm))}"
                     f"\nOverall {100 * plain['overall']:.1f} %, kappa {plain['kappa']:.2f}"
                     f"\nWeighted by map area: overall {100 * area['overall']:.1f} %, kappa {area['kappa']:.2f}")
        return text

    def draw_point(self):
        r = self.r
        year = self.point_year.get()
        self.fill_point_list()
        p = self.current_point()
        fig = self.acc_fig
        fig.clear()
        if p is None:
            ax = fig.add_subplot()
            ax.text(0.5, 0.5, f"No reference points for {year} yet - run analysis.py", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            write(self.point_info, self.point_accuracy(year))
            self.acc_canvas.draw_idle()
            return
        row, col = int(p["row"]), int(p["col"])
        photos = self.point_photos(year, p["point"])
        square = analysis.pixel_square(row, col, r["profile"])
        lat, lon = np.array([c[0] for c in square]), np.array([c[1] for c in square])
        kappa_rows = r["day5_threshold_vs_kappa"]
        grid = fig.add_gridspec(2 if kappa_rows else 1, max(1, len(photos)) + 1,
                                height_ratios=[3, 1.1] if kappa_rows else [1])
        for i, photo in enumerate(photos or [None]):
            ax = fig.add_subplot(grid[0, i])
            img = self.photo_image(photo) if photo else None
            if img is None:
                ax.text(0.5, 0.5, "no photo yet -\nrun download.py photos", ha="center", va="center",
                        transform=ax.transAxes, color="#52514e")
                ax.axis("off")
                continue
            height, width = img.shape[:2]
            ax.imshow(img)
            x, y = analysis.on_photo(lon, lat, photo, width, height)
            ax.plot(x, y, color="yellow", lw=1.6)
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_title(f"{photo['capture_date']}  ({photo.get('satellite', '')}, "
                         f"{float(photo.get('metres_per_pixel') or 0):.2f} m/px)", fontsize=9)
            ax.axis("off")
        ax = fig.add_subplot(grid[0, max(1, len(photos))])
        k = YEARS.index(year)
        r0, c0 = max(0, row - 7), max(0, col - 7)
        ax.imshow(self.fcc(k)[r0:row + 8, c0:col + 8], interpolation="nearest")
        ax.add_patch(Rectangle((col - c0 - 0.5, row - r0 - 0.5), 1, 1, fill=False, ec="yellow", lw=1.6))
        ax.set_title(f"Landsat 8, {r['log'][year]['date']}\n(false colour, 450 m across)", fontsize=9)
        ax.axis("off")
        if kappa_rows:
            ax = fig.add_subplot(grid[1, :])
            cuts = [float(x["threshold"]) for x in kappa_rows]
            ax.plot(cuts, [float(x["kappa_vs_worldcover_2021"]) for x in kappa_rows], "o-", color="#2a78d6", ms=3)
            ax.axvline(r["t"], color="#0b0b0b", lw=1)
            ax.set_xlabel("NDVI threshold")
            ax.set_ylabel("kappa")
            ax.set_title("Agreement with WorldCover 2021 for every threshold (line: Otsu's T)", fontsize=9)
            analysis.tidy(ax)
        label = p["label"].strip()
        dates = ", ".join(f"{ph['capture_date']} ({ph.get('satellite', '')})" for ph in photos) or "none yet"
        ours = "vegetation" if r["veg"][k][row, col] else "no vegetation"
        text = (f"{p['point']} ({year}): {float(p['lat']):.5f} N, {float(p['lon']):.5f} E\n"
                f"Row {row}, column {col}\nPhotos: {dates}\n"
                f"Label: {label + ' (' + LABEL_NAMES.get(label, '') + ')' if label else 'not labelled'}\n"
                + (f"Note: {p['note']}\n" if p.get("note") else "")
                + (f"Our map (January {year}): {ours}\n" if self.show_class.get() else "Our map: hidden\n")
                + "\n" + self.point_accuracy(year)
                + "\n\nKeys: v / n / ? label the point, left and right arrows move.")
        write(self.point_info, text)
        self.acc_canvas.draw_idle()

    def set_label(self, value):
        path = self.points_file(self.point_year.get())
        if not path.exists():
            return
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row["point"] == self.point_name.get():
                row["label"] = value
        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        except PermissionError:
            messagebox.showerror("Could not save", f"{path.name} is open in another program (Excel?). Close it and "
                                                   "try again.")
            return
        self.status.set(f"{self.point_name.get()} ({self.point_year.get()}) labelled '{value}' in {path.name}")
        self.step_point(1)

    def open_earth(self):
        p = self.current_point()
        if p:
            webbrowser.open(f"https://earth.google.com/web/@{p['lat']},{p['lon']},30a,300d,35y,0h,0t,0r")

    def draw_site(self):
        r = self.r
        name = self.site_name.get()
        photos = sorted([p for p in r["photos"] if p["kind"] == "site" and p["name"] == name],
                        key=lambda p: p["capture_date"])
        fig = self.acc_fig
        fig.clear()
        if not photos:
            ax = fig.add_subplot()
            ax.text(0.5, 0.5, "No photos for this site yet - run download.py photos", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            self.acc_canvas.draw_idle()
            return
        fig_w, fig_h = fig.get_size_inches()
        cols = max(range(1, len(photos) + 1),
                   key=lambda c: min(fig_w / c, 0.8 * fig_h / np.ceil(len(photos) / c)))
        rows = int(np.ceil(len(photos) / cols))
        for i, photo in enumerate(photos):
            ax = fig.add_subplot(rows, cols, i + 1)
            img = self.photo_image(photo)
            if img is None:
                ax.axis("off")
                continue
            height, width = img.shape[:2]
            ax.imshow(img)
            if name in r["outlines"]:
                lat, lon = zip(*r["outlines"][name])
                x, y = analysis.on_photo(np.array(lon), np.array(lat), photo, width, height)
                ax.plot(x, y, color="yellow", lw=1.4)
            else:
                lines = edge_lines(r["patch"] > 0, r["profile"], photo, width, height)
                ax.add_collection(LineCollection(lines, colors="yellow", linewidths=1.0))
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_title(f"{photo['capture_date']}\n{photo.get('satellite', '')}", fontsize=9)
            ax.axis("off")
        check = r["checks"].get(name, {})
        fig.suptitle(f"{name}" + (f" - our year of loss {check['our_year_of_loss']}" if check.get("our_year_of_loss")
                                  else "") + "\n(yellow: outline / our lasting-loss patches)", fontsize=11)
        write(self.point_info, f"{name}\n" + "\n".join(f"{k.replace('_', ' ')}: {v}" for k, v in check.items()
                                                       if v and k not in ("site",)))
        self.acc_canvas.draw_idle()

    def key_pressed(self, event):
        if self.tabs.select() != str(self.accuracy_tab):
            return
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)) and event.widget is not \
                self.point_info:
            return
        if event.char in ("v", "n", "?"):
            self.set_label(event.char)
        elif event.keysym == "Left":
            self.step_point(-1)
        elif event.keysym == "Right":
            self.step_point(1)

    def make_airport(self):
        left, right = self.page("Second area")
        heading(left, "Year")
        self.air_year = tk.IntVar(value=YEARS[-1])
        tk.Scale(left, from_=YEARS[0], to=YEARS[-1], orient="horizontal", variable=self.air_year, length=270,
                 command=lambda _: self.draw_airport()).pack(anchor="w")
        heading(left, "Show")
        self.air_layer = tk.StringVar(value="False colour")
        for name in ["False colour", "Vegetation map", "Year of lasting loss"]:
            ttk.Radiobutton(left, text=name, value=name, variable=self.air_layer,
                            command=self.draw_airport).pack(anchor="w")
        heading(left, "Numbers")
        self.air_box = text_box(left, 14)
        self.air_fig, self.air_canvas, _ = add_figure(right)

    def draw_airport(self):
        r = self.r
        fig = self.air_fig
        fig.clear()
        if "air_refl" not in r:
            ax = fig.add_subplot()
            ax.text(0.5, 0.5, "The second area is not there yet:\nrun  download.py airport  and then  analysis.py",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            self.air_canvas.draw_idle()
            return
        year = self.air_year.get()
        k = YEARS.index(year)
        layer = self.air_layer.get()
        grid = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
        ax = fig.add_subplot(grid[0, 0])
        colour = "#0b0b0b"
        if layer == "False colour":
            ax.imshow(false_colour(r["air_refl"], r["air_limits"], k), interpolation="nearest")
            colour = "#ffffff"
        elif layer == "Vegetation map":
            veg = r["air_classes"][k] == 2
            ax.imshow(analysis.background(r["air_valid"], veg), interpolation="nearest")
        else:
            years, cmap, norm = year_colours(YEARS[1], YEARS[-1])
            img = analysis.background(r["air_valid"], r["air_classes"][0] == 2)
            for i, y in enumerate(years):
                img[r["air_lost_in"] == y] = cmap(i)[:3]
            ax.imshow(img, interpolation="nearest")
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=years[::2], shrink=0.7, label="lost in")
        ax.axis("off")
        analysis.map_marks(ax, colour=colour)
        ax.set_title(f"January {year}: {layer.lower()}", fontsize=11)

        rows = r["day5_airport_area"]
        ax = fig.add_subplot(grid[0, 1])
        if rows:
            years = [int(x["year"]) for x in rows]
            ax.plot(years, [float(x["vegetation_pct"]) for x in rows], "o-", color="#2e7d32", lw=2, ms=5,
                    label="airport box")
            ax.plot(years, [float(x["main_box_vegetation_pct"]) for x in rows], "o--", color="#898781", lw=1.5, ms=4,
                    label="main box (Powai-Aarey-Kanjurmarg)")
            ax.axvline(year, color="#2a78d6", lw=1)
            ax.set_xticks(years[::4])
            ax.set_ylabel("vegetation, % of the land")
            ax.legend(fontsize=8, loc="center right")
            ax.set_title("Same scenes, same steps,\nsame threshold", fontsize=10)
            analysis.tidy(ax)
            row = rows[k]
            write(self.air_box, f"January {year} ({row['date']})\nLand used: {row['land_km2']} km²\n"
                                f"Vegetation: {row['vegetation_km2']} km² ({row['vegetation_pct']} %)\n"
                                f"Without the forest scaling: {row['vegetation_pct_unscaled']} %\n"
                                f"Open water in the box: {row['open_water_km2']} km²\n"
                                + (f"Change from the year before: {row['change_km2']} km²\n"
                                   f"Lasting loss first seen this year: {row['lasting_loss_km2']} km²\n"
                                   f"Lasting gain first seen this year: {row['lasting_gain_km2']} km²"
                                   if row["change_km2"] else ""))
        self.air_canvas.draw_idle()

    def make_run(self):
        left, right = self.page("Run")
        heading(left, "Run the project")
        steps = [("Run analysis.py", ["analysis.py"]), ("Download everything", ["download.py"])]
        if "def get_photos" in (HERE / "download.py").read_text(encoding="utf-8"):
            steps.append(("Download the photos", ["download.py", "photos"]))
        steps.append(("Redraw the Day 1 pictures", ["download.py", "figures"]))
        self.run_buttons = []
        for text, args in steps:
            button = ttk.Button(left, text=text, command=lambda a=args: self.start(a))
            button.pack(fill="x", pady=2)
            self.run_buttons.append(button)
        ttk.Button(left, text="Stop", command=self.stop).pack(fill="x", pady=(10, 2))
        ttk.Button(left, text="Reload the results", command=self.reload).pack(fill="x", pady=2)
        ttk.Label(left, text="analysis.py takes 1-3 minutes. download.py needs the internet: everything takes "
                             "6-15 minutes, the photos alone a few minutes. After a run, press 'Reload the results' "
                             "to see the new numbers in the other tabs.", wraplength=280,
                  foreground="#52514e").pack(anchor="w", pady=8)
        self.log_box = ScrolledText(right, font=("Consolas", 9), wrap="none")
        self.log_box.pack(fill="both", expand=True, padx=6, pady=6)

    def start(self, args):
        if self.process:
            return
        self.log_box.insert("end", f"\n> python {' '.join(args)}\n")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen([sys.executable, "-u"] + args, cwd=HERE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                        env=env, creationflags=flags)
        for button in self.run_buttons:
            button.state(["disabled"])
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()
        self.after(100, self.show_output)
        self.status.set(f"Running {' '.join(args)} ...")

    def read_output(self, process):
        for line in process.stdout:
            self.output.put(line)
        process.wait()
        self.output.put(None)

    def show_output(self):
        finished = False
        while not self.output.empty():
            line = self.output.get()
            if line is None:
                finished = True
            else:
                self.log_box.insert("end", line)
                self.log_box.see("end")
        if finished:
            code = self.process.returncode
            self.log_box.insert("end", f"\n[finished, exit code {code}]\n")
            self.status.set(f"Finished (exit code {code}). Press 'Reload the results' to see the new results.")
            self.process = None
            for button in self.run_buttons:
                button.state(["!disabled"])
        else:
            self.after(100, self.show_output)

    def stop(self):
        if self.process:
            self.process.terminate()

    def reload(self):
        self.status.set("Reloading the results ...")
        self.config(cursor="watch")
        self.update_idletasks()
        self.r = load_results()
        self.photo_cache, self.fcc_cache = {}, {}
        self.draw_everything()
        self.config(cursor="")
        self.status.set(f"Results reloaded.  Otsu threshold T = {self.r['t']:.3f}")

    def close(self):
        if self.process and not messagebox.askokcancel("Still running", "A run is still going. Stop it and close?"):
            return
        self.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
