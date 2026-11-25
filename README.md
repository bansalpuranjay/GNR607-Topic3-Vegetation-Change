# GNR 607 Topic 3: Vegetation cover change around IIT Bombay (2014 to 2026)

**Team:** Puranjay Bansal (23B0731), Arjun Bhardwaj (23B0664), Rishabh Agarwal (23B0758), Shlok Agrawal (23B2240)

In this project we took one Landsat 8 image for every January from 2014 to 2026 (13 images) of the area around
IIT Bombay (Powai, Aarey and Kanjurmarg in Mumbai). Then using our own code we made vegetation maps for every year,
found where vegetation was lost or gained and checked how correct our maps are. There is also a small app (a window
with tabs) to see all the results.

This file tells how to set everything up on any laptop, how to use the code and the app, and how to run the same
thing for some other place, for example Kerala.

---

## 1. What is inside this folder

```
GNR607_Topic3_Final_Submission/
    README.md                          (this file)
    GNR607_Topic3_Presentation.pptx    (our slides)
    code/
        download.py        downloads the satellite images and other data
        analysis.py        does all the processing and makes every figure and table
        app.py             the app (window) to see the results
        requirements.txt   list of Python libraries needed
        data/              the downloaded images, our maps, photos and the point labels
        output/            all figures (.png), tables (.csv), Google Earth files (.kml) and run_log.txt
    kochi_check/           our test of the same code for Kochi in Kerala (see section 6)
```

The `data` and `output` folders are already full, so **the app works directly, without internet and without
downloading anything again.**

---

## 2. Setting it up on a new laptop

You only need Python and 4 libraries. Do these steps once on every new laptop.

### Step 1: Install Python

- Download Python 3.12 from https://www.python.org/downloads/ (we used 3.12; 3.14 also worked for us).
- **Windows:** while installing, tick the box **"Add python.exe to PATH"**. Keep the other options as they are
  (the option "tcl/tk and IDLE" must stay ticked, the app window needs it).
- **Mac:** the installer from python.org is fine. (If you use Homebrew Python, also run `brew install python-tk`.)
- **Linux (Ubuntu):** run `sudo apt install python3 python3-venv python3-tk`

### Step 2: Copy the folder and open a terminal inside `code`

- Copy the whole `GNR607_Topic3_Final_Submission` folder to the laptop (or unzip it).
- **Windows:** open the `code` folder in File Explorer, click on the address bar at the top, type `cmd` and press
  Enter. A black window (Command Prompt) opens, already inside the `code` folder.
- **Mac / Linux:** open Terminal and go to the folder with `cd`, for example
  `cd ~/Downloads/GNR607_Topic3_Final_Submission/code`

### Step 3: Make a virtual environment and install the libraries

A virtual environment (`venv`) is a separate Python only for this project, so nothing else on the laptop gets
disturbed. It needs internet only this one time (less than 100 MB, takes 1 to 2 minutes).

**Windows:**
```
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

**Mac / Linux:**
```
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

There is no need to "activate" the venv, we always write `venv\Scripts\python` (Windows) or `venv/bin/python`
(Mac / Linux) in front of the file name.

#### Libraries to download

Only these 4 libraries have to be downloaded. The pip command above installs all of them from `requirements.txt`.

| Library | What we use it for | Version we used |
|---|---|---|
| numpy | all the array maths on the images | 2.5 |
| rasterio | reading and writing the satellite images (GeoTIFF) and changing coordinates (UTM to latitude / longitude) | 1.5.2 |
| matplotlib | all the figures and the graphs inside the app | 3.11.2 |
| pillow | reading and saving the photo files (.jpg / .png) | 13.0 |

If you do not want to use `requirements.txt`, this one line does the same thing:
```
venv\Scripts\python -m pip install numpy rasterio matplotlib pillow
```

- pip also downloads the small helper libraries these 4 need (for example affine, certifi, click, pyparsing,
  fonttools, contourpy, kiwisolver), automatically. You do not have to install them one by one.
- rasterio already has GDAL inside it, so there is no need to install GDAL, QGIS or anything else separately.
- These come with Python itself, nothing to download: tkinter (the app window), csv, json, urllib, threading,
  subprocess, concurrent.futures, datetime, pathlib.
- No API key, account or login is needed. All the data sources are free (see section 7).
- Internet is needed only for the pip command (once) and when you run `download.py`. The app and `analysis.py`
  work offline.

If Windows says `'python' is not recognized`, either Python was installed without "Add to PATH", or use
`py -m venv venv` instead of `python -m venv venv`.

### Step 4: Open the app

**Windows:**
```
venv\Scripts\python app.py
```
**Mac / Linux:**
```
venv/bin/python app.py
```

It takes a few seconds to load and then the window opens. Folder names with spaces are no problem.

### If something goes wrong

| Problem | What to do |
|---|---|
| `No module named rasterio` (or numpy etc.) | You ran the normal `python` and not the one in `venv`. Write `venv\Scripts\python app.py`. |
| `No module named tkinter` / `_tkinter` | Windows: install Python again with "tcl/tk and IDLE" ticked. Linux: `sudo apt install python3-tk`. |
| pip install fails | Check the internet, then run the same pip command again. |
| `DLL load failed ... An Application Control policy has blocked this file` (Windows) | This is Windows Smart App Control checking the newly installed library files the first time. Just run the same command again, it worked for us on the second try. If it still comes, delete the `venv` folder and do step 3 again. |
| App says a file is missing | The `data` or `output` folder is not complete. Copy it again, or run `analysis.py` (see section 4). |

---

## 3. How to use the app

The window has 7 tabs at the top. On every tab the controls are on the left and the picture is on the right.
Under every picture there is a small toolbar (magnifier = zoom, cross arrows = move, house = back to normal).

**Overview**
The main graph: vegetation area (km²) for every January, green triangles where it went up and red where it went
down, with the trend line. On the left are the key results in short.

**Images**
- Move the **Year** slider to choose a year.
- Under **Show** choose *False colour* (vegetation looks red), *NDVI* or *Vegetation map*.
- **Click on any pixel** of the map. A small graph next to the map shows the NDVI of that pixel in all 13 Januaries
  (dashed line = threshold T), and on the left under "The pixel you clicked" you get its row, column, latitude,
  longitude and WorldCover class.
- Note: if the zoom or move button of the toolbar is on, clicking will zoom instead. Press it again to switch it off.

**Threshold**
- Move the **NDVI threshold T** slider (0.20 to 0.70). When you leave the slider (or press **Recompute**) the maps
  of all years and the area graph are made again with this T.
- **Back to Otsu's T** goes back to the threshold our code found (0.43).
- **Map of the year:** choose a year. Its map shows in blue the pixels that are vegetation only with your T, and in
  red the ones that are vegetation only with Otsu's T. "What changes" gives the numbers.
- Nothing is saved here, so you can play with it freely.

**Change**
- Choose two years under **From and to**.
- Under **Show**: *Loss and gain between two years*, *Year of lasting loss*, *Year of lasting gain* or
  *NDVI difference (2-sigma rule)*.
- The **Numbers** box gives the areas in km².

**Accuracy**
- The table at the top of this tab gives the accuracy against ESA WorldCover and against our reference points.
- **Reference points:** choose 2014 or 2026, then go through the points with **< Prev** and **Next >** (or the left
  and right arrow keys). For each point you see a high resolution photo with the 30 m Landsat pixel marked.
- **Show our map's class (blind if off):** keep it off while labelling so you are not influenced by our map.
- **Label this point:** the buttons *v: vegetation*, *n: none*, *?: unsure* (or the keys v, n, ?).
  **Careful: these save the label into `data/points_2014.csv` or `data/points_2026.csv`.** If you only want to
  show the app, do not press them.
- **Open this point in Google Earth (web)** opens the point in the browser.
- **Hotspot photo series:** choose a site (for example the Aarey Metro depot) and press **Show** to see dated
  photos of it.

**Second area**
The same method on a second box, around the new Navi Mumbai airport at Ulwe. Choose the **Year** and under
**Show** pick *False colour*, *Vegetation map* or *Year of lasting loss*.

**Run**
Buttons to run the code from inside the app: **Run analysis.py**, **Download everything**,
**Download the photos**, **Redraw the Day 1 pictures**, plus **Stop** and **Reload the results**. What the
program prints comes on the right side. After a run press **Reload the results** to see the new numbers.
"Download everything" needs internet and replaces the files in `data`, so do not press it just for showing.

---

## 4. Running the code without the app

All commands are typed in the terminal inside the `code` folder (on Mac / Linux write `venv/bin/python` instead
of `venv\Scripts\python`).

| Command | What it does | Time |
|---|---|---|
| `venv\Scripts\python analysis.py` | Does all the processing again and rewrites everything in `output` and the maps in `data`. Your point labels are never overwritten. No internet needed. | about 1 minute |
| `venv\Scripts\python download.py` | Downloads everything again (Landsat images, WorldCover, rain, site outlines, second area). Needs internet. | 5 to 10 minutes |
| `venv\Scripts\python download.py photos` | Downloads the high resolution photos of the reference points and hotspots. | a few minutes |
| `venv\Scripts\python download.py figures` | Redraws the Day 1 pictures from the files in `data`, no internet. | seconds |
| `venv\Scripts\python download.py airport` | Downloads only the second area. | 1 to 2 minutes |
| `venv\Scripts\python download.py day4` | Downloads only the rain data and the two site outlines. | seconds |
| `venv\Scripts\python download.py box W S E N` | Helper for a new place (see section 6). | seconds |

Everything that `analysis.py` prints is also saved in `output/run_log.txt`.

**Order when starting from an empty `data` folder:**
1. `download.py` (images and other data)
2. `analysis.py` (makes the maps and also writes the random reference points `data/points_2014.csv` and `points_2026.csv`)
3. `download.py photos` (photos of those points)
4. label the points in the app (Accuracy tab)
5. `analysis.py` again (now the accuracy of the points is calculated)

---

## 5. What the code does (short)

- **download.py:** searches Microsoft Planetary Computer for every Landsat 8 scene of our area from 11 December to
  21 February, measures the cloud and haze inside our box from the quality bands, picks the best January scene and
  cuts out only our box (390 x 300 pixels of 30 m). It also downloads ESA WorldCover 2020 and 2021, daily rain
  (ERA5, through Open-Meteo), two site outlines from OpenStreetMap, the second area and the photos (Esri World
  Imagery Wayback).
- **analysis.py:** Day 2: reflectance, cloud and water masks, NDVI, MNDWI, NDBI, checks of the radiometry and the
  registration, and scaling of NDVI with a forest patch. Day 3: one threshold for all years with Otsu's method,
  vegetation maps cleaned with a 3 x 3 majority filter, area every year. Day 4: change maps, lasting loss and gain,
  NDVI difference with the 2-sigma rule, loss patches (opening, conditional dilation, boundary). Day 5: accuracy
  (confusion matrix, overall / producer's / user's accuracy, kappa), principal components of the 13 NDVI images
  and the second area. Day 7: image sizes and ground control points.
- **app.py:** reads the files in `data` and `output` and shows them. It uses the functions of `analysis.py`
  when it has to calculate something again (for example a new threshold).

The libraries are used only for reading and writing files (rasterio, pillow), drawing (matplotlib) and array maths
(numpy). The image processing steps (Otsu, morphology, majority filter, change detection, accuracy, PCA covariance
and so on) are written by us. The full list of library calls is in `output/day7_library_calls.csv`.

---

## 6. Running it for some other place (example: Kerala)

The code can run for any place in India. We tried it for **Kochi in Kerala (Kakkanad and Infopark area)** and it
worked fully: the images of all 13 years were found, Otsu gave T = 0.54 there and vegetation came out as 55 to 64 %
of the land. These are the exact steps.

### Step 1: Choose your box

Open Google Maps and right click on the top left corner of the area you want. It shows the latitude and longitude,
for example `10.05, 76.30`. Do the same for the bottom right corner. Now you have:

- west = smaller longitude, east = bigger longitude
- south = smaller latitude, north = bigger latitude

For Kochi we took west 76.30, south 9.97, east 76.41, north 10.05 (about 12 km x 9 km, almost the same size as our
Mumbai box). A bigger box also works but is slower.

The box should have some water that stays all year (lake, backwater, river or sea), some trees and some buildings,
because the code uses deep water and big roofs as fixed reference points.

### Step 2: Make a separate copy of the code

Copy the `code` folder and name the copy for example `code_kerala`. In the copy, **delete everything inside
`data` and `output`** (the old files are for Mumbai, especially the point files and the photos). Do not copy
the `venv` folder. Make a new venv inside `code_kerala` exactly like in section 2, step 3.

### Step 3: Let the helper find the numbers for your box

In the terminal inside `code_kerala`:
```
venv\Scripts\python download.py box 76.30 9.97 76.41 10.05
```
(the order is west south east north). For Kochi it printed:
```
put these in download.py:
BOX = (642435, 1102365, 654585, 1111275)
LONLAT = [76.3, 9.97, 76.41, 10.05]
TILE = "N09E075"
UTM = "EPSG:32643"
(the box is 405 x 297 pixels of 30 m)
Landsat 8 path and row over this box:
  PATH, ROW = "144", "053"    whole box inside the scene: yes
```
It gives the box in UTM metres (with the edges exactly on Landsat pixel edges), the WorldCover tile, the map
projection and the Landsat path and row. If it shows two paths, take the one with "whole box inside the scene: yes".

### Step 4: Change the settings at the top of `download.py`

Open `download.py` in any editor (Notepad, VS Code) and change these lines at the top:

```python
BOX = (642435, 1102365, 654585, 1111275)
LONLAT = [76.30, 9.97, 76.41, 10.05]
IITB = (76.36, 10.01)
AIRPORT = (650055, 1120095, 656685, 1125675)
PATH, ROW = "144", "053"
TILE = "N09E075"
UTM = "EPSG:32643"
SITES = {}
```

- `BOX`, `LONLAT`, `PATH, ROW`, `TILE`, `UTM`: copy from the helper.
- `IITB`: the main point of your area as (longitude, latitude). The name stays IITB, only change the numbers. It is
  used for the rain data, the pin in Google Earth and the distances of the loss patches. For Kochi we used
  Infopark (76.36, 10.01).
- `AIRPORT`: the second area. It can be any smaller box **in the same Landsat scene**. For Kochi we took the
  Cochin airport: run `venv\Scripts\python download.py box 76.37 10.13 76.43 10.18`, check that it prints the same
  PATH and ROW, and copy only its BOX numbers into AIRPORT.
- `SITES = {}`: the Aarey Metro depot and the Kanjur landfill are Mumbai places. When SITES is empty the code just
  skips the parts about these two sites.
- `SUNRISE, SUNSET` (optional): only used in the time of day picture. For Kerala in January about 6.75 and 18.33.

### Step 5: Download

```
venv\Scripts\python download.py
```
This takes about 5 to 10 minutes. Then open `data/scene_log.csv` (Excel is fine) and look at the columns
`box_cloud_pct`, `box_haze_pct` and `notes`. In our Kochi test 10 years had no cloud at all in the box, 2021 and
2024 had less than 0.5 %, 2025 had 4.45 %, and for 2020 there was no clear January image, so the code took
26 December 2019 (the notes column says "fallback date outside January").

### Step 6: Change the settings at the top of `analysis.py`

```python
BASE = 2023
VIHAR = (slice(None), slice(None))
FOREST = (slice(None), slice(None))
ZOOM = (slice(None), slice(None))
IITB = (76.36, 10.01)
UTM = "EPSG:32643"
```

- `BASE`: the reference year. Take a January year with 0 cloud and low haze in `scene_log.csv` (for Kochi 2023).
- `VIHAR`: where the code looks for deep water as the dark reference (for Mumbai it was Vihar Lake).
- `FOREST`: a forest patch that should look the same every January. NDVI of every year is scaled so this patch has
  the same greenness every year.
- `ZOOM`: the part shown big in the "cleaning" figure.
- `(slice(None), slice(None))` means "the whole box". This is the easy way for the first run.
- `IITB` and `UTM`: same as in `download.py`.

Later, if you want proper areas: run the steps below once, open the app, go to **Images**, click on the corners of a
good forest patch (or lake) and read the row and column under "The pixel you clicked". Then write for example
`FOREST = (slice(0, 40), slice(110, 230))`, which means rows 0 to 39 and columns 110 to 229, and run
`analysis.py` again. (With FOREST as the whole box, the "city trees / forest" check prints nan, that is ok.)

### Step 7: Run the analysis and open the app

```
venv\Scripts\python analysis.py
venv\Scripts\python app.py
```
For Kochi the analysis took about 25 seconds. Now every tab of the app shows Kochi.

### Step 8 (optional): accuracy with photos

```
venv\Scripts\python download.py photos
```
Then in the app go to **Accuracy**, label the 2014 and 2026 points with v / n / ? (keep "Show our map's class" off),
and run `analysis.py` again. Now the accuracy table also has the reference points. You can also fill
`data/hotspot_checks.csv` by looking at the hotspot photo series.

### Things to keep in mind for Kerala

- Kerala has more cloud than Mumbai. A pixel that is cloudy in even one year is left out in all years, so check
  `scene_log.csv`. If many years have cloud, take a smaller box.
- The titles of some figures and of the app still say IIT Bombay, Mumbai or Navi Mumbai airport. That is only text;
  you can change it in the .py files with find and replace.
- All of Kerala is in UTM zone 43 (72° E to 78° E), like Mumbai, so UTM stays `EPSG:32643`. For a place outside
  this (for example Chennai at 80° E) the helper prints the new UTM, put it in both `download.py` and `analysis.py`.
- WorldCover comes in 3° x 3° tiles (lines at 9° N, 12° N, 75° E, 78° E and so on). Keep the box inside one tile.
- Our numbers for Mumbai (T = 0.43, 1.70 km² lasting loss and so on) will be different for another place, the code
  finds everything again from the new images.
- To use other years, change `YEARS` at the top of both `download.py` and `analysis.py` (Landsat 8 images start
  from 2013).
- No extra library is needed for another place, the same 4 libraries are enough.

### Our Kochi check (what happened when we actually ran it)

We did all the steps above for Kochi on a fresh copy of this same code, with a new venv. Only the settings at the
top of the two files were changed. The settings, our labels, the hotspot checks, the run log and three figures are in the
`kochi_check` folder.

| Step | What we got |
|---|---|
| Installing the libraries (new venv) | ok |
| `download.py` | all 13 Januaries found, Landsat path 144 / row 053, every image taken at 10:41 IST. 2020 used 26 December 2019 (no clear January image). Cloud in the box: 0 % in 10 years, under 0.5 % in 2021 and 2024, 4.45 % in 2025. |
| `analysis.py` | finished in about 25 seconds. Otsu T = 0.54 (Mumbai was 0.43). Vegetation 55 to 64 % of the 101 km² of land. Lasting loss 5.04 km² and lasting gain 3.51 km² (2016 to 2024). 184 loss patches, the biggest 46.8 ha, lost in 2020, 2.8 km south-east of the centre point. |
| Accuracy against WorldCover | 83.4 % (2020) and 86.5 % (2021) on all pixels, above 99 % away from class edges |
| Second area (Cochin airport) | vegetation 77.8 % in 2014 and 75.6 % in 2026 |
| Images and control points | all 13 images the same size and grid; every ground control point within one pixel |
| `download.py photos` | 380 photos downloaded: 200 for the 2014 points, 100 for the 2026 points and 80 for the 10 loss patches. This is the slowest step, about 20 minutes, because the photo server is slow. |
| Labelling the points (optional step 8) | we labelled all 200 points by eye with our map's class hidden: 2014: 44 v, 24 n, 32 ?; 2026: 48 v, 29 n, 23 ?. Mostly mixed pixels (half roof, half trees) got "?". |
| Accuracy against the photos | 2026: 89.6 % (77 points, kappa 0.79). 2014: 76.5 % (68 points, kappa 0.52). 2014 is lower because the nearest photos for Kochi are from December 2010 and February 2016, 2 to 3 years away from the image. |
| Hotspot checks | all 10 loss patches checked on the photos: 9 confirmed (trees cleared for roads and buildings), 1 partly (part of it was cleared earlier than our year of loss). |
| `app.py` | all 7 tabs open and show Kochi (with the labels and photos), no errors |

Things we had to correct in the code during this check (they are already corrected in this folder, and our
Mumbai results did not change at all because of them):

- Kochi has no mangroves, so one WorldCover class had no pixels and the threshold check crashed. Now an empty class
  is just skipped.
- For the same reason the mangrove graph had no values and crashed. Now it is drawn without them.
- With `SITES = {}` there is no Aarey depot or Kanjur landfill. Now those parts are skipped.
- In some quarter of the Kochi images no spot stayed the same in all 13 years (haze), so the ground control point
  picker took a bad point. Now it takes the most steady corner in that quarter.

What a computer cannot do for a new place, a person has to do: label the reference points (v / n / ?) in the app,
fill `data/hotspot_checks.csv` after looking at the photos (we did both for Kochi, see the table), and change
the Mumbai names in the figure titles if you want. The photo dates also depend on the place: for Kochi the
2014 points only had photos from 2010 and 2016.

---

## 7. Data sources

- Landsat 8 Collection 2 Level-2 surface reflectance (USGS), from Microsoft Planetary Computer
- ESA WorldCover 2020 (v100) and 2021 (v200)
- ERA5 daily rain, through the Open-Meteo archive
- OpenStreetMap (outlines of the Aarey Metro depot and the Kanjur landfill)
- Esri World Imagery Wayback (dated high resolution photos)

All of these are free and need no login. The full references are on the last two slides of the presentation.
