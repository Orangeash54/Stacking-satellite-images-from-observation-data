#%%
import os
import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from reproject import reproject_interp
from skyfield.api import EarthSatellite, load, wgs84

warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", message=".*non-standard WCS keywords.*")
warnings.filterwarnings("ignore", message=".*All-NaN slice.*")

folder_path = Path(r"C:\Users\orang\OneDrive - University of Warwick\Desktop\29486 - Copy")
search_radius_arcmin = 25.0
max_pointing_drift_arcmin = 5.0
max_time_gap_sec = 300.0

# Basically groups each satellite observation by recording when the telescope is just looking at 1 direction at one time

# Defining the small patch around the satellite
CUTOUT_SIZE = 201              # odd so that their is 1 central pixel, That hopefully the satellite is at
INTERP_ORDER = "bilinear"      # reproject_interp order: "nearest-neighbor", "bilinear", "biquadratic", "bicubic"

# Timing factors that we use to define the satellite position + a possible extra constant to possibly match data better (maybe account for light travel time)
READOUT_BOTTOM_FIRST = True    # Read from the bottom of the image up
TIME_OFFSET_SEC = 0.0          # possible extra constant
N_TIME_ITER = 3                # Number of iterations we use to define the position of the satellite

# Getting the data from celestrack to predict where the satellite is in each image
max_days = 7.0
group = "gps-ops"
json_filename = f"{group}.json"
json_url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=json"

def load_celestrak_json(url, filename, max_days=7.0):
    def read():
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Cached {filename} is not valid JSON. First 200 chars:\n{text[:200]!r}")
            return None

    if load.exists(filename) and load.days_old(filename) < max_days:
        data = read()
        if data:
            return data
        os.remove(filename)                      # bad cache: delete and download again

    load.download(url, filename=filename)
    data = read()
    if not data:
        raise RuntimeError("CelesTrak did not return JSON. See the text printed above.")
    return data


ts = load.timescale()
data = load_celestrak_json(json_url, json_filename, max_days)

gps_satellites = [EarthSatellite.from_omm(ts, fields) for fields in data]
print(f"Loaded {len(gps_satellites)} GPS satellites from JSON")

# getting our position
def get_observer_from_header(header):
    lat = float(header.get("GPS-LAT", 52.376868))
    lon = float(header.get("GPS-LON", -1.567202))
    elev = float(header.get("SITEELEV", 94.0))
    return wgs84.latlon(lat, lon, elevation_m=elev)

# function to get the satellites RA and DEC
def get_sat_radec(sat, observer, astropy_time):
    t_sf = ts.from_astropy(astropy_time)
    topocentric = (sat - observer).at(t_sf)
    ra, dec, _ = topocentric.radec()
    return ra._degrees, dec.degrees


def satellite_position_in_frame(sat, observer, wcs, t_start, exptime, row_period_sec, h):
    # first guess: middle of the frame
    t = t_start + (TIME_OFFSET_SEC + 0.5 * exptime + 0.5 * h * row_period_sec) * u.s
    for _ in range(N_TIME_ITER):
        ra, dec = get_sat_radec(sat, observer, t)
        _, y = wcs.world_to_pixel_values(ra, dec)
        row = float(np.clip(y, 0, h - 1))
        readout_row = row if READOUT_BOTTOM_FIRST else (h - 1) - row
        t = t_start + (TIME_OFFSET_SEC + 0.5 * exptime + readout_row * row_period_sec) * u.s
    ra, dec = get_sat_radec(sat, observer, t)
    x, y = wcs.world_to_pixel_values(ra, dec)
    return t, float(ra), float(dec), float(x), float(y)
# This is the main function which calculates the satellites RA and DEC and its pixel position in x and y on the origional images as well as the exact time.
# It does this by taking the time of the image, then adding an arbitrary time offset that we can define later if needed to fit data
# better, 0.5* the exposure time for, then adding to h (the height of the row)* by the row period which is 117 micro seconds per row.
# It iterates through this procedure to find the time and x y pixel coordinate 3 times to give the most accurate value

def make_satellite_grid(sat_ra_deg, sat_dec_deg, ref_wcs, size):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [sat_ra_deg, sat_dec_deg]
    w.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]      # FITS 1-based -> 0-based index (size-1)/2
    w.wcs.cd = ref_wcs.celestial.pixel_scale_matrix     # deg/pixel, includes rotation
    w.pixel_shape = (size, size)
    return w
# this function sets the WCS of each of the images and zooms in on an area of 201x201 pixel grid where each pixels RA and DEC is defined by the WCS

# --- 3. SCAN FILES & GROUP INTO PASSES ---
all_files = [str(p) for p in folder_path.glob("*.fit*") if p.is_file() and not p.name.startswith("stacked_")]
if not all_files:
    raise FileNotFoundError(f"No .fit/.fits files found in {folder_path}")
print(f"Scanning {len(all_files)} FITS files...")

metadata_list = []
for fp in all_files:
    with fits.open(fp) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        hdr = hdu.header
        h, w = hdu.data.shape
        wcs = WCS(hdr)
        t_obs = Time(hdr["DATE-OBS"], format="isot", scale="utc")
        ra_c, dec_c = wcs.pixel_to_world_values(w / 2.0, h / 2.0)
        coord = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
        metadata_list.append({"path": fp, "time": t_obs, "coord": coord})

metadata_list.sort(key=lambda x: x["time"])

passes = []
current_pass = []
for item in metadata_list:
    if not current_pass:
        current_pass.append(item)
        continue
    prev = current_pass[-1]
    sep_arcmin = prev["coord"].separation(item["coord"]).arcminute
    dt_sec = (item["time"] - prev["time"]).sec
    if sep_arcmin <= max_pointing_drift_arcmin and dt_sec <= max_time_gap_sec:
        current_pass.append(item)
    else:
        passes.append(current_pass)
        current_pass = [item]
if current_pass:
    passes.append(current_pass)

print(f"Found {len(passes)} distinct telescope passes.\n")

# basically just groups the data into seperate passes by recording when the telescope is just looking in 1 direction

# --- 4. IDENTIFY & REPROJECT-STACK EACH PASS ---
saved_outputs = []
centre_idx = (CUTOUT_SIZE - 1) / 2.0                    # 0-based centre pixel of the cutout

for pass_idx, pass_items in enumerate(passes, 1):
    print(f"=== PASS {pass_idx}/{len(passes)} ({len(pass_items)} frames) ===")
    if len(pass_items) < 2:
        continue

    mid_item = pass_items[len(pass_items) // 2]
    with fits.open(mid_item["path"]) as hdul:
        mid_hdr = next(h for h in hdul if h.data is not None).header
        observer = get_observer_from_header(mid_hdr)
        exptime = float(mid_hdr.get("EXPTIME", 0.5))
        ref_wcs = WCS(mid_hdr)                           # scale + rotation for every grid in this pass

    # Rough mid-frame time, only used to identify which satellite this is
    t_mid = mid_item["time"] + (0.5 * exptime + 0.125) * u.s
    t_mid_sf = ts.from_astropy(t_mid)
    center_coord = mid_item["coord"]

    candidates = []
    for sat in gps_satellites:
        ra, dec, dist = (sat - observer).at(t_mid_sf).radec()
        sat_coord = SkyCoord(ra=ra._degrees * u.deg, dec=dec.degrees * u.deg)
        sep_arcmin = center_coord.separation(sat_coord).arcminute
        if sep_arcmin <= search_radius_arcmin:
            candidates.append((sep_arcmin, sat, dist.km))

    candidates.sort(key=lambda x: x[0])

    best_sep, best_sat, _ = candidates[0]
    print(f"Matched: {best_sat.name} (NORAD {best_sat.model.satnum}) | Midpoint Offset: {best_sep:.2f}'")

    cutouts = []
    frame_rows = []          # per-frame record: time, predicted RA/Dec, predicted pixel in original frame

    for item in pass_items:
        with fits.open(item["path"]) as hdul:
            hdu = next(h for h in hdul if h.data is not None)
            data = hdu.data.astype(float)
            hdr = hdu.header
        h, w = data.shape
        wcs_in = WCS(hdr).celestial
        observer = get_observer_from_header(hdr)

        t_start = Time(hdr["DATE-OBS"], format="isot", scale="utc")
        exptime = float(hdr.get("EXPTIME", 0.5))
        rowdelta_us = float(hdr.get("ROWDELTA", 39.027))
        binning = int(hdr.get("CAM-BIN", 3))
        row_period_sec = (rowdelta_us * binning) * 1e-6

        # reading what is needed from earlier functions

        t_sat, ra_sat, dec_sat, px_x, px_y = satellite_position_in_frame(
            best_sat, observer, wcs_in, t_start, exptime, row_period_sec, h)

        # Background-subtract the stars from each image (so they dont clutter visual)
        data_sub = data - np.nanmedian(data[10:2120, 20:3190])

        # Satellite-centred grid for THIS frame, then resample onto it
        wcs_out = make_satellite_grid(ra_sat, dec_sat, ref_wcs, CUTOUT_SIZE)
        cutout, footprint = reproject_interp((data_sub, wcs_in), wcs_out,
                                             shape_out=(CUTOUT_SIZE, CUTOUT_SIZE),
                                             order=INTERP_ORDER)
        cutout[footprint == 0] = np.nan
        cutouts.append(cutout)
        frame_rows.append((t_sat.isot, ra_sat, dec_sat, px_x, px_y, Path(item["path"]).name))
    #gets rid of pixels with no data that may arise from the repoject plotting

    cube = np.array(cutouts)
    stacked_image = np.nanmedian(cube, axis=0)

    # this is what stacks the image, it stores each cutout image in a matrix containing the frame, row and column data
    # then it takes for each pixel position the median of its values for each pass resulting in 1 201x201 stacked image of the satellite

    # Reference time = middle usable frame. Its predicted position defines the
    # WCS of the stacked image, so pixel_to_world on the stack gives
    # predicted position at T_REF + mean offset of the satellite from prediction.
    ref_row = frame_rows[len(frame_rows) // 2]
    t_ref, ra_ref, dec_ref = ref_row[0], ref_row[1], ref_row[2]
    wcs_stack = make_satellite_grid(ra_ref, dec_ref, ref_wcs, CUTOUT_SIZE)

    out_hdr = wcs_stack.to_header()
    out_hdr["T_REF"] = (t_ref, "UTC time the WCS centre refers to")
    out_hdr["SATNAME"] = best_sat.name
    out_hdr["NORAD"] = best_sat.model.satnum
    out_hdr["NFRAMES"] = (len(cutouts), "frames in median stack")
    out_hdr["TOFFSET"] = (TIME_OFFSET_SEC, "extra time offset applied [s]")
    out_hdr["RDBOTTOM"] = (READOUT_BOTTOM_FIRST, "row 0 read out first")
    out_hdr["COMMENT"] = "Satellite-centred frame: WCS valid for the satellite only, not stars."
    out_hdr["COMMENT"] = "Centre pixel = TLE-predicted satellite position at T_REF."

    # storing all of the metadata of our stacked image for further steps where we want to analyse it

    frame_table = Table(rows=frame_rows,
                        names=("T_SAT", "RA_PRED", "DEC_PRED", "X_PRED", "Y_PRED", "FILE"))

    hdul_out = fits.HDUList([
        fits.PrimaryHDU(stacked_image.astype(np.float32), header=out_hdr),
        fits.ImageHDU(cube.astype(np.float32), name="CUTOUTS"),      # each frame's aligned cutout
        fits.table_to_hdu(frame_table),                              # per-frame times/predictions
    ])
    hdul_out[2].name = "FRAMES"

    clean_name = best_sat.name.replace(" ", "_").replace("(", "").replace(")", "")
    timestamp_tag = pass_items[0]["time"].strftime("%Y%m%d_%H%M%S")
    out_file = folder_path / f"stacked_{clean_name}_pass{pass_idx}_{timestamp_tag}.fits"
    hdul_out.writeto(out_file, overwrite=True)
    saved_outputs.append(out_file)
    print(f"Saved -> {out_file.name}  ({len(cutouts)} frames, T_REF {t_ref})\n")

# --- 5. ZOOMED-IN 100x100 PLOT AROUND CENTRE ---
if saved_outputs:
    zoom_radius = 50
    fig, axes = plt.subplots(1, len(saved_outputs), figsize=(5 * len(saved_outputs), 5))
    if len(saved_outputs) == 1:
        axes = [axes]

    for ax, fpath in zip(axes, saved_outputs):
        img = fits.getdata(fpath, ext=0, memmap=False)
        c = int(centre_idx)
        crop = img[c - zoom_radius:c + zoom_radius + 1, c - zoom_radius:c + zoom_radius + 1]

        med = np.nanmedian(crop)
        std = np.nanstd(crop)
        ax.imshow(crop, cmap="gray", origin="lower", vmin=med - 2 * std, vmax=med + 8 * std)
        ax.set_title(f"{fpath.name}\n(centre zoom, crosshair = predicted position)", fontsize=9)
        ax.axhline(zoom_radius, color="cyan", ls="--", lw=0.8, alpha=0.8)
        ax.axvline(zoom_radius, color="cyan", ls="--", lw=0.8, alpha=0.8)

    plt.tight_layout()
    plt.show()