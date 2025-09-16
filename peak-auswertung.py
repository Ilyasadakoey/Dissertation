import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybaselines import Baseline
from scipy.signal import find_peaks
from scipy.integrate import simpson
from scipy.ndimage import minimum_filter
from scipy.signal import savgol_filter

# ---------------------------
# Funktion: Metadaten aus Dateiname extrahieren
# ---------------------------
def extract_metadata(filename):
    """
    Extrahiert Datum, Wabe und Versuch aus Dateinamen wie:
    '2025-09-10_Wabe8_TR200.asc'
    """
    meta = {}
    base = os.path.basename(filename).replace(".asc", "")
    parts = base.split("_")
    if len(parts) >= 3:
        meta['Datum'] = parts[0]
        meta['Wabe'] = parts[1]
        meta['Versuch'] = parts[2]
    return meta

# ---------------------------
# Funktion: Datei einlesen, Baseline korrigieren & Peaks finden
# ---------------------------
def process_file(filepath,
                 lam=1e6,
                 p=0.001,
                 peak_distance=5,
                 peak_window=3,
                 baseline_start=90,
                 peak_mask_prominence_factor=0.02,
                 fallback_minfilter_fraction=0.02,
                 final_prominence_factor=0.01):
    """
    Lädt Datei, sortiert nach m/z, maskiert Peaks im Fit-Bereich (mit Prominenz),
    fitet Baseline ab baseline_start (ALS). Falls keine Peaks gefunden werden,
    probiert ein alternatives Baseline-Verfahren (minimum_filter + savgol).
    Rückgabe: mz, intensity, baseline, corrected, peaks_indices, peak_data
    """

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Header suchen
    try:
        start_index = next(
            i for i, line in enumerate(lines) if line.strip().startswith("Mass / Counts")
        ) + 1
    except StopIteration:
        print(f"⚠️ Kein Header 'Mass / Counts' in Datei: {filepath}")
        return None, None, None, None, [], []

    data = [line.strip().split("\t") for line in lines[start_index:] if line.strip()]
    mz = np.array([float(row[0]) for row in data])
    intensity = np.array([float(row[1]) for row in data])

    # --- Nur Daten ab m/z 15 verwenden ---
    mask = mz >= 15
    mz = mz[mask]
    intensity = intensity[mask]

    if len(mz) == 0 or len(intensity) == 0:
        print(f"⚠️ Keine Daten >= 15 m/z in Datei: {filepath}")
        return None, None, None, None, [], []

    # sicherstellen, dass mz aufsteigend ist
    order = np.argsort(mz)
    mz = mz[order]
    intensity = intensity[order]

    # Startindex für Baseline-Fit
    start_idx = np.searchsorted(mz, baseline_start)
    if start_idx >= len(mz):
        print(f"⚠️ Kein m/z >= {baseline_start} in Datei: {os.path.basename(filepath)}")
        baseline = np.zeros_like(intensity)
        corrected = intensity - baseline
        peaks_final = np.array([], dtype=int)
        peak_data = []
        return mz, intensity, baseline, corrected, peaks_final, peak_data

    # Fit-Bereich
    intensity_fit = intensity[start_idx:]
    n_fit = len(intensity_fit)

    # --- 1) Peaks im Fit-Bereich maskieren ---
    prom_thresh = peak_mask_prominence_factor * max(np.max(intensity_fit), 1.0)
    peaks_in_fit, props = find_peaks(intensity_fit, prominence=prom_thresh, distance=max(1, int(peak_distance/2)))
    peak_mask = np.zeros_like(intensity_fit, dtype=bool)
    mask_width = max(1, int(peak_window))
    for pi in peaks_in_fit:
        lo = max(0, pi - mask_width)
        hi = min(n_fit, pi + mask_width + 1)
        peak_mask[lo:hi] = True

    x_fit = np.arange(n_fit)
    good = ~peak_mask
    intensity_fit_masked = intensity_fit.copy()
    if good.sum() >= 2:
        intensity_fit_masked[~good] = np.interp(x_fit[~good], x_fit[good], intensity_fit[good])

    # --- 2) ALS-Baseline ---
    baseline_fit = Baseline(intensity_fit_masked).asls(lam=lam, p=p)[0]
    baseline = np.zeros_like(intensity)
    baseline[start_idx:] = baseline_fit
    corrected = intensity - baseline

    # Peaks auf korrigierten Daten
    peaks, _ = find_peaks(corrected, height=np.std(corrected), distance=peak_distance)
   # print(np.std(corrected))

    # Integration pro Peak
    peak_data = []
    for peak in peaks:
        start = max(peak - peak_window, 0)
        end = min(peak + peak_window + 1, len(mz))
        area = simpson(corrected[start:end], mz[start:end])
        peak_data.append({'m/z': mz[peak], 'area': area})

    # Fallback
    fallback_used = False
    if len(peaks) == 0:
        fallback_used = True
        win = max(3, int(n_fit * fallback_minfilter_fraction))
        if win % 2 == 0:
            win += 1
        minf = minimum_filter(intensity_fit, size=win)
        sg_win = min(101, win if win % 2 == 1 else win+1)
        if sg_win < 3:
            sg_win = 3
        baseline_fit_alt = savgol_filter(minf, window_length=sg_win, polyorder=2)
        baseline_alt = np.zeros_like(intensity)
        baseline_alt[start_idx:] = baseline_fit_alt
        corrected_alt = intensity - baseline_alt

        max_corr_alt = np.max(corrected_alt)
        std_corr_alt = np.std(corrected_alt)
        prom_alt = max(0.0005 * max_corr_alt, std_corr_alt*0.005)
        #height_alt = max(1e-6 * max_corr_alt, np.mean(corrected_alt) + 0.000001 * std_corr_alt)
        height_alt = 10
        #peaks_alt, props_alt = find_peaks(corrected_alt, height=height_alt, prominence=prom_alt, distance=max(1, int(peak_distance/2)))
        peaks_alt, props_alt = find_peaks(corrected_alt, height=height_alt, prominence=prom_alt, distance=1000)
        print(height_alt)
        if len(peaks_alt) > 0:
            baseline = baseline_alt
            corrected = corrected_alt
            peaks_final = peaks_alt
            peak_data = []
            for peak in peaks_final:
                start = max(peak - peak_window, 0)
                end = min(peak + peak_window + 1, len(mz))
                area = simpson(corrected[start:end], mz[start:end])
                peak_data.append({'m/z': mz[peak], 'area': area})
        else:
            peaks_final = peaks
    else:
        peaks_final = peaks

    # Debug
    print(f"File: {os.path.basename(filepath)} | mz_range: {mz.min():.2f}-{mz.max():.2f} | "
          f"fit points: {n_fit} | peaks_masked_in_fit: {len(peaks_in_fit)} | peaks_found_total: {len(peaks_final)}"
          + (f" | fallback_used" if fallback_used else ""))

    return mz, intensity, baseline, corrected, peaks_final, peak_data

# ---------------------------
# Hauptprogramm
# ---------------------------
folder = r"C:\Users\adako\Desktop\Wabenreaktor_Alacac3\Messdaten_Ilyas\8mm Wabe_Reproduzierung\2025-09-10\TR200"
files = [f for f in os.listdir(folder) if f.endswith(".asc")]
print(f"{len(files)} .asc-Dateien gefunden.")

all_peaks = []
all_spectra = []

for file in files:
    filepath = os.path.join(folder, file)
    print(f"\n== Bearbeite Datei: {file} ==")

    mz, intensity, baseline, corrected, peaks, peak_data = process_file(filepath)
    if mz is None:
        continue

    meta = extract_metadata(file)

    # Spektrum speichern
    df_spec = pd.DataFrame({
        'm/z': mz,
        'intensity_raw': intensity,
        'baseline': baseline,
        'intensity_corrected': corrected
    })
    for key, value in meta.items():
        df_spec[key] = value
    df_spec['Datei'] = file
    all_spectra.append(df_spec)

    # Peaks speichern
    for peak in peak_data:
        peak.update(meta)
        peak['Datei'] = file
        all_peaks.append(peak)

    # Plot zur Kontrolle
    plt.figure(figsize=(9, 5))
    plt.plot(mz, intensity, color='gray', alpha=0.5, label="Raw")
    plt.plot(mz, baseline, 'r--', label="Baseline")
    plt.plot(mz, corrected, 'b', label="Corrected")
    if len(peaks) > 0:
        plt.scatter(mz[peaks], corrected[peaks], color='k', marker='x', label="Peaks")
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(f"Baseline-Korrektur: {file}")
    plt.legend()
    plt.ylim(0, 3000)
    plt.tight_layout()
    plt.show()

# Ergebnisse speichern
if len(all_spectra) > 0:
    df_all_spec = pd.DataFrame(pd.concat(all_spectra, ignore_index=True))
    df_all_spec.to_csv("alle_spektren_mit_baseline.csv", index=False)
    print("✅ Spektren mit Baseline-Korrektur gespeichert in 'alle_spektren_mit_baseline.csv'")

if len(all_peaks) > 0:
    df_all_peaks = pd.DataFrame(all_peaks)
    df_all_peaks.to_csv("alle_peaks_korrigiert.csv", index=False)
    print("✅ Integrierte Peaks gespeichert in 'alle_peaks_korrigiert.csv'")
