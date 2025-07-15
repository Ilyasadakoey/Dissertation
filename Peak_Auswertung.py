
import os
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import simpson

# process_file bleibt unverändert

def process_file(filepath):
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    try:
        start_index = next(
            i for i, line in enumerate(lines) if line.strip().startswith("Mass / Counts")
        ) + 1
    except StopIteration:
        print(f"⚠️ Kein Header 'Mass / Counts' in Datei: {filepath}")
        return None, None, None, []

    data = [line.strip().split("\t") for line in lines[start_index:] if line.strip()]
    mz = np.array([float(row[0]) for row in data])
    intensity = np.array([float(row[1]) for row in data])

    if len(mz) == 0 or len(intensity) == 0:
        print(f"⚠️ Leere Daten in Datei: {filepath}")
        return None, None, None, []

    max_intensity = np.max(intensity)
    threshold = 0.01 * max_intensity  # Du kannst hier evtl. noch anpassen
    print(f"→ Datei: {os.path.basename(filepath)} | Max: {max_intensity:.2f} | Schwelle: {threshold:.2f}")

    peaks, _ = find_peaks(intensity, height=threshold, distance=5)
    print(f"→ Gefundene Peaks: {len(peaks)}")

    window = 3
    peak_data = []
    for peak in peaks:
        start = max(peak - window, 0)
        end = min(peak + window + 1, len(mz))
        area = simpson(intensity[start:end], mz[start:end])
        peak_data.append({
            'm/z': mz[peak],
            'area': area
        })

    return mz, intensity, peaks, peak_data

def extract_metadata(filename):
    match = re.match(r"(\d{3})_(\d+)sccm_TR(\d+)", filename)
    if match:
        return {
            'Messung': int(match.group(1)),
            'sccm': int(match.group(2)),
            'Temperatur': int(match.group(3))
        }
    else:
        return {'Messung': None, 'sccm': None, 'Temperatur': None}

folder = r"C:\Users\adako\Desktop\TR400"
all_results = []

files = [f for f in os.listdir(folder) if f.endswith(".asc")]
print(f"{len(files)} .asc-Dateien gefunden: {files}")

for file in files:
    filepath = os.path.join(folder, file)
    print(f"\n== Bearbeite Datei: {file} ==")

    mz, intensity, peaks, peak_data = process_file(filepath)
    if peak_data is None or len(peak_data) == 0:
        print("⚠️ Keine Peaks gefunden oder Fehler beim Verarbeiten")
        continue

    meta = extract_metadata(file)
    # Füge Metadaten zu jedem Peak hinzu und sammle alles
    for peak in peak_data:
        peak.update(meta)
        peak['Datei'] = file
        all_results.append(peak)

# Am Ende: speichere alle Ergebnisse in CSV
if len(all_results) > 0:
    df = pd.DataFrame(all_results)
    df.to_csv("alle_integrierten_peaks.csv", index=False)
    print("\n✅ Fertig. Ergebnisse gespeichert in 'alle_integrierten_peaks.csv'.")
else:
    print("⚠️ Keine Peaks aus irgendeiner Datei gefunden. CSV wurde nicht gespeichert.")






