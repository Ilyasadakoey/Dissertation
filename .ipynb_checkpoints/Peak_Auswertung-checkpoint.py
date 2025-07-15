#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import simpson

# Datei einlesen, Start nach "Mass / Counts"
with open("017_210sccm_TR400_MS.asc", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Finde Startindex
start_index = next(i for i, line in enumerate(lines) if "Mass / Counts" in line) + 1

# Lade relevante Daten
data = [line.strip().split("\t") for line in lines[start_index:] if line.strip()]
mz = np.array([float(row[0]) for row in data])
intensity = np.array([float(row[1]) for row in data])

#Signal glätten
intensity_smooth = savgol_filter(intensity, window_length = 11, polyorder = 3)


# Peaks finden
peaks, _ = find_peaks(intensity_smooth, height=10, distance=1000)

# Peaks integrieren
window = 3
peak_data = []
for peak in peaks:
    start = max(peak - window, 0)
    end = min(peak + window + 1, len(mz))
    area = simpson(intensity[start:end], mz[start:end])
    peak_data.append({'m/z': mz[peak], 'area': area})

peak_df = pd.DataFrame(peak_data)
display(peak_df)

# Plot
#plt.figure(figsize=(12, 5))
#plt.plot(mz, intensity, label="Massenspektrum")
#plt.plot(mz[peaks], intensity[peaks], "rx", label="Gefundene Peaks")
#for d in peak_data:
 #   plt.text(d['m/z'], max(intensity)*0.05 + intensity[mz.tolist().index(d['m/z'])],
  #           f"{d['m/z']:.1f}\nA={d['area']:.1f}", ha='center', fontsize=8)
#plt.xlabel("m/z")
#plt.ylabel("Intensität")
#plt.title("Massenspektrum mit integrierten Peaks")
#plt.legend()
#plt.tight_layout()
#plt.show()



# Speichern
peak_df.to_csv("integrierte_peaks.csv", index=False)


# In[3]:


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


# In[ ]:




