import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_excel('Mappe12.xlsx')

counts = {}

# Iterieren Sie durch jede Zeile des DataFrame
for index, row in df.iterrows():
    # Teilen Sie den Wert in der Spalte auf, um K und V zu erhalten
    k_v_pairs = row['Codierung '].split(';')  # Trennen Sie die Wertepaare anhand des Semikolons
    for k_v_pair in k_v_pairs:
        if ':' in k_v_pair:  # Überprüfen Sie, ob ein Doppelpunkt vorhanden ist
            k, v_str = k_v_pair.split(':')  # Teilen Sie die Wertepaare anhand des Doppelpunktes
            k = k.strip()
            values = v_str.split(',')  # Trennen Sie die Werte anhand des Kommas

            # Entfernen Sie Leerzeichen und führende/trailing Whitespaces von V
            values = [v.strip() for v in values]

            # Aktualisieren Sie die Zählungen
            if k not in counts:
                counts[k] = {}
            for v in values:
                if v not in counts[k]:
                    counts[k][v] = 0
                counts[k][v] += 1

# Drucken Sie die Ergebnisse
for k, v_counts in counts.items():
    print(f"Für {k}:")
    for v, count in v_counts.items():
        print(f"{v} kommt {count} mal vor")