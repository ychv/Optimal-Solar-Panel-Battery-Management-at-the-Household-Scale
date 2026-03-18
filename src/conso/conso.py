"""
Estimation empirique de μ(t) depuis les données UCI
=====================================================
Télécharge automatiquement le dataset UCI :
  "Individual Household Electric Power Consumption"
  Maison à Sceaux (92), France — déc. 2006 à nov. 2010
  2 075 259 mesures à 1 minute — Licence CC BY 4.0

Puis calcule et trace :
  - μ(t)        : moyenne sur toutes les journées (profil type)
  - μ_semaine   : profil moyen jours de semaine
  - μ_weekend   : profil moyen weekend
  - μ_hiver     : profil moyen hiver (déc-fév)
  - μ_été       : profil moyen été (juin-août)
  + ajustement de la formule analytique sur les données réelles

pip install numpy pandas matplotlib scipy requests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import urllib.request
import zipfile
import os

# ─── 1. Téléchargement ───────────────────────────────────────────
URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
DIR  = os.path.dirname(os.path.abspath(__file__))
ZIP  = os.path.join(DIR, "household_power_consumption.zip")
FILE = os.path.join(DIR, "household_power_consumption.txt")

if not os.path.exists(FILE):
    print("Téléchargement du dataset UCI (~20 Mo)...")
    urllib.request.urlretrieve(URL, ZIP)
    with zipfile.ZipFile(ZIP, 'r') as z:
        z.extractall(DIR)
    os.remove(ZIP)
    print("Téléchargement terminé.")
else:
    print("Fichier déjà présent, chargement...")

# ─── 2. Chargement & nettoyage ───────────────────────────────────
print("Chargement des données...")
df = pd.read_csv(
    FILE,
    sep=';',
    parse_dates={'datetime': ['Date', 'Time']},
    dayfirst=True,
    na_values='?',
    low_memory=False,
    infer_datetime_format=True,
)
df = df.set_index('datetime').sort_index()
df = df[['Global_active_power']].dropna()
df['power_W'] = df['Global_active_power'] * 1000   # kW → W

print(f"  {len(df):,} mesures chargées — {df.index[0].date()} → {df.index[-1].date()}")

# ─── 3. Calcul des profils empiriques ────────────────────────────
df['hour']    = df.index.hour + df.index.minute / 60   # heure décimale
df['minute']  = df.index.hour * 60 + df.index.minute   # minute 0-1439
df['month']   = df.index.month
df['weekday'] = df.index.weekday  # 0=lundi, 6=dimanche

# Profil global — moyenne par minute de la journée
mu_global = df.groupby('minute')['power_W'].mean()
sigma_global = df.groupby('minute')['power_W'].std()



print(f"\n=== Statistiques empiriques ===")
print(f"  Conso moyenne globale  : {mu_global.mean():.0f} W  ({mu_global.mean()*24/1000:.1f} kWh/j)")
print(f"  Pic moyen (heure)      : {mu_global.max():.0f} W  à {mu_global.idxmax()/60:.1f}h")
print(f"  Creux moyen (nuit)     : {mu_global.min():.0f} W  à {mu_global.idxmin()/60:.1f}h")

idx_fit = np.arange(0, 1440, 10)
heures   = mu_global.index / 60  # minute → heure décimale
t_fit   = heures[idx_fit]
y_fit   = mu_global.values[idx_fit]

def mu_analytique(t, C, A1, t1, s1, A2, t2, s2, A3):
    g1 = A1 * np.exp(-0.5 * ((t - t1) / s1) ** 2)
    g2 = A2 * np.exp(-0.5 * ((t - t2) / s2) ** 2)
    cy = A3 * np.cos(2 * np.pi * (t - 14) / 24)
    return np.clip(C + g1 + g2 + cy, 0, None)

p0 = [400, 400, 7.5, 1.2, 700, 20.0, 2.0, 100]

fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    "μ(t) empirique — Dataset UCI · Maison à Sceaux (92), France · 2006–2010\n"
    "2 075 259 mesures à 1 min  ·  Licence CC BY 4.0  ·  Hebrail & Berard (2012)",
    color='white', fontsize=12, fontweight='bold'
)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.42, wspace=0.28,
                       left=0.07, right=0.97, top=0.90, bottom=0.07)



# ── Graphe 1 : μ global + formule ajustée ────────────────────────
ax1 = fig.add_subplot(gs[0, :])

ax1.plot(heures, mu_global.values / 1000,
         lw=1.5, alpha=0.7, label='μ(t) empirique (données brutes)')

ax2 = ax1.twinx()
ax2.plot(heures, sigma_global.values / 1000,
            lw=1.5, alpha=0.7, color='orange', label='σ(t) empirique (écart-type)')

ax1.set_ylabel('Puissance (kW)')
ax2.set_ylabel('Écart-type (kW)')

ax1.legend()
plt.show()


# ─── Sauvegarde des résultats ────────────────────────────────────
output_path = os.path.join(DIR, "consceaux.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== Statistiques empiriques — Dataset UCI ===\n\n")
    f.write(f"Conso moyenne globale  : {mu_global.mean():.0f} W  ({mu_global.mean()*24/1000:.1f} kWh/j)\n")
    f.write(f"Pic moyen (heure)      : {mu_global.max():.0f} W  à {mu_global.idxmax()/60:.1f}h\n")
    f.write(f"Creux moyen (nuit)     : {mu_global.min():.0f} W  à {mu_global.idxmin()/60:.1f}h\n\n")

    # Optionnel : exporter le profil complet minute par minute
    f.write("=== Profil μ(t) — minute par minute ===\n")
    f.write("minute;heure;puissance_W\n")
    for minute, val in mu_global.items():
        f.write(f"{minute};{minute/60:.4f};{val:.2f}\n")

print(f"Résultats sauvegardés dans : {output_path}")


