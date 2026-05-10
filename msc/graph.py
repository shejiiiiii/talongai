import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter

# ── Data ───────────────────────────────────────────────────────────────────
df = pd.read_csv('eggplant_spectral_data_v5.csv')
df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()
df['Base_ID'] = df['Eggplant_ID'].str.replace(r'_S\d$', '', regex=True).str.strip()

ch_cols = [f'Ch_{i}' for i in range(1, 19)]
wavelengths = [410,435,460,485,510,535,560,585,610,645,680,705,730,760,810,860,900,940]

healthy_ids  = ['H01, H02', 'H03, H04', 'H05, H06']
infested_ids = ['I01, I02', 'I03, I04', 'I05, I06']
h_names = ['Healthy Sample 1 (H01–H02)', 'Healthy Sample 2 (H03–H04)', 'Healthy Sample 3 (H05–H06)']
i_names = ['Infested Sample 1 (I01–I02)', 'Infested Sample 2 (I03–I04)', 'Infested Sample 3 (I05–I06)']

def get_stats(group, base_id):
    sub = group[group['Base_ID'] == base_id][ch_cols].values.astype(float)
    return sub.mean(axis=0), sub.std(axis=0)

healthy_group  = df[df['Label'] == 'Healthy']
infested_group = df[df['Label'] == 'Infested']

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Individual Healthy Spectral Profiles (3-panel)
# ─────────────────────────────────────────────────────────────────────────────
HC = ['#2ECC71', '#27AE60', '#1A8C47']   # green shades

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
fig.patch.set_facecolor('white')

for ax, hid, hname, col in zip(axes, healthy_ids, h_names, HC):
    mean, std = get_stats(healthy_group, hid)
    ax.fill_between(wavelengths, mean - std, mean + std,
                    color=col, alpha=0.18, label='±1 SD')
    ax.plot(wavelengths, mean, color=col, lw=2.2, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=1.8,
            markeredgecolor=col, zorder=3)

    # Band shading
    ax.axvspan(410, 535, alpha=0.06, color='#4B9CD3', label='UV–Blue')
    ax.axvspan(560, 705, alpha=0.06, color='#F4C430', label='Vis–Green/Red')
    ax.axvspan(730, 940, alpha=0.06, color='#C0392B', label='NIR')

    ax.set_xlabel('Wavelength (nm)', fontsize=11, labelpad=6)
    ax.set_ylabel('Calibrated Reflectance (µW/cm²)', fontsize=9.5)
    ax.set_title(hname, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(wavelengths[::2])
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(400, 950)
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Shared legend
handles = [
    mpatches.Patch(color='#4B9CD3', alpha=0.5, label='UV–Blue (410–535 nm)'),
    mpatches.Patch(color='#F4C430', alpha=0.5, label='Vis/Green–Red (560–705 nm)'),
    mpatches.Patch(color='#C0392B', alpha=0.5, label='NIR (730–940 nm)'),
    plt.Line2D([0],[0], color='gray', alpha=0.6, linewidth=6, label='±1 SD band'),
]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, 0.01), framealpha=0.9)
fig.suptitle('Spectral Reflectance Profiles — Healthy Eggplant Samples',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig('msc/spectral_healthy.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: spectral_healthy.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Individual Infested Spectral Profiles (3-panel)
# ─────────────────────────────────────────────────────────────────────────────
IC = ['#E74C3C', '#C0392B', '#922B21']   # red shades

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
fig.patch.set_facecolor('white')

for ax, iid, iname, col in zip(axes, infested_ids, i_names, IC):
    mean, std = get_stats(infested_group, iid)
    ax.fill_between(wavelengths, mean - std, mean + std,
                    color=col, alpha=0.18, label='±1 SD')
    ax.plot(wavelengths, mean, color=col, lw=2.2, marker='s',
            markersize=5, markerfacecolor='white', markeredgewidth=1.8,
            markeredgecolor=col, zorder=3)

    ax.axvspan(410, 535, alpha=0.06, color='#4B9CD3')
    ax.axvspan(560, 705, alpha=0.06, color='#F4C430')
    ax.axvspan(730, 940, alpha=0.06, color='#C0392B')

    ax.set_xlabel('Wavelength (nm)', fontsize=11, labelpad=6)
    ax.set_ylabel('Calibrated Reflectance (µW/cm²)', fontsize=9.5)
    ax.set_title(iname, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(wavelengths[::2])
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(400, 950)
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [
    mpatches.Patch(color='#4B9CD3', alpha=0.5, label='UV–Blue (410–535 nm)'),
    mpatches.Patch(color='#F4C430', alpha=0.5, label='Vis/Green–Red (560–705 nm)'),
    mpatches.Patch(color='#C0392B', alpha=0.5, label='NIR (730–940 nm)'),
    plt.Line2D([0],[0], color='gray', alpha=0.6, linewidth=6, label='±1 SD band'),
]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, 0.01), framealpha=0.9)
fig.suptitle('Spectral Reflectance Profiles — Infested Eggplant Samples',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig('msc/spectral_infested.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: spectral_infested.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Combined Comparison: Mean ± SD, Healthy vs Infested (3 pairs)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
fig.patch.set_facecolor('white')

pair_labels = ['Sample Pair 1', 'Sample Pair 2', 'Sample Pair 3']

for ax, hid, iid, plabel in zip(axes, healthy_ids, infested_ids, pair_labels):
    hm, hs = get_stats(healthy_group, hid)
    im, is_ = get_stats(infested_group, iid)

    ax.fill_between(wavelengths, hm - hs, hm + hs, color='#2ECC71', alpha=0.20)
    ax.fill_between(wavelengths, im - is_, im + is_, color='#E74C3C', alpha=0.20)

    ax.plot(wavelengths, hm, color='#27AE60', lw=2.2, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=1.8,
            markeredgecolor='#27AE60', label='Healthy', zorder=4)
    ax.plot(wavelengths, im, color='#C0392B', lw=2.2, marker='s',
            markersize=5, markerfacecolor='white', markeredgewidth=1.8,
            markeredgecolor='#C0392B', label='Infested', zorder=4)

    ax.axvspan(410, 535, alpha=0.05, color='#4B9CD3')
    ax.axvspan(560, 705, alpha=0.05, color='#F4C430')
    ax.axvspan(730, 940, alpha=0.05, color='#C0392B')

    ax.set_xlabel('Wavelength (nm)', fontsize=11, labelpad=6)
    ax.set_ylabel('Calibrated Reflectance (µW/cm²)', fontsize=9.5)
    ax.set_title(plabel, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(wavelengths[::2])
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(400, 950)
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, frameon=True, loc='upper right')

fig.suptitle('Spectral Reflectance Comparison — Healthy vs. Infested Eggplant',
             fontsize=14, fontweight='bold', y=1.01)
handles = [
    plt.Line2D([0],[0], color='#27AE60', lw=2.2, marker='o', markersize=5,
               markerfacecolor='white', markeredgewidth=1.8, markeredgecolor='#27AE60', label='Healthy (mean)'),
    plt.Line2D([0],[0], color='#C0392B', lw=2.2, marker='s', markersize=5,
               markerfacecolor='white', markeredgewidth=1.8, markeredgecolor='#C0392B', label='Infested (mean)'),
    mpatches.Patch(color='#2ECC71', alpha=0.4, label='Healthy ±1 SD'),
    mpatches.Patch(color='#E74C3C', alpha=0.4, label='Infested ±1 SD'),
    mpatches.Patch(color='#4B9CD3', alpha=0.4, label='UV–Blue'),
    mpatches.Patch(color='#F4C430', alpha=0.4, label='Vis/Green–Red'),
    mpatches.Patch(color='#C0392B', alpha=0.3, label='NIR'),
]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig('msc/spectral_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: spectral_comparison.png")
