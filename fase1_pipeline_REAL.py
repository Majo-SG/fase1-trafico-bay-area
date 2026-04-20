import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

# ── FILE PATHS ────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_SPEED     = os.path.join(DATA_DIR, "PEMS-BAY.csv")
FILE_ADJ       = os.path.join(DATA_DIR, "adj_mx_bay.pkl")
FILE_ACCIDENTS = os.path.join(DATA_DIR, "US_Accidents_March23.csv")

# ── VERIFY FILES ──────────────────────────────────────────────────
print("=" * 65)
print("  PHASE 1 — PEMS-BAY + INCIDENTS PREPROCESSING")
print("=" * 65)
print("\nVerifying files...")
all_ok = True
for f in [FILE_SPEED, FILE_ADJ, FILE_ACCIDENTS]:
    exists = os.path.exists(f)
    size   = os.path.getsize(f)/(1024*1024) if exists else 0
    status = f"OK  {size:.1f} MB" if exists else "NOT FOUND"
    print(f"  {os.path.basename(f):40s} {status}")
    if not exists:
        all_ok = False
if not all_ok:
    print("\nERROR: Missing files in data/ folder")
    exit(1)

# ── STEP 1: LOAD SPEED DATA ───────────────────────────────────────
print("\n" + "="*65)
print("STEP 1: Loading PEMS-BAY.csv...")
print("="*65)
df = pd.read_csv(FILE_SPEED, index_col=0, parse_dates=True)
print(f"  Timestamps: {df.shape[0]:,}   Sensors: {df.shape[1]}")
print(f"  Period:     {df.index.min()} -> {df.index.max()}")
print(f"  NaN:        {df.isna().sum().sum():,}")

# ── STEP 2: DATA CLEANING ─────────────────────────────────────────
print("\n" + "="*65)
print("STEP 2: Cleaning NaN values...")
print("="*65)
nan_before = df.isna().sum().sum()
df_clean   = df.ffill().bfill()  # forward fill then backward fill
nan_after  = df_clean.isna().sum().sum()
print(f"  NaN before: {nan_before:,}")
print(f"  NaN after:  {nan_after}")

# ── STEP 3: TIME SERIES RESAMPLING ───────────────────────────────
print("\n" + "="*65)
print("STEP 3: Resampling 5min -> 1 hour...")
print("="*65)
df_hourly = df_clean.resample('h').mean().round(2)
print(f"  {df_clean.shape[0]:,} rows (5min) -> {df_hourly.shape[0]:,} rows (1h)")

# ── STEP 4: ADJACENCY MATRIX ──────────────────────────────────────
print("\n" + "="*65)
print("STEP 4: Loading adj_mx_bay.pkl...")
print("="*65)
with open(FILE_ADJ, 'rb') as f:
    adj_data = pickle.load(f, encoding='latin1')
sensor_ids     = adj_data[0]
sensor_id2idx  = adj_data[1]
adj_mx         = adj_data[2]
active_weights = adj_mx[adj_mx > 0].flatten()
print(f"  Matrix shape:       {adj_mx.shape}")
print(f"  Active connections: {(adj_mx>0).sum():,}")
print(f"  Network density:    {(adj_mx>0).mean()*100:.2f}%")
print(f"  Average weight:     {active_weights.mean():.4f}")

# ── STEP 5: LOAD US ACCIDENTS ─────────────────────────────────────
print("\n" + "="*65)
print("STEP 5: Loading US_Accidents_March23.csv...")
print("        (large file ~1.1 GB, may take 1-2 minutes)")
print("="*65)

# Load only the columns we need to save memory
cols = ['ID','Severity','Start_Time','End_Time',
        'Start_Lat','Start_Lng','Distance(mi)','Description',
        'Street','City','County','State',
        'Temperature(F)','Humidity(%)','Visibility(mi)',
        'Wind_Speed(mph)','Precipitation(in)',
        'Weather_Condition','Sunrise_Sunset']

df_acc = pd.read_csv(FILE_ACCIDENTS, usecols=cols,
                     parse_dates=['Start_Time','End_Time'],
                     low_memory=False)
print(f"  Total records USA:       {len(df_acc):,}")

# Filter Bay Area: lat 37.2-38.0 / lon -122.6 to -121.8 / state CA
df_bay = df_acc[
    (df_acc['State'] == 'CA') &
    (df_acc['Start_Lat'].between(37.2, 38.0)) &
    (df_acc['Start_Lng'].between(-122.6, -121.8))
].copy()
print(f"  Bay Area CA records:     {len(df_bay):,}")

# Force datetime conversion (in case it was read as string after filtering)
df_bay['Start_Time'] = pd.to_datetime(df_bay['Start_Time'], errors='coerce')
df_bay = df_bay.dropna(subset=['Start_Time'])

# Filter same period as PEMS-BAY
start_date = df_clean.index.min().tz_localize(None)
end_date   = df_clean.index.max().tz_localize(None)
df_bay['Start_Time'] = df_bay['Start_Time'].dt.tz_localize(None)
df_bay = df_bay[
    (df_bay['Start_Time'] >= start_date) &
    (df_bay['Start_Time'] <= end_date)
].copy()
print(f"  Records in 2017 period:  {len(df_bay):,}")

if len(df_bay) == 0:
    print("\n  WARNING: No accidents found in Bay Area for that period.")
    print("  Trying with all California + available years...")
    years    = df_acc['Start_Time'].dt.year.unique()
    ref_year = min(years, key=lambda y: abs(y - 2017))
    df_bay   = df_acc[
        (df_acc['State'] == 'CA') &
        (df_acc['Start_Lat'].between(37.2, 38.0)) &
        (df_acc['Start_Lng'].between(-122.6, -121.8)) &
        (df_acc['Start_Time'].dt.year == ref_year)
    ].copy()
    print(f"  Using year {ref_year}: {len(df_bay):,} records")
    diff_days = (start_date - df_bay['Start_Time'].min()).days
    df_bay['Start_Time'] = df_bay['Start_Time'] + pd.Timedelta(days=diff_days)
    df_bay['End_Time']   = df_bay['End_Time']   + pd.Timedelta(days=diff_days)

# Round to nearest hour for alignment with PEMS-BAY
df_bay['Start_Time'] = pd.to_datetime(df_bay['Start_Time']).dt.floor('h')

print(f"\n  Severity 1 (minor):    {(df_bay.Severity==1).sum():,}")
print(f"  Severity 2 (moderate): {(df_bay.Severity==2).sum():,}")
print(f"  Severity 3 (serious):  {(df_bay.Severity==3).sum():,}")
print(f"  Severity 4 (critical): {(df_bay.Severity==4).sum():,}")

# ── STEP 6: BUILD MASTER TABLE ────────────────────────────────────
print("\n" + "="*65)
print("STEP 6: Building master table...")
print("="*65)

# Aggregate incidents by hour
inc_agg = (
    df_bay.groupby('Start_Time')
    .agg(
        incident_text    = ('Description',      lambda x: ' | '.join(x.dropna().unique()[:2])),
        max_severity     = ('Severity',          'max'),
        n_incidents      = ('Severity',          'count'),
        affected_streets = ('Street',            lambda x: ', '.join(x.dropna().unique()[:3])),
        weather          = ('Weather_Condition', lambda x: x.dropna().mode()[0] if len(x.dropna())>0 else 'Clear'),
        temperature_f    = ('Temperature(F)',    'mean'),
    )
    .round({'temperature_f': 1})
    .rename_axis('timestamp')
)

# Use central sensor as reference
sensor_ref = df_hourly.columns[len(df_hourly.columns)//2]
df_master  = df_hourly[[sensor_ref]].copy()
df_master.index.name = 'timestamp'
df_master.columns    = ['speed_mph']
df_master = df_master.join(inc_agg, how='left')

# Fill rows with no incident
df_master['incident_text']    = df_master['incident_text'].fillna('Normal traffic flow')
df_master['max_severity']     = df_master['max_severity'].fillna(0).astype(int)
df_master['n_incidents']      = df_master['n_incidents'].fillna(0).astype(int)
df_master['affected_streets'] = df_master['affected_streets'].fillna('')
df_master['weather']          = df_master['weather'].fillna('Clear')
df_master['temperature_f']    = df_master['temperature_f'].fillna(
    df_master['temperature_f'].mean()).round(1)

df_master = df_master[['speed_mph','incident_text',
                        'max_severity','n_incidents',
                        'affected_streets','weather','temperature_f']]

with_inc = (df_master['n_incidents'] > 0).sum()
no_inc   = (df_master['n_incidents'] == 0).sum()
print(f"  Total rows:       {len(df_master):,}")
print(f"  With incident:    {with_inc:,} ({with_inc/len(df_master)*100:.1f}%)")
print(f"  Without incident: {no_inc:,} ({no_inc/len(df_master)*100:.1f}%)")
print(f"\n  Master table sample:")
print(df_master.head(3).to_string())

df_master.to_csv(os.path.join(OUTPUT_DIR, "tabla_maestra.csv"))
print(f"\n  OK tabla_maestra.csv saved")

# ── CHART 1: TIME SERIES — 1 WEEK ─────────────────────────────────
print("\n" + "="*65)
print("CHART 1: Time series — 1 week with incident markers...")
print("="*65)

first_monday = df_clean.index[df_clean.index.dayofweek == 0][0]
week_end     = first_monday + pd.Timedelta(days=6)
week         = df_clean[sensor_ref].loc[first_monday:week_end]
inc_week     = df_bay[(df_bay['Start_Time'] >= first_monday) &
                       (df_bay['Start_Time'] <= week_end)]

fig, ax = plt.subplots(figsize=(16, 5.5))
ax.plot(week.index, week.values, color='#1A5FA8',
        linewidth=0.85, alpha=0.88, zorder=3, label=f'Sensor {sensor_ref}')

for d in range(7):
    day = first_monday + pd.Timedelta(days=d)
    ax.axvspan(day.replace(hour=20),
               (day+pd.Timedelta(days=1)).replace(hour=6),
               alpha=0.05, color='#222', zorder=1)
    if day.dayofweek < 5:
        ax.axvspan(day.replace(hour=7),  day.replace(hour=9),
                   alpha=0.07, color='#E8A020', zorder=1)
        ax.axvspan(day.replace(hour=16), day.replace(hour=19),
                   alpha=0.07, color='#E8A020', zorder=1)

sev_colors = {1:'#3498DB', 2:'#F39C12', 3:'#E67E22', 4:'#C0392B'}
sev_labels = {1:'Sev.1 minor', 2:'Sev.2 moderate',
              3:'Sev.3 serious', 4:'Sev.4 critical'}
plotted    = set()
for _, row in inc_week.iterrows():
    ts  = row['Start_Time']
    sev = row['Severity']
    if ts not in week.index:
        continue
    label = sev_labels[sev] if sev not in plotted else '_nolegend_'
    ax.scatter(ts, week[ts]+2, marker='v', s=60,
               color=sev_colors[sev], zorder=5, label=label, alpha=0.9)
    plotted.add(sev)

avg = week.mean()
ax.axhline(avg, color='#888', linestyle='--', lw=1.1,
           label=f'Weekly average: {avg:.1f} mph')
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6,12,18]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
ax.tick_params(axis='x', which='minor', labelsize=7, labelcolor='#999')
ax.set_xlabel(f'Week from {first_monday.strftime("%d %b")} to {week_end.strftime("%d %b %Y")}', fontsize=11)
ax.set_ylabel('Speed (mph)', fontsize=11)
ax.set_title(f'Chart 1 — Time Series · Sensor {sensor_ref}\n'
             f'PEMS-BAY · San Francisco Bay Area · 5-minute interval',
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(-2, week.max()*1.22)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.85, ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.grid(axis='x', which='major', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grafico1_serie_tiempo_sensor.png"), dpi=160, bbox_inches='tight')
plt.close()
print("  OK grafico1_serie_tiempo_sensor.png")

# ── CHART 2: ADJACENCY MATRIX HEATMAP ────────────────────────────
print("\n" + "="*65)
print("CHART 2: Adjacency matrix heatmap...")
print("="*65)

N_SHOW  = 60
adj_sub = adj_mx[:N_SHOW, :N_SHOW]
fig, axes = plt.subplots(1, 2, figsize=(17,7), gridspec_kw={'width_ratios':[2.2,1]})
sns.heatmap(adj_sub, ax=axes[0], cmap='YlOrRd',
            mask=(adj_sub==0), vmin=0, vmax=1, linewidths=0,
            cbar_kws={'label':'Weight  w = exp(-d^2/sigma^2)', 'shrink':0.82})
axes[0].set_title(f'Adjacency matrix - first {N_SHOW}x{N_SHOW} sensors\n(white = no direct connection)',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sensor (index)', fontsize=10)
axes[0].set_ylabel('Sensor (index)', fontsize=10)
axes[0].tick_params(labelsize=7)

axes[1].hist(active_weights, bins=50, color='#C0392B', alpha=0.72, edgecolor='white', linewidth=0.3)
axes[1].axvline(active_weights.mean(), color='#1A5FA8', linestyle='--',
                lw=1.6, label=f'Mean = {active_weights.mean():.3f}')
axes[1].axvline(np.median(active_weights), color='#27AE60', linestyle=':',
                lw=1.6, label=f'Median = {np.median(active_weights):.3f}')
axes[1].set_title('Weight distribution\n(active connections only)', fontsize=11)
axes[1].set_xlabel('Connection weight', fontsize=10)
axes[1].set_ylabel('Frequency', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
stats = (f"Total sensors:   {adj_mx.shape[0]}\n"
         f"Connections:     {(adj_mx>0).sum():,}\n"
         f"Density:         {(adj_mx>0).mean()*100:.1f}%\n"
         f"Avg weight:      {active_weights.mean():.4f}\n"
         f"Max weight:      {active_weights.max():.4f}")
axes[1].text(0.97, 0.97, stats, transform=axes[1].transAxes,
             va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFF0', alpha=0.8))
plt.suptitle('Chart 2 — PEMS-BAY Sensor Network\nSan Francisco Bay Area · Gaussian Kernel Weighting',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"grafico2_heatmap_adyacencia.png"), dpi=160, bbox_inches='tight')
plt.close()
print("  OK grafico2_heatmap_adyacencia.png")

# ── CHART 3: INCIDENT ANALYSIS ────────────────────────────────────
print("\n" + "="*65)
print("CHART 3: Incident distribution analysis...")
print("="*65)

fig, axes = plt.subplots(2, 2, figsize=(15,10))
fig.suptitle('Incident Analysis · US Accidents · Bay Area CA', fontsize=13, fontweight='bold', y=1.01)

df_bay['hour'] = df_bay['Start_Time'].dt.hour
hour_counts    = df_bay.groupby('hour').size()
hour_colors    = ['#E74C3C' if (7<=h<=9 or 16<=h<=19) else '#3498DB' for h in hour_counts.index]
axes[0,0].bar(hour_counts.index, hour_counts.values, color=hour_colors, alpha=0.82, width=0.8)
axes[0,0].set_xlabel('Hour of day', fontsize=10)
axes[0,0].set_ylabel('Number of incidents', fontsize=10)
axes[0,0].set_title('By hour of day (red = peak hours)', fontsize=11, fontweight='bold')
axes[0,0].set_xticks(range(0,24,2))
axes[0,0].grid(axis='y', alpha=0.3, linestyle='--')

sev_counts = df_bay['Severity'].value_counts().sort_index()
cols_sev   = ['#3498DB','#F39C12','#E67E22','#C0392B']
bars       = axes[0,1].bar(sev_counts.index, sev_counts.values,
                            color=cols_sev[:len(sev_counts)], alpha=0.85, width=0.6)
for bar, val in zip(bars, sev_counts.values):
    axes[0,1].text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+max(sev_counts)*0.01,
                   f'{val}\n({val/len(df_bay)*100:.0f}%)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0,1].set_xlabel('Severity', fontsize=10)
axes[0,1].set_ylabel('Number of incidents', fontsize=10)
axes[0,1].set_title('By severity level', fontsize=11, fontweight='bold')
axes[0,1].grid(axis='y', alpha=0.3, linestyle='--')

df_vel  = pd.DataFrame({'Speed': df_clean[sensor_ref].values,
                         'Hour': df_clean.index.hour,
                         'IsWeekend': df_clean.index.dayofweek >= 5})
weekday = df_vel[~df_vel['IsWeekend']].groupby('Hour')['Speed'].mean()
weekend = df_vel[ df_vel['IsWeekend']].groupby('Hour')['Speed'].mean()
axes[1,0].plot(weekday.index, weekday.values, color='#1A5FA8', lw=2, marker='o', ms=4, label='Weekdays (Mon-Fri)')
axes[1,0].plot(weekend.index, weekend.values, color='#27AE60', lw=2, marker='s', ms=4, linestyle='--', label='Weekends (Sat-Sun)')
axes[1,0].fill_between(weekday.index, weekday.values, weekend.values, alpha=0.07, color='gray')
axes[1,0].set_xlabel('Hour of day', fontsize=10)
axes[1,0].set_ylabel('Average speed (mph)', fontsize=10)
axes[1,0].set_title('Average speed by hour\nWeekdays vs weekends', fontsize=11, fontweight='bold')
axes[1,0].legend(fontsize=9)
axes[1,0].set_xticks(range(0,24,2))
axes[1,0].grid(alpha=0.3, linestyle='--')

df_bay['month'] = df_bay['Start_Time'].dt.month
month_names     = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',
                   6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
pivot  = df_bay.groupby(['month','Severity']).size().unstack(fill_value=0)
pivot.index = [month_names.get(m,str(m)) for m in pivot.index]
bottom = np.zeros(len(pivot))
for sev_col, color in zip(sorted(df_bay['Severity'].unique()), cols_sev):
    if sev_col in pivot.columns:
        vals = pivot[sev_col].values
        axes[1,1].bar(pivot.index, vals, bottom=bottom, color=color,
                      alpha=0.85, label=f'Sev.{sev_col}', width=0.6)
        bottom += vals
axes[1,1].set_xlabel('Month', fontsize=10)
axes[1,1].set_ylabel('Number of incidents', fontsize=10)
axes[1,1].set_title('By month and severity', fontsize=11, fontweight='bold')
axes[1,1].legend(fontsize=9, loc='upper right')
axes[1,1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"grafico3_analisis_incidentes.png"), dpi=160, bbox_inches='tight')
plt.close()
print("  OK grafico3_analisis_incidentes.png")

# ── CHART 4: SPEED VS INCIDENTS ───────────────────────────────────
print("\n" + "="*65)
print("CHART 4: Speed vs incidents relationship...")
print("="*65)

fig, axes = plt.subplots(1, 2, figsize=(15,6))
fig.suptitle('Chart 4 - Speed vs Incidents · PEMS-BAY\nCausal impact of accidents on traffic flow',
             fontsize=13, fontweight='bold', y=1.02)

groups = {0: df_master[df_master['max_severity']==0]['speed_mph'].dropna().values}
for s in sorted(df_bay['Severity'].unique()):
    v = df_master[df_master['max_severity']==s]['speed_mph'].dropna().values
    if len(v) > 0:
        groups[s] = v
label_map  = {0:'No\nincident',1:'Sev.1\nminor',2:'Sev.2',3:'Sev.3\nserious',4:'Sev.4\ncritical'}
box_data   = [groups[k] for k in sorted(groups)]
box_labels = [label_map[k] for k in sorted(groups)]
box_colors = ['#27AE60','#3498DB','#F39C12','#E67E22','#C0392B'][:len(box_data)]
bp = axes[0].boxplot(box_data, labels=box_labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='.', alpha=0.3, markersize=3))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
axes[0].set_ylabel('Traffic speed (mph)', fontsize=11)
axes[0].set_title('Speed by incident severity level', fontsize=11)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

week_plot  = df_master.iloc[:7*24]
speed_line = week_plot['speed_mph']
axes[1].fill_between(speed_line.index, speed_line.values, alpha=0.12, color='#1A5FA8')
axes[1].plot(speed_line.index, speed_line.values, color='#1A5FA8', lw=1.1, zorder=3)
sev_colors2 = {1:'#3498DB',2:'#F39C12',3:'#E67E22',4:'#C0392B'}
for ts, row in week_plot[week_plot['n_incidents']>0].iterrows():
    c = sev_colors2.get(row['max_severity'],'#888')
    axes[1].axvline(ts, color=c, alpha=0.4, lw=1.5, zorder=2)
    axes[1].scatter(ts, row['speed_mph'], color=c, s=50, zorder=4, alpha=0.9)
axes[1].xaxis.set_major_locator(mdates.DayLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
axes[1].set_xlabel('Date', fontsize=10)
axes[1].set_ylabel('Average speed (mph)', fontsize=10)
axes[1].set_title('Hourly speed + incident markers\n(colored dots = hour with accident)', fontsize=11)
axes[1].grid(alpha=0.25, linestyle='--')
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([0],[0],marker='o',color='w',
               markerfacecolor=c,markersize=9,label=f'Sev.{s}')
               for s,c in sev_colors2.items()], fontsize=8.5, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"grafico4_velocidad_vs_incidentes.png"), dpi=160, bbox_inches='tight')
plt.close()
print("  OK grafico4_velocidad_vs_incidentes.png")

# ── SUMMARY ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PHASE 1 COMPLETED")
print("="*65)
print(f"\n  Files in outputs/:")
for file in sorted(os.listdir(OUTPUT_DIR)):
    fp   = os.path.join(OUTPUT_DIR, file)
    size = os.path.getsize(fp)/1024
    print(f"    {file:50s}  {size:7.1f} KB")
print(f"""
  Processed data summary:
    PEMS-BAY sensors:    {df_clean.shape[1]}
    Timestamps (5min):   {df_clean.shape[0]:,}
    Timestamps (hourly): {df_hourly.shape[0]:,}
    NaN removed:         {nan_before:,}
    Bay Area incidents:  {len(df_bay):,}
    Master table rows:   {len(df_master):,}
""")
