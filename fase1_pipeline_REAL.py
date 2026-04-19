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

DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_SPEED     = os.path.join(DATA_DIR, "PEMS-BAY.csv")
FILE_ADJ       = os.path.join(DATA_DIR, "adj_mx_bay.pkl")
FILE_ACCIDENTS = os.path.join(DATA_DIR, "US_Accidents_March23.csv")

# ── VERIFICAR ARCHIVOS ────────────────────────────────────────────
print("=" * 65)
print("  FASE 1 — PREPROCESAMIENTO PEMS-BAY + INCIDENTES")
print("=" * 65)
print("\nVerificando archivos...")
todos_ok = True
for f in [FILE_SPEED, FILE_ADJ, FILE_ACCIDENTS]:
    existe = os.path.exists(f)
    size   = os.path.getsize(f)/(1024*1024) if existe else 0
    estado = f"OK  {size:.1f} MB" if existe else "NO ENCONTRADO"
    print(f"  {os.path.basename(f):40s} {estado}")
    if not existe:
        todos_ok = False
if not todos_ok:
    print("\nERROR: Faltan archivos en la carpeta data/")
    exit(1)

# ── PASO 1: CARGAR VELOCIDADES ────────────────────────────────────
print("\n" + "="*65)
print("PASO 1: Cargando PEMS-BAY.csv...")
print("="*65)
df = pd.read_csv(FILE_SPEED, index_col=0, parse_dates=True)
print(f"  Timestamps: {df.shape[0]:,}   Sensores: {df.shape[1]}")
print(f"  Período:    {df.index.min()} → {df.index.max()}")
print(f"  NaN:        {df.isna().sum().sum():,}")

# ── PASO 2: LIMPIEZA ──────────────────────────────────────────────
print("\n" + "="*65)
print("PASO 2: Limpieza de NaN...")
print("="*65)
nan_antes  = df.isna().sum().sum()
df_clean   = df.ffill().bfill()
nan_despues = df_clean.isna().sum().sum()
print(f"  NaN antes:   {nan_antes:,}")
print(f"  NaN después: {nan_despues}")

# ── PASO 3: RESAMPLING ────────────────────────────────────────────
print("\n" + "="*65)
print("PASO 3: Resampling 5min → 1 hora...")
print("="*65)
df_hourly = df_clean.resample('h').mean().round(2)
print(f"  {df_clean.shape[0]:,} filas (5min) → {df_hourly.shape[0]:,} filas (1h)")

# ── PASO 4: MATRIZ DE ADYACENCIA ──────────────────────────────────
print("\n" + "="*65)
print("PASO 4: Cargando adj_mx_bay.pkl...")
print("="*65)
with open(FILE_ADJ, 'rb') as f:
    adj_data = pickle.load(f, encoding='latin1')
sensor_ids    = adj_data[0]
sensor_id2idx = adj_data[1]
adj_mx        = adj_data[2]
pesos_nz = adj_mx[adj_mx > 0].flatten()
print(f"  Dimensión:          {adj_mx.shape}")
print(f"  Conexiones activas: {(adj_mx>0).sum():,}")
print(f"  Densidad:           {(adj_mx>0).mean()*100:.2f}%")
print(f"  Peso promedio:      {pesos_nz.mean():.4f}")

# ── PASO 5: US ACCIDENTS ──────────────────────────────────────────
print("\n" + "="*65)
print("PASO 5: Cargando US_Accidents_March23.csv...")
print("        (archivo ~1.1 GB, puede tardar 1-2 min)")
print("="*65)

cols = ['ID','Severity','Start_Time','End_Time',
        'Start_Lat','Start_Lng','Distance(mi)','Description',
        'Street','City','County','State',
        'Temperature(F)','Humidity(%)','Visibility(mi)',
        'Wind_Speed(mph)','Precipitation(in)',
        'Weather_Condition','Sunrise_Sunset']

df_acc = pd.read_csv(FILE_ACCIDENTS, usecols=cols,
                     parse_dates=['Start_Time','End_Time'],
                     low_memory=False)
print(f"  Registros totales USA:     {len(df_acc):,}")

# Filtrar Bay Area: lat 37.2–38.0 / lon -122.6 a -121.8 / estado CA
df_bay = df_acc[
    (df_acc['State'] == 'CA') &
    (df_acc['Start_Lat'].between(37.2, 38.0)) &
    (df_acc['Start_Lng'].between(-122.6, -121.8))
].copy()
print(f"  Registros Bay Area CA:     {len(df_bay):,}")

# Forzar conversión a datetime (por si quedó como string al filtrar)
df_bay['Start_Time'] = pd.to_datetime(df_bay['Start_Time'], errors='coerce')
df_bay = df_bay.dropna(subset=['Start_Time'])

# Filtrar mismo período que PEMS-BAY
fecha_ini = df_clean.index.min().tz_localize(None)
fecha_fin = df_clean.index.max().tz_localize(None)
df_bay['Start_Time'] = df_bay['Start_Time'].dt.tz_localize(None)
df_bay = df_bay[
    (df_bay['Start_Time'] >= fecha_ini) &
    (df_bay['Start_Time'] <= fecha_fin)
].copy()
print(f"  Registros en período 2017: {len(df_bay):,}")

if len(df_bay) == 0:
    print("\n  AVISO: No hay accidentes en Bay Area para ese período.")
    print("  Probando con todo California + años disponibles...")
    anios = df_acc['Start_Time'].dt.year.unique()
    print(f"  Años disponibles en el dataset: {sorted(anios)}")
    # Tomar el año más cercano disponible
    anio_ref = min(anios, key=lambda y: abs(y - 2017))
    df_bay = df_acc[
        (df_acc['State'] == 'CA') &
        (df_acc['Start_Lat'].between(37.2, 38.0)) &
        (df_acc['Start_Lng'].between(-122.6, -121.8)) &
        (df_acc['Start_Time'].dt.year == anio_ref)
    ].copy()
    print(f"  Usando año {anio_ref}: {len(df_bay):,} registros")
    # Ajustar fechas para que coincidan con PEMS-BAY
    diff_dias = (fecha_ini - df_bay['Start_Time'].min()).days
    df_bay['Start_Time'] = df_bay['Start_Time'] + pd.Timedelta(days=diff_dias)
    df_bay['End_Time']   = df_bay['End_Time']   + pd.Timedelta(days=diff_dias)

df_bay['Start_Time'] = pd.to_datetime(df_bay['Start_Time']).dt.floor('h')

print(f"\n  Severidad 1 (leve):     {(df_bay.Severity==1).sum():,}")
print(f"  Severidad 2 (moderado): {(df_bay.Severity==2).sum():,}")
print(f"  Severidad 3 (grave):    {(df_bay.Severity==3).sum():,}")
print(f"  Severidad 4 (muy grav): {(df_bay.Severity==4).sum():,}")

# ── PASO 6: TABLA MAESTRA ─────────────────────────────────────────
print("\n" + "="*65)
print("PASO 6: Construyendo tabla maestra...")
print("="*65)

inc_agg = (
    df_bay.groupby('Start_Time')
    .agg(
        Texto_Incidente  = ('Description',      lambda x: ' | '.join(x.dropna().unique()[:2])),
        Severidad_Max    = ('Severity',          'max'),
        N_Incidentes     = ('Severity',          'count'),
        Calles_Afectadas = ('Street',            lambda x: ', '.join(x.dropna().unique()[:3])),
        Clima            = ('Weather_Condition', lambda x: x.dropna().mode()[0] if len(x.dropna())>0 else 'Clear'),
        Temperatura_F    = ('Temperature(F)',    'mean'),
    )
    .round({'Temperatura_F': 1})
    .rename_axis('Fecha_Hora')
)

sensor_ref = df_hourly.columns[len(df_hourly.columns)//2]
df_master  = df_hourly[[sensor_ref]].copy()
df_master.index.name = 'Fecha_Hora'
df_master.columns    = ['Velocidad_mph']
df_master = df_master.join(inc_agg, how='left')

df_master['Texto_Incidente']  = df_master['Texto_Incidente'].fillna('Normal traffic flow')
df_master['Severidad_Max']    = df_master['Severidad_Max'].fillna(0).astype(int)
df_master['N_Incidentes']     = df_master['N_Incidentes'].fillna(0).astype(int)
df_master['Calles_Afectadas'] = df_master['Calles_Afectadas'].fillna('')
df_master['Clima']            = df_master['Clima'].fillna('Clear')
df_master['Temperatura_F']    = df_master['Temperatura_F'].fillna(
    df_master['Temperatura_F'].mean()).round(1)
df_master = df_master[['Velocidad_mph','Texto_Incidente',
                        'Severidad_Max','N_Incidentes',
                        'Calles_Afectadas','Clima','Temperatura_F']]

con_inc = (df_master['N_Incidentes'] > 0).sum()
sin_inc = (df_master['N_Incidentes'] == 0).sum()
print(f"  Total filas:    {len(df_master):,}")
print(f"  Con incidente:  {con_inc:,} ({con_inc/len(df_master)*100:.1f}%)")
print(f"  Sin incidente:  {sin_inc:,} ({sin_inc/len(df_master)*100:.1f}%)")
print(f"\n  Muestra:")
print(df_master.head(3).to_string())

df_master.to_csv(os.path.join(OUTPUT_DIR, "tabla_maestra.csv"))
print(f"\n  OK tabla_maestra.csv guardada")

# ── GRÁFICO 1: SERIE DE TIEMPO ────────────────────────────────────
print("\n" + "="*65)
print("GRÁFICO 1: Serie de tiempo 1 semana...")
print("="*65)

primer_lunes = df_clean.index[df_clean.index.dayofweek == 0][0]
semana_fin   = primer_lunes + pd.Timedelta(days=6)
semana       = df_clean[sensor_ref].loc[primer_lunes:semana_fin]
inc_sem      = df_bay[(df_bay['Start_Time'] >= primer_lunes) &
                       (df_bay['Start_Time'] <= semana_fin)]

fig, ax = plt.subplots(figsize=(16, 5.5))
ax.plot(semana.index, semana.values, color='#1A5FA8',
        linewidth=0.85, alpha=0.88, zorder=3, label=f'Sensor {sensor_ref}')

for d in range(7):
    dia = primer_lunes + pd.Timedelta(days=d)
    ax.axvspan(dia.replace(hour=20),
               (dia+pd.Timedelta(days=1)).replace(hour=6),
               alpha=0.05, color='#222', zorder=1)
    if dia.dayofweek < 5:
        ax.axvspan(dia.replace(hour=7),  dia.replace(hour=9),
                   alpha=0.07, color='#E8A020', zorder=1)
        ax.axvspan(dia.replace(hour=16), dia.replace(hour=19),
                   alpha=0.07, color='#E8A020', zorder=1)

colores_sev = {1:'#3498DB', 2:'#F39C12', 3:'#E67E22', 4:'#C0392B'}
etiquetas   = {1:'Sev.1 leve', 2:'Sev.2 moderado',
               3:'Sev.3 grave', 4:'Sev.4 muy grave'}
ya_puestos  = set()
for _, row in inc_sem.iterrows():
    ts  = row['Start_Time']
    sev = row['Severity']
    if ts not in semana.index:
        continue
    lbl = etiquetas[sev] if sev not in ya_puestos else '_nolegend_'
    ax.scatter(ts, semana[ts]+2, marker='v', s=60,
               color=colores_sev[sev], zorder=5, label=lbl, alpha=0.9)
    ya_puestos.add(sev)

prom = semana.mean()
ax.axhline(prom, color='#888', linestyle='--', lw=1.1,
           label=f'Promedio: {prom:.1f} mph')
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6,12,18]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
ax.tick_params(axis='x', which='minor', labelsize=7, labelcolor='#999')
ax.set_xlabel(f'Semana del {primer_lunes.strftime("%d %b")} al {semana_fin.strftime("%d %b %Y")}', fontsize=11)
ax.set_ylabel('Velocidad (mph)', fontsize=11)
ax.set_title(f'Gráfico 1 — Serie de Tiempo · Sensor {sensor_ref}\n'
             f'PEMS-BAY · Bay Area, California · Intervalo: 5 minutos',
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(-2, semana.max()*1.22)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.85, ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.grid(axis='x', which='major', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grafico1_serie_tiempo_sensor.png"), dpi=160, bbox_inches='tight')
plt.close()
print("  OK grafico1_serie_tiempo_sensor.png")

# ── GRÁFICO 2: HEATMAP ADYACENCIA ────────────────────────────────
print("\n" + "="*65)
print("GRÁFICO 2: Heatmap matriz de adyacencia...")
print("="*65)

N_SHOW  = 60
adj_sub = adj_mx[:N_SHOW, :N_SHOW]
fig, axes = plt.subplots(1, 2, figsize=(17,7),
                          gridspec_kw={'width_ratios':[2.2,1]})
sns.heatmap(adj_sub, ax=axes[0], cmap='YlOrRd',
            mask=(adj_sub==0), vmin=0, vmax=1, linewidths=0,
            cbar_kws={'label':'Peso  w = exp(−d²/σ²)', 'shrink':0.82})
axes[0].set_title(f'Matriz de adyacencia — primeros {N_SHOW}×{N_SHOW} sensores\n'
                  '(blanco = sin conexión directa)',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sensor (índice)', fontsize=10)
axes[0].set_ylabel('Sensor (índice)', fontsize=10)
axes[0].tick_params(labelsize=7)

axes[1].hist(pesos_nz, bins=50, color='#C0392B', alpha=0.72,
             edgecolor='white', linewidth=0.3)
axes[1].axvline(pesos_nz.mean(), color='#1A5FA8', linestyle='--',
                lw=1.6, label=f'Media = {pesos_nz.mean():.3f}')
axes[1].axvline(np.median(pesos_nz), color='#27AE60', linestyle=':',
                lw=1.6, label=f'Mediana = {np.median(pesos_nz):.3f}')
axes[1].set_title('Distribución de pesos\n(conexiones activas)', fontsize=11)
axes[1].set_xlabel('Peso de conexión', fontsize=10)
axes[1].set_ylabel('Frecuencia', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
stats = (f"Total sensores:  {adj_mx.shape[0]}\n"
         f"Conexiones:      {(adj_mx>0).sum():,}\n"
         f"Densidad:        {(adj_mx>0).mean()*100:.1f}%\n"
         f"Peso promedio:   {pesos_nz.mean():.4f}\n"
         f"Peso máximo:     {pesos_nz.max():.4f}")
axes[1].text(0.97, 0.97, stats, transform=axes[1].transAxes,
             va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFF0', alpha=0.8))
plt.suptitle('Gráfico 2 — Red de Sensores PEMS-BAY\n'
             'Bay Area, California · Gaussian Kernel Weighting',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"grafico2_heatmap_adyacencia.png"), dpi=160, bbox_inches='tight')
plt.close()
print("  OK grafico2_heatmap_adyacencia.png")

# ── GRÁFICO 3: ANÁLISIS INCIDENTES ───────────────────────────────
print("\n" + "="*65)
print("GRÁFICO 3: Análisis de incidentes...")
print("="*65)

fig, axes = plt.subplots(2, 2, figsize=(15,10))
fig.suptitle('Análisis de Incidentes · US Accidents · Bay Area CA',
             fontsize=13, fontweight='bold', y=1.01)

df_bay['hora'] = df_bay['Start_Time'].dt.hour
hora_counts    = df_bay.groupby('hora').size()
cols_hora      = ['#E74C3C' if (7<=h<=9 or 16<=h<=19) else '#3498DB'
                  for h in hora_counts.index]
axes[0,0].bar(hora_counts.index, hora_counts.values, color=cols_hora, alpha=0.82, width=0.8)
axes[0,0].set_xlabel('Hora del día', fontsize=10)
axes[0,0].set_ylabel('Número de incidentes', fontsize=10)
axes[0,0].set_title('Por hora del día (rojo=horas pico)', fontsize=11, fontweight='bold')
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
axes[0,1].set_xlabel('Severidad', fontsize=10)
axes[0,1].set_ylabel('Número de incidentes', fontsize=10)
axes[0,1].set_title('Por severidad', fontsize=11, fontweight='bold')
axes[0,1].grid(axis='y', alpha=0.3, linestyle='--')

df_vel = pd.DataFrame({'Velocidad': df_clean[sensor_ref].values,
                        'Hora': df_clean.index.hour,
                        'FinDeSemana': df_clean.index.dayofweek >= 5})
sem  = df_vel[~df_vel['FinDeSemana']].groupby('Hora')['Velocidad'].mean()
wknd = df_vel[ df_vel['FinDeSemana']].groupby('Hora')['Velocidad'].mean()
axes[1,0].plot(sem.index,  sem.values,  color='#1A5FA8', lw=2,
               marker='o', ms=4, label='Laborales (Lun–Vie)')
axes[1,0].plot(wknd.index, wknd.values, color='#27AE60', lw=2,
               marker='s', ms=4, linestyle='--', label='Fin de semana')
axes[1,0].fill_between(sem.index, sem.values, wknd.values, alpha=0.07, color='gray')
axes[1,0].set_xlabel('Hora del día', fontsize=10)
axes[1,0].set_ylabel('Velocidad promedio (mph)', fontsize=10)
axes[1,0].set_title('Velocidad por hora\nLaborales vs fin de semana', fontsize=11, fontweight='bold')
axes[1,0].legend(fontsize=9)
axes[1,0].set_xticks(range(0,24,2))
axes[1,0].grid(alpha=0.3, linestyle='--')

df_bay['mes'] = df_bay['Start_Time'].dt.month
nombres_mes   = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',
                 6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
pivot  = df_bay.groupby(['mes','Severity']).size().unstack(fill_value=0)
pivot.index = [nombres_mes.get(m,str(m)) for m in pivot.index]
bottom = np.zeros(len(pivot))
for sev_col, color in zip(sorted(df_bay['Severity'].unique()), cols_sev):
    if sev_col in pivot.columns:
        vals = pivot[sev_col].values
        axes[1,1].bar(pivot.index, vals, bottom=bottom, color=color,
                      alpha=0.85, label=f'Sev.{sev_col}', width=0.6)
        bottom += vals
axes[1,1].set_xlabel('Mes', fontsize=10)
axes[1,1].set_ylabel('Número de incidentes', fontsize=10)
axes[1,1].set_title('Por mes y severidad', fontsize=11, fontweight='bold')
axes[1,1].legend(fontsize=9, loc='upper right')
axes[1,1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"grafico3_analisis_incidentes.png"), dpi=160, bbox_inches='tight')
plt.close()
print("  OK grafico3_analisis_incidentes.png")

# ── GRÁFICO 4: VELOCIDAD vs INCIDENTES ───────────────────────────
print("\n" + "="*65)
print("GRÁFICO 4: Velocidad vs incidentes...")
print("="*65)

fig, axes = plt.subplots(1, 2, figsize=(15,6))
fig.suptitle('Gráfico 4 — Relación Velocidad vs Incidentes · PEMS-BAY',
             fontsize=13, fontweight='bold', y=1.02)

grupos = {0: df_master[df_master['Severidad_Max']==0]['Velocidad_mph'].dropna().values}
for s in sorted(df_bay['Severity'].unique()):
    v = df_master[df_master['Severidad_Max']==s]['Velocidad_mph'].dropna().values
    if len(v) > 0:
        grupos[s] = v
etiq_map = {0:'Sin\nincidente',1:'Sev.1\nleve',
            2:'Sev.2',3:'Sev.3\ngrave',4:'Sev.4\nmuy grave'}
datos_bp = [grupos[k] for k in sorted(grupos)]
etiq_bp  = [etiq_map[k] for k in sorted(grupos)]
cols_bp  = ['#27AE60','#3498DB','#F39C12','#E67E22','#C0392B'][:len(datos_bp)]
bp = axes[0].boxplot(datos_bp, labels=etiq_bp, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='.', alpha=0.3, markersize=3))
for patch, color in zip(bp['boxes'], cols_bp):
    patch.set_facecolor(color); patch.set_alpha(0.75)
axes[0].set_ylabel('Velocidad del tráfico (mph)', fontsize=11)
axes[0].set_title('Velocidad por severidad del incidente', fontsize=11)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

semana_plot = df_master.iloc[:7*24]
vel_line    = semana_plot['Velocidad_mph']
axes[1].fill_between(vel_line.index, vel_line.values, alpha=0.12, color='#1A5FA8')
axes[1].plot(vel_line.index, vel_line.values, color='#1A5FA8', lw=1.1, zorder=3)
cols_sev2 = {1:'#3498DB',2:'#F39C12',3:'#E67E22',4:'#C0392B'}
for ts, row in semana_plot[semana_plot['N_Incidentes']>0].iterrows():
    c = cols_sev2.get(row['Severidad_Max'],'#888')
    axes[1].axvline(ts, color=c, alpha=0.4, lw=1.5, zorder=2)
    axes[1].scatter(ts, row['Velocidad_mph'], color=c, s=50, zorder=4, alpha=0.9)
axes[1].xaxis.set_major_locator(mdates.DayLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
axes[1].set_xlabel('Fecha', fontsize=10)
axes[1].set_ylabel('Velocidad promedio (mph)', fontsize=10)
axes[1].set_title('Velocidad horaria + marcadores de incidentes', fontsize=11)
axes[1].grid(alpha=0.25, linestyle='--')
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([0],[0],marker='o',color='w',
               markerfacecolor=c,markersize=9,label=f'Sev.{s}')
               for s,c in cols_sev2.items()], fontsize=8.5, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"grafico4_velocidad_vs_incidentes.png"), dpi=160, bbox_inches='tight')
plt.close()
print("  OK grafico4_velocidad_vs_incidentes.png")

# ── RESUMEN ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("  FASE 1 COMPLETADA")
print("="*65)
print(f"\n  Archivos en outputs/:")
for archivo in sorted(os.listdir(OUTPUT_DIR)):
    fp   = os.path.join(OUTPUT_DIR, archivo)
    size = os.path.getsize(fp)/1024
    print(f"    {archivo:50s}  {size:7.1f} KB")
print(f"""
  Datos procesados:
    Sensores PEMS-BAY:   {df_clean.shape[1]}
    Timestamps 5min:     {df_clean.shape[0]:,}
    Timestamps horarios: {df_hourly.shape[0]:,}
    NaN eliminados:      {nan_antes:,}
    Incidentes Bay Area: {len(df_bay):,}
    Filas tabla maestra: {len(df_master):,}
""")
