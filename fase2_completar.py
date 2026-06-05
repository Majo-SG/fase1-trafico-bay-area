import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import KBinsDiscretizer
from pathlib import Path

CSV_PATH = Path("fase1-trafico-bay-area/outputs/tabla_maestra.csv")
PT_PATH  = Path("fase2-embedding-&-discretizing/outputs/vectores_accidentes.pt")
OUT_DIR  = Path("fase3-outputs")
OUT_DIR.mkdir(exist_ok=True)

print("Cargando tabla_maestra.csv...")
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
T = len(df)
print(f"  Timestamps: {T}, Columnas: {df.columns.tolist()}")

print("Tokenizando velocidad...")
speed_values = df["speed_mph"].fillna(df["speed_mph"].median()).values.reshape(-1, 1)
kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
kbd.fit(speed_values)
speed_tokens = kbd.transform(speed_values).astype(np.int32).flatten()
np.save(OUT_DIR / "tokenized_speed.npy", speed_tokens)
print(f"  tokenized_speed.npy guardado {speed_tokens.shape}")

config = {"n_bins": 10, "strategy": "quantile", "bin_edges": kbd.bin_edges_[0].tolist(), "n_timestamps": T}
with open(OUT_DIR / "tokenizer_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("  tokenizer_config.json guardado")

print("Cargando embeddings...")
sem_np = torch.load(PT_PATH, map_location="cpu").numpy().astype(np.float32)
assert sem_np.shape[0] == T, f"Shape mismatch: {sem_np.shape[0]} vs {T}"
np.save(OUT_DIR / "semantic_embeddings.npy", sem_np)
print(f"  semantic_embeddings.npy guardado {sem_np.shape}")

print("Creando secuencias multimodales...")
multimodal = np.concatenate([speed_tokens.astype(np.float32).reshape(-1,1), sem_np], axis=1)
np.save(OUT_DIR / "multimodal_sequences.npy", multimodal)
print(f"  multimodal_sequences.npy guardado {multimodal.shape}")

print("\n=== LISTO ===")
print(f"tokenized_speed:      {speed_tokens.shape}")
print(f"semantic_embeddings:  {sem_np.shape}")
print(f"multimodal_sequences: {multimodal.shape}")
