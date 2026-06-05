"""
fase3_llm_lora_tinyllama.py
============================
Fase 3 - Fine-tuning multimodal con TinyLlama-1.1B + LoRA
Hardware: Apple Silicon (MPS), ~17GB RAM unificada
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
import csv
from datetime import datetime

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────
OUT_DIR     = Path("fase3-outputs")
MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
SEQ_LEN     = 24       # ventana de 24 horas
BATCH_SIZE  = 2
EPOCHS      = 3
LR          = 2e-4
LORA_R      = 8
LORA_ALPHA  = 16

print(f"Device: {DEVICE}")
print(f"Configuración: SEQ_LEN={SEQ_LEN}, BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}")

# ─────────────────────────────────────────────
# 1. Cargar datos
# ─────────────────────────────────────────────
print("\nCargando artifacts de Fase 2...")
speed_tokens = np.load(OUT_DIR / "tokenized_speed.npy")        # (4344,)
sem_emb      = np.load(OUT_DIR / "semantic_embeddings.npy")    # (4344, 768)
multimodal   = np.load(OUT_DIR / "multimodal_sequences.npy")   # (4344, 769)

T = len(speed_tokens)
print(f"  Timestamps totales: {T}")

# ─────────────────────────────────────────────
# 2. Split cronológico 70/10/20
# ─────────────────────────────────────────────
train_end = int(T * 0.70)
val_end   = int(T * 0.80)

train_data = multimodal[:train_end]
val_data   = multimodal[train_end:val_end]
test_data  = multimodal[val_end:]

train_tokens = speed_tokens[:train_end]
val_tokens   = speed_tokens[train_end:val_end]

print(f"  Train: {train_end} | Val: {val_end-train_end} | Test: {T-val_end}")

# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
class TrafficDataset(Dataset):
    def __init__(self, multimodal_data, token_data, seq_len):
        self.data   = torch.tensor(multimodal_data, dtype=torch.float32)
        self.tokens = torch.tensor(token_data, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]          # (SEQ_LEN, 769)
        y = self.tokens[idx + 1 : idx + self.seq_len + 1] # (SEQ_LEN,)
        return x, y

train_dataset = TrafficDataset(train_data, train_tokens, SEQ_LEN)
val_dataset   = TrafficDataset(val_data,   val_tokens,   SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ─────────────────────────────────────────────
# 4. Cargar TinyLlama + LoRA
# ─────────────────────────────────────────────
print(f"\nCargando {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(DEVICE)

HIDDEN_DIM = model.config.hidden_size  # 2048 en TinyLlama

# ─────────────────────────────────────────────
# 5. Proyección multimodal (769 → 2048)
# ─────────────────────────────────────────────
class MultimodalProjection(nn.Module):
    def __init__(self, input_dim=769, hidden_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x):
        return self.proj(x)

projector = MultimodalProjection(769, HIDDEN_DIM).to(DEVICE)

# ─────────────────────────────────────────────
# 6. Entrenamiento
# ─────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(projector.parameters()),
    lr=LR, weight_decay=0.01
)

total_steps = len(train_loader) * EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
criterion = nn.CrossEntropyLoss()

log_path = OUT_DIR / "training_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "step", "train_loss", "val_loss", "timestamp"])

print("\nIniciando entrenamiento...")
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    projector.train()
    total_train_loss = 0
    steps = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)   # (B, SEQ_LEN, 769)
        y = y.to(DEVICE)   # (B, SEQ_LEN)

        # Proyectar a espacio de embeddings del LLM
        B, S, D = x.shape
        x_flat = x.view(B * S, D)
        projected = projector(x_flat).view(B, S, HIDDEN_DIM)  # (B, SEQ_LEN, 2048)

        # Forward pass
        outputs = model(inputs_embeds=projected, labels=y)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(projector.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        steps += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Step {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / steps

    # Validación
    model.eval()
    projector.eval()
    total_val_loss = 0
    val_steps = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            B, S, D = x.shape
            projected = projector(x.view(B * S, D)).view(B, S, HIDDEN_DIM)
            outputs = model(inputs_embeds=projected, labels=y)
            total_val_loss += outputs.loss.item()
            val_steps += 1

    avg_val_loss = total_val_loss / val_steps
    print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

    # Log
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, steps, avg_train_loss, avg_val_loss, datetime.now()])

    # Guardar mejor modelo
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(OUT_DIR / "fase3_lora_weights")
        torch.save(projector.state_dict(), OUT_DIR / "projector_weights.pt")
        print(f"  Mejor modelo guardado (val_loss={best_val_loss:.4f})")

# ─────────────────────────────────────────────
# 7. Predicciones en test
# ─────────────────────────────────────────────
print("\nGenerando predicciones en test set...")
model.eval()
projector.eval()

test_tensor = torch.tensor(test_data, dtype=torch.float32).to(DEVICE)
predictions = []

with torch.no_grad():
    for i in range(0, len(test_tensor) - SEQ_LEN, SEQ_LEN):
        x = test_tensor[i:i+SEQ_LEN].unsqueeze(0)  # (1, SEQ_LEN, 769)
        B, S, D = x.shape
        projected = projector(x.view(B*S, D)).view(B, S, HIDDEN_DIM)
        outputs = model(inputs_embeds=projected)
        logits = outputs.logits  # (1, SEQ_LEN, vocab_size)
        preds = logits.argmax(dim=-1).squeeze().cpu().numpy()
        predictions.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])

predictions = np.array(predictions)
np.save(OUT_DIR / "fase3_predictions.npy", predictions)
print(f"  Predicciones guardadas: {predictions.shape}")

print("\n=== ENTRENAMIENTO COMPLETO ===")
print(f"Mejor val_loss: {best_val_loss:.4f}")
print(f"Outputs en: {OUT_DIR}/")
