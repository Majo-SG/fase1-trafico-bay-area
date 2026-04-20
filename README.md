# Descripción del Proyecto

## Título
**Modelo Unificado de Predicción de Congestión e Incidentes de Tráfico mediante Arquitectura Multimodal Espacio-Temporal basada en LLMs**

## Problemática

Actualmente existe una desconexión fundamental en los sistemas de predicción de tráfico:

- Los modelos existentes predicen **variables numéricas** (velocidad, flujo) usando redes neuronales espacio-temporales
- O bien analizan **texto de incidentes** usando modelos de lenguaje natural
- Pero **ningún modelo procesa ambas modalidades como una secuencia unificada**

Esto limita la capacidad predictiva porque un accidente no es solo un número — es un evento con contexto semántico (*"Multi-car crash on I-580, major delays expected"*) que afecta directamente los valores numéricos de velocidad en los sensores cercanos minutos después.

## Solución Propuesta

Una arquitectura multimodal espacio-temporal basada en tokens donde cada par tiempo-ubicación se representa como un **"token-paquete"** que contiene simultáneamente:

- El valor numérico cuantizado del flujo vehicular
- El embedding semántico del reporte de incidente asociado

El LLM aprende a predecir el siguiente token-paquete de forma autorregresiva, integrando ambas modalidades en una sola secuencia.

## Fases del Proyecto

### ✅ Fase 1 — Preprocesamiento e Ingeniería de Datos
Construcción de la base de datos unificada que alimentará el modelo.

**Entregables:**
- `train.csv` — 70% inicial cronológico (30,441 registros)
- `val.csv` — 10% siguiente (4,348 registros)
- `test.csv` — 20% final (8,699 registros)
- `adj_mx.pkl` — Matriz de adyacencia 325×325 sensores
- `tabla_maestra.csv` — Dataset completo alineado
- 4 gráficos de análisis exploratorio

**Técnicas aplicadas:**
- Time Series Resampling (5 min → 1 hora)
- Función heurística f(v) → T para generación de descripciones textuales
- Detector de gradiente temporal (∆v > 40 km/h en 5 minutos)
- Partición cronológica estricta sin data leakage temporal
- Gaussian Kernel Weighting para matriz de adyacencia

### 🔲 Fase 2 — Tokenización Numérica y Embeddings Semánticos
Conversión de las series de tiempo continuas en tokens discretos que el LLM pueda leer, y generación de embeddings semánticos para los textos de incidentes.

**Técnicas a implementar:**
- Uniform Quantization o VQ-VAE para tokenización numérica
- Embeddings semánticos con modelo de lenguaje preentrenado
- Construcción del token-paquete multimodal

### 🔲 Fase 3 — Arquitectura y Fine-tuning del LLM
Adaptación del LLM backbone para procesar la secuencia unificada de tokens.

**Técnicas a implementar:**
- Backbone: Llama 3 (8B) o Mistral 7B
- Low-Rank Adaptation (LoRA) para fine-tuning eficiente
- Partially Frozen Attention para captura de dependencias espacio-temporales

### 🔲 Fase 4 — Evaluación y Resultados
Evaluación del modelo unificado contra baselines del estado del arte.

**Métricas:** MAE, RMSE, MAPE en horizontes de 15, 30 y 60 minutos.

## Dataset

| Dataset | Fuente | Registros | Período |
|---|---|---|---|
| PEMS-BAY | Caltrans PeMS / Zenodo | 325 sensores × 52,116 timestamps | Ene–Jun 2017 |
| US Accidents | Moosavi et al. / Kaggle | 14,243 (Bay Area) de 7.7M totales | 2017 |

## Stack Tecnológico

| Herramienta | Uso |
|---|---|
| Python + Pandas | Ingeniería de datos y preprocesamiento |
| NumPy + SciPy | Cálculo matricial y estadístico |
| Matplotlib + Seaborn | Visualización |
| PyTorch | Entrenamiento del modelo |
| Hugging Face Transformers | Backbone LLM y tokenización |
| LoRA (PEFT) | Fine-tuning eficiente |

## Referencias Académicas

- Li, Y. et al. (2018). *Diffusion Convolutional Recurrent Neural Network.* ICLR 2018.
- Moosavi, S. et al. (2019). *A Countrywide Traffic Accident Dataset.* arXiv:1906.05409.
- Liu, C. et al. (2024). *Spatial-Temporal Large Language Model for Traffic Prediction.* arXiv:2401.10134.
- Liu, L. et al. (2024). *How Can Large Language Models Understand Spatial-Temporal Data?* arXiv:2401.14192.
- Yang, X. et al. (2025). *LLeCaT: LLM Enhanced Causality-aware Traffic Accidents Post-effects Prediction.* IEEE TITS.
- Li, Z. et al. (2025). *Open Spatio-Temporal Foundation Models for Traffic Prediction.* ACM TIST.
