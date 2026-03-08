# 🫁 Deep Learning with PyTorch — MLOps Pipeline
## Cas d'usage clinique : Détection automatisée de pneumonies

> **Auteur :** Yanis GOUTAL-GUÉRIN
> **Cours :** Deep Learning with PyTorch – MLOps Pipeline : From Data to Production
> **Professeur :** Clément GICQUEL — ISEN Méditerranée 2026
> **Date :** 8 mars 2026

---

## 📋 Vue d'ensemble

Ce projet implémente un pipeline MLOps complet **end-to-end** pour la détection de pneumonies sur des radiographies pulmonaires (dataset *Chest X-Ray Pneumonia*). Il couvre l'intégralité du cycle de vie d'un modèle de Deep Learning : de l'exploration des données jusqu'au déploiement progressif (Canary Release), en passant par le monitoring, la détection de drift et le réentraînement automatisé.

### Trajectoire de performance

| Phase | Architecture | Acc. Clean | Acc. Drift | Statut |
|---|---|---|---|---|
| Baseline | Custom CNN (1.7M params) | 79.49% | — | Proof of Concept |
| Optimisation | DenseNet121 Transfer Learning | 94.07% | — | Champion V1 |
| Incident Drift | Simulation "Nano-Dosing" | 93.91% | 82.85% | Défaillant |
| **Récupération** | **Retrained V2 (Data Mixing)** | **93.26%** | **97.75%** | ✅ **Champion V2** |

---

## 🗂️ Structure du Projet

```
IA_GEN_PROJET/
├── README.md
├── requirements.txt                         # pip freeze — dépendances exactes
│
├── notebooks/
│   ├── theme1_data_analysis.ipynb
│   ├── theme2_baseline_model.ipynb
│   ├── theme3_optimization.ipynb
│   ├── theme4_onnx_deployment.ipynb
│   ├── theme5_monitoring.ipynb
│   ├── theme6_drift_detection.ipynb
│   ├── theme7_retraining.ipynb
│   ├── theme7-bis_canary.ipynb
│   └── theme8_synthesis.ipynb
│
├── artifacts/
│   ├── densenet121_tl_champion.onnx         # Modèle V1 de production
│   ├── densenet121_tl_retrained_theme7.onnx # Modèle V2 — Champion final
│   ├── champion_config.json                 # Métadonnées + métriques V1
│   └── monitoring/
│       ├── logs/                            # Cold Storage CSV (inférences, Canary)
│       └── tensorboard/                     # Hot Storage (15+ runs TensorBoard)
│
├── checkpoints/
│   ├── best_model.pth                       # Poids CNN Baseline (Thème 2)
│   ├── optimized_best_densenet121_TL.pth    # Golden Source V1 (Thème 3)
│   ├── theme7_retrained_best_v2_seed42.pth  # Golden Source V2 (Thème 7)
│   ├── baseline_metrics.pt                  # Stats dataset de référence (Thème 1)
│   ├── baseline_production_metrics.pt       # État nominal production (Thème 5)
│   └── model_card.md                        # Fiche identité du modèle
│
├── chest_xray/
│   ├── train/ | val/ | test/                # Dataset original sain
│   └── test_drifted_v4_nano14/              # Dataset drifté calibré (Thème 6)
│       ├── NORMAL/
│       ├── PNEUMONIA/
│       └── drift_params.json                # Paramètres de corruption reproductibles
│
└── images/                                  # Outputs graphiques (matrices, ROC, benchmarks)
```

---

## ⚙️ Installation & Reproductibilité

### Prérequis matériels et logiciels

| Composant | Version utilisée |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4070 Laptop (8.59 GB VRAM) |
| CUDA | 12.x |
| cuDNN | 9.x |
| Python | 3.10+ |
| PyTorch | 2.5 (build CUDA) |
| ONNX Runtime GPU | 1.24 |

> **Note :** Le projet a été développé en local pour garantir la stabilité des stress-tests continus (Thèmes 6 et 7). Les quotas GPU de Google Colab ne permettent pas de reproduire l'intégralité du pipeline sans interruptions. Une compatibilité Colab partielle est possible via le backend CPU d'ONNX Runtime.

---

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/Yanis232/mlops-pneumonia-detection.git
cd mlops-pneumonia-detection
```

---

### Étape 2 — Installer les dépendances

```bash
pip install -r requirements.txt
```

Le `requirements.txt` est un `pip freeze` complet de l'environnement de développement. Il contient toutes les dépendances transitives. Les packages principaux sont :

| Package | Usage dans le projet |
|---|---|
| `torch` / `torchvision` | Entraînement et inférence PyTorch (GPU CUDA) |
| `onnx` / `onnxruntime-gpu` | Export et inférence ONNX Runtime (Thème 4) |
| `tensorboard` | Dashboard de monitoring production (Thème 5) |
| `scipy` | Test de Kolmogorov-Smirnov (Thème 6) |
| `scikit-learn` | Métriques : Accuracy, F1, AUC, matrices de confusion |
| `matplotlib` / `seaborn` | Visualisations |
| `pandas` | Lecture / écriture des logs CSV (Thèmes 5, 7-bis) |
| `numpy` | Manipulations numériques |
| `Pillow` | Transformations d'images (pipeline de drift) |
| `tqdm` | Barres de progression |
| `kaggle` | Téléchargement du dataset via API |

---

### Étape 3 — Télécharger le dataset (Kaggle)

**3a. Configurer l'API Kaggle**

Créer un compte sur [kaggle.com](https://www.kaggle.com), générer un token API (`Account → API → Create New Token`), puis :

```bash
# Linux / macOS
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
mkdir $env:USERPROFILE\.kaggle
copy kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```

**3b. Télécharger et extraire le dataset**

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -q chest-xray-pneumonia.zip -d chest_xray/
```

La structure attendue après extraction :

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/       # ⚠️ Seulement 16 images d'origine → re-split au Thème 1
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

### Étape 4 — Vérifier l'environnement GPU

```python
import torch

print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA disponible  : {torch.cuda.is_available()}")
print(f"GPU détecté      : {torch.cuda.get_device_name(0)}")
print(f"VRAM disponible  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Résultat attendu sur l'environnement de référence :
```
PyTorch version  : 2.5.0+cu121
CUDA disponible  : True
GPU détecté      : NVIDIA GeForce RTX 4070 Laptop GPU
VRAM disponible  : 8.5 GB
```

---

### Étape 5 — Seed de reproductibilité

**Chaque notebook commence par ce bloc — ne pas le déplacer ni le supprimer :**

```python
import torch, numpy as np, random
from datetime import datetime

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Déterminisme GPU strict
# (léger coût en performance, indispensable pour des benchmarks comparables)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"[{datetime.now().strftime('%H:%M:%S')}] Seed fixée à {SEED} — environnement déterministe.")
```

> ⚠️ Sans `cudnn.deterministic = True`, deux runs identiques peuvent diverger
> de plusieurs points de pourcentage sur un dataset déséquilibré comme Chest X-Ray.

**Reproductibilité des DataLoaders (workers parallèles) :**

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

trainloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g
)
```

**Reproductibilité du dataset drifté :**

Le dossier `test_drifted_v4_nano14/` peut être régénéré à l'identique depuis les images propres
grâce au fichier de configuration :

```json
{
  "seed": 42,
  "params": {
    "noise": 0.014,
    "blur": 0.35,
    "rot": 1.0,
    "bright": 0.025,
    "cutout": 0.007
  }
}
```

Voir `theme6_drift_detection.ipynb` — Section "Nano-Dosing Pipeline" pour la procédure complète.

---

### Étape 6 — Exécuter les notebooks dans l'ordre

Les notebooks sont **séquentiels** : chaque thème consomme les artefacts produits par le précédent.

```
theme1_data_analysis.ipynb
    └─→ produit : checkpoints/baseline_metrics.pt

theme2_baseline_model.ipynb
    └─→ produit : checkpoints/best_model.pth

theme3_optimization.ipynb
    └─→ produit : checkpoints/optimized_best_densenet121_TL.pth

theme4_onnx_deployment.ipynb
    └─→ produit : artifacts/densenet121_tl_champion.onnx
                  artifacts/champion_config.json

theme5_monitoring.ipynb
    └─→ produit : checkpoints/baseline_production_metrics.pt
                  artifacts/monitoring/logs/production_logs.csv

theme6_drift_detection.ipynb
    └─→ produit : chest_xray/test_drifted_v4_nano14/

theme7_retraining.ipynb
    └─→ produit : checkpoints/theme7_retrained_best_v2_seed42.pth
                  artifacts/densenet121_tl_retrained_theme7.onnx

theme7-bis_canary.ipynb
    └─→ produit : artifacts/monitoring/logs/canary_rollout_*.csv

theme8_synthesis.ipynb
    └─→ synthèse finale (aucun artefact produit)
```

**Lancer le monitoring TensorBoard à tout moment :**

```bash
tensorboard --logdir artifacts/monitoring/tensorboard
# Ouvrir http://localhost:6006 dans le navigateur
```

---

## 📊 Résultats Clés

| Étape | Architecture | Format | Acc. Clean | Acc. Drift | Latence (b=1) | Statut |
|---|---|---|---|---|---|---|
| Baseline | CNN Custom (1.7M) | PyTorch | 79.49% | — | ~12 ms | Obsolète |
| Optimisation | DenseNet121 TL | PyTorch | 94.07% | — | ~15 ms | Recherche |
| Production V1 | DenseNet121 TL | ONNX | 93.91% | — | 25.8 ms | Déployé |
| Incident Drift | DenseNet121 TL | ONNX | 93.91% | 82.85% | 25.8 ms | ⚠️ Défaillant |
| **Production V2** | **DenseNet121-FT** | **ONNX** | **93.26%** | **97.75%** | **24.1 ms** | ✅ **Champion** |

---

## 📖 Vue d'ensemble des Thèmes

| Thème | Contenu principal | Résultat clé | Livrable |
|---|---|---|---|
| **1 – Data Analysis** | EDA, re-split 80/20, normalisation adaptative monocanal | Mean=0.4823, Std=0.2361 | `baseline_metrics.pt` |
| **2 – Baseline Model** | CNN from scratch (1.7M params), AMP, TensorBoard | Accuracy **79.49%**, Overfitting Gap **0.16%** | `best_model.pth` |
| **3 – Optimization** | ResNet18 vs DenseNet121 TL, LR Finder, OneCycleLR | DenseNet121 → **94.07%**, AUC **0.988** | `optimized_best_densenet121_TL.pth` |
| **4 – ONNX Deployment** | Export Opset 18, benchmark GPU, fallback TensorRT | Speedup **1.58×** (batch=1), Top-1 Agreement **100%** | `densenet121_tl_champion.onnx` |
| **5 – Monitoring** | InferenceLogger CSV, TensorBoard production, Cold Start | P95 latence 35.48 ms, Confiance moy. 0.958 | `baseline_production_metrics.pt` |
| **6 – Drift Detection** | Stress-Test 5 vecteurs, KS-Test, Nano-Dosing | KS p-value < 4.86×10⁻⁹, drop −11.06 pts → Trigger | `test_drifted_v4_nano14/` |
| **7 – Retraining** | Data Mixing 90/10, Differential LR, Validation Gate | **97.75%** sur données driftées, MLOps Score **0.906** | `theme7_retrained_best_v2_seed42.pth` |
| **7-bis – Canary** | Routeur probabiliste A/B, rollout 10→100% | PROMOTE déclenché, 7 FP sur 624 requêtes | `canary_rollout_*.csv` |
| **8 – Synthesis** | Model Registry, Decision Framework, Maturité MLOps Niv.1 | Sculley et al. (2015) — Hidden Technical Debt | Rapport final PDF |

---

## 🔬 Stratégie Anti-Drift

Le pipeline détecte et corrige automatiquement la dérive des données en 3 étapes :

1. **Détection (KS-Test)** — Comparaison de la distribution des scores de confiance Softmax entre la baseline de production et le flux courant
2. **Trigger automatique** — Alerte si `KS p-value < 0.05` AND `Accuracy Drop > 5%`
3. **Réentraînement (Data Mixing)** — 90% données saines + 10% données driftées pour prévenir l'oubli catastrophique

```python
IF (KS_p_value < 0.05) AND (Accuracy_Drop > 0.05):
    TRIGGER = "RETRAINING_PIPELINE"
    STATUS  = "MODEL_UNRELIABLE"
```

---

## 📁 Artefacts Clés

| Fichier | Format | Usage |
|---|---|---|
| `densenet121_tl_champion.onnx` | ONNX | Modèle V1 — déploiement production initial |
| `densenet121_tl_retrained_theme7.onnx` | ONNX | **Modèle V2 — Champion final** |
| `champion_config.json` | JSON | Métadonnées, timestamp, métriques de référence |
| `baseline_metrics.pt` | PT | Statistiques dataset (mean, std, class weights) |
| `baseline_production_metrics.pt` | PT | État nominal production (confiance, latence P95) |
| `drift_params.json` | JSON | Paramètres de corruption reproductibles (seed=42) |
| `model_card.md` | Markdown | Fiche identité du modèle (architecture, métriques) |

---

## 🐛 Common Issues & Solutions

### CUDA out of memory
```python
BATCH_SIZE = 16         # Réduire si OOM sur GPU < 8 GB
torch.cuda.empty_cache()
```

### TensorRT échoue sur Windows
```python
# Comportement attendu et documenté en Thème 4.
# Le fallback automatique vers CUDAExecutionProvider est implémenté.
# L'objectif de speedup est atteint via ONNX Runtime seul (1.58×).
print("TensorRT unavailable → Falling back to CUDAExecutionProvider")
```

### DataLoader lent ou erreurs num_workers
```python
# Sur Windows, num_workers > 0 peut causer des erreurs de multiprocessing
trainloader = DataLoader(..., num_workers=0, pin_memory=False)
```

### Résultats non reproductibles entre deux runs
```python
# S'assurer que ces 5 lignes sont présentes AVANT tout autre code dans le notebook
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Cold Start ONNX (~800 ms sur la 1ère inférence)
```python
# Implémenter des warmup requests avant d'ouvrir le trafic réel
dummy = np.zeros((1, 1, 224, 224), dtype=np.float32)
for _ in range(5):
    session.run(None, {"input": dummy})
# La 6ème inférence sera au régime de croisière (~25 ms)
```

---

## ⚠️ Notes Techniques

**TensorRT :** Non fonctionnel sur l'environnement Windows local (DLLs natives `nvinfer.dll` non résolues). ONNX Runtime bascule automatiquement sur `CUDAExecutionProvider`. Dans un environnement de production, un conteneur **NVIDIA Triton Inference Server sous Linux** est recommandé pour activer TensorRT.

**Cold Start :** La première inférence ONNX/CUDA prend ~810 ms (initialisation du contexte CUDA + compilation JIT des kernels). En production, implémenter des **warmup requests** avant ouverture du trafic.

**Google Colab :** Les quotas GPU limitent les stress-tests continus. Pour reproduire l'intégralité des expériences (Thèmes 6 et 7), un GPU local est indispensable.

---

## 📚 Références

| Papier | Usage dans ce projet |
|---|---|
| He et al., *Deep Residual Learning for Image Recognition* (2015) | Architecture ResNet18 — Thème 3 |
| Huang et al., *Densely Connected Convolutional Networks* (2017) | Architecture DenseNet121 — Thème 3 |
| Ioffe & Szegedy, *Batch Normalization* (2015) | `BatchNorm2d` dans CNN et DenseNet — Thèmes 2 & 3 |
| Micikevicius et al., *Mixed Precision Training* (2018) | `torch.cuda.amp` — Thème 2 |
| Rabanser et al., *Failing Loudly* (2019) | KS-Test sur scores Softmax vs pixels bruts — Thème 6 |
| Sculley et al., *Hidden Technical Debt in ML Systems* (2015) | Justification architecture MLOps — Thème 8 |

Documentation : [PyTorch](https://pytorch.org/docs) · [ONNX Runtime](https://onnxruntime.ai/docs) · [TensorBoard](https://tensorboard.dev) · [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

*Ce projet démontre qu'un modèle de Deep Learning n'est pas un produit fini au moment de son déploiement — c'est le début d'un cycle de vie qui exige monitoring, détection d'anomalies et capacité d'auto-réparation.*