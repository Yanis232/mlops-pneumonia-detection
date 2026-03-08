
# 🩻 Model Card: Baseline CNN (Theme 2)

## 1. Détails du Modèle
- **Date de création:** 2026-02-10 10:41
- **Type:** Convolutional Neural Network (CNN) - Custom Baseline
- **Framework:** PyTorch 2.10.0+cu126
- **Device:** NVIDIA GeForce RTX 4070 Laptop GPU
- **Dataset:** Chest X-Ray Images (Pneumonia)

## 2. Architecture Technique
Le modèle est un CNN léger conçu pour respecter une contrainte de **1M à 2M de paramètres**.

### Structure :
1.  **Extraction de Features (3 Blocs) :**
    * Conv2d (32 filtres) $\rightarrow$ BatchNormal $\rightarrow$ ReLU $\rightarrow$ MaxPool
    * Conv2d (64 filtres) $\rightarrow$ BatchNormal $\rightarrow$ ReLU $\rightarrow$ MaxPool
    * Conv2d (128 filtres) $\rightarrow$ BatchNormal $\rightarrow$ ReLU $\rightarrow$ MaxPool
2.  **Transition Spatiale :**
    * `AdaptiveAvgPool2d((7, 7))` : Force une taille fixe avant le classifier, rendant le modèle robuste aux dimensions d'entrée variables.
3.  **Classification (MLP) :**
    * Flatten $\rightarrow$ Linear (256 neurones) $\rightarrow$ Dropout (0.5) $\rightarrow$ Output (2 classes).

- **Total Paramètres :** 1,699,522 (Objectif 1-2M respecté ✅)
- **Input Shape :** (1, 224, 224) - Grayscale

## 3. Stratégie d'Entraînement
- **Optimiseur :** Adam (lr=0.001) avec Weight Decay (1e-4) pour la régularisation.
- **Loss Function :** `CrossEntropyLoss` **Pondérée**.
    * *Pourquoi ?* Le dataset contient 3x plus de pneumonies. Nous avons inversé les poids pour pénaliser fortement les erreurs sur la classe minoritaire.
- **Data Augmentation :** RandomRotation, HorizontalFlip (Train set uniquement).

## 4. Performance (Test Set)
> Évaluation réalisée sur **624 images** jamais vues (Test Set officiel).

| Métrique | Score | Interprétation |
| :--- | :--- | :--- |
| **Accuracy Globale** | **79.49%** | Objectif du cours (>70%) validé. |
| **Recall (PNEUMONIA)** | **98.46%** | 🚨 **CRITIQUE**. Capacité à détecter les malades. Excellent score. |
| **Recall (NORMAL)** | 47.86% | Capacité à confirmer les cas sains. Score faible (bruit). |
| **F1-Score (Pneumonia)** | 85.71% | Moyenne harmonique Précision/Rappel. |

### 🧠 Analyse du Comportement
🛡️ **Modèle Sécuritaire (High Sensitivity)** : Le modèle privilégie la détection des malades (Recall Pneumonia élevé) au risque de créer des fausses alertes sur les patients sains (Faux Positifs).

## 5. Limitations & Améliorations (Thème 3)
- **Faux Positifs :** Le modèle classe trop de patients "Normaux" comme "Malades" (Recall Normal faible).
- **Architecture :** Le modèle est entraîné "from scratch". Il manque de connaissances préalables sur les formes médicales.
- **Piste d'amélioration :** Utiliser le **Transfer Learning** (ResNet18 ou EfficientNet) pré-entraîné sur ImageNet pour améliorer la distinction des features complexes.

## 6. Visualisations
- Voir `images/theme2_confusion_matrix.png` pour la matrice de confusion détaillée.
- Voir TensorBoard (`runs/`) pour les courbes d'apprentissage.
