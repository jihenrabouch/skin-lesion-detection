# 🎓 Livrable 2 – Exploration et Équilibrage du Dataset HAM10000

## 📌 Contexte et Problématique

Le dataset **HAM10000** présente un **déséquilibre critique** entre les classes :

| Classe | Type de lésion | Images | Pourcentage |
|--------|----------------|--------|-------------|
| **nv** | Naevus mélanocytaire | 6 705 | 66.9% |
| **mel** | Mélanome | 1 113 | 11.1% |
| **bkl** | Kératose bénigne | 1 099 | 11.0% |
| **bcc** | Carcinome basocellulaire | 514 | 5.1% |
| **akiec** | Carcinome épidermoïde | 327 | 3.3% |
| **vasc** | Lésion vasculaire | 142 | 1.4% |
| **df** | Dermatofibrome | 115 | 1.1% |

**Conséquences** :
- Biais du modèle vers la classe majoritaire (`nv`)
- Mauvaise généralisation sur les classes rares (`df`, `vasc`)
- Métriques trompeuses (accuracy élevée mais recall faible)

**Ratio de déséquilibre** : `58.3x` (6705 / 115)

---

## 🎯 Objectif du Livrable

✅ **Corriger le déséquilibre** pour un entraînement Deep Learning non-biaisé  
✅ **Visualiser** la distribution avant/après équilibrage  
✅ **Charger les vraies images** avec augmentations  
✅ **Préparer un dataset PyTorch** prêt pour l'entraînement  

---

## 🛠️ Stratégie d'Équilibrage

### 📉 Undersampling
- **Classe majoritaire `nv`** : 6 705 → **300 images**

### 📈 Oversampling
- **Classes minoritaires** (`df`, `vasc`, `akiec`) : 115–327 → **300 images**
- Technique : duplication avec remplacement

### 🎨 Augmentations appliquées
| Type | Paramètres |
|------|------------|
| 🔄 Rotation | ±30 degrés |
| ↔️ Flip horizontal | 50% |
| ↕️ Flip vertical | 30% |
| ✂️ Random crop | 224×224, échelle 0.7–1.0 |
| 🎨 ColorJitter | Brightness ±0.2, Contrast ±0.2, Saturation ±0.15, Hue ±0.05 |
| 📊 Normalisation | Mean [0.485,0.456,0.406], Std [0.229,0.224,0.225] |

---

## 📊 Résultats Obtenus

### Distribution avant/après

| Classe | Original | Équilibré | Variation |
|--------|----------|-----------|-----------|
| nv | 6 705 | 300 | **-95.5%** |
| mel | 1 113 | 300 | -73.0% |
| bkl | 1 099 | 300 | -72.7% |
| bcc | 514 | 300 | -41.6% |
| akiec | 327 | 300 | -8.3% |
| vasc | 142 | 300 | **+111.3%** |
| df | 115 | 300 | **+160.9%** |
| **Total** | **10 015** | **2 100** | -79.0% |

### Métriques d'équilibrage

| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Ratio max/min | **58.3x** | **1.00x** | ✅ **-98.3%** |
| Écart-type | 2 370 | 0 | ✅ **-100%** |
| Coefficient de variation | 1.66 | 0 | ✅ **-100%** |

---

## 🖼️ Visualisations

Le notebook `01_exploration_equilibre.ipynb` contient :

1. **Distribution des classes** (barplot + pie chart)
2. **Comparaison original vs équilibré**
3. **4 échantillons par classe** avec augmentations visibles
4. **Batch d'entraînement** (16 images)
5. **Statistiques des pixels** (mean, std, histogrammes)

![Visualisation](https://via.placeholder.com/800x400?text=4+images+par+classe)

---

## 📁 Fichiers du Livrable

| Fichier | Description |
|---------|-------------|
| `01_exploration_equilibre.ipynb` | Notebook complet d'analyse et visualisation |
| `advanced_augmentation_simple.py` | Dataset PyTorch équilibré avec augmentations |
| `README.md` | Documentation du livrable |

---

## 🚀 Utilisation

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
