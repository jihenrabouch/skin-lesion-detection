
# Livrable 1 – Initialisation du projet

## Détection de lésions cutanées par Deep Learning

**Nom du projet** : Skin Lesion Detection  

**Équipe (Team)** :  
- Jihen Benrabouch  
- Ahmed Toujeni  
- Ines Fehri  

**Encadrant** : M. Haythem Ghazouani


#  Objectif du projet

L’objectif de ce projet est de développer un **système intelligent de détection des lésions cutanées** à partir d’images de la peau, afin de :

- Identifier automatiquement les lésions cutanées  
- Aider au **dépistage précoce** de maladies dermatologiques  
  (ex. lésions bénignes / suspectes)
- Réduire le temps d’analyse et l’erreur humaine  
- Fournir un **outil d’aide à la décision** pour les professionnels de santé  

 Ce système ne remplace pas un médecin, mais sert comme **support d’analyse**.

---

#  Architecture du projet

Le projet suit une **architecture modulaire** orientée **vision par ordinateur** et **intelligence artificielle**.

skin-lesion-detection/
├── data/
│ ├── raw/ # Images originales
│ ├── processed/ # Images prétraitées
│ └── labels/ # Fichiers d’annotations
├── models/
│ ├── cnn_model.py # Modèle de deep learning
│ └── saved_models/ # Modèles entraînés
├── src/
│ ├── preprocessing/ # Redimensionnement, normalisation
│ ├── training/ # Entraînement du modèle
│ ├── evaluation/ # Évaluation des performances
│ └── prediction/ # Prédiction sur nouvelles images
├── livrable/
│ ├── rapport.pdf
│ ├── presentation.pptx
│ └── README.md
├── notebooks/ # Expérimentations (Jupyter)
├── requirements.txt
└── README.md

 Cette architecture facilite :
- la maintenance  
- la réutilisation du code  
- l’évolution du projet (ajout de nouvelles classes de lésions)

---

#  Dataset (Jeu de données)

Le projet utilise un **dataset d’images dermatologiques** contenant différentes catégories de lésions cutanées.

##  Exemples de datasets utilisés
- **HAM10000**

##  Description du dataset
- **Type** : Images (.JPG)
- **Nombre d’images** : 10 015
- **Classes possibles** :
  nv → mélanocytique bénin

  mel → mélanome

  bkl → kératose bénigne

  bcc → carcinome basocellulaire

  akiec → kératose actinique / épithélioma

  vasc → lésion vasculaire

  df → dermatofibrome |
      | Métadonnées | Contenues dans HAM10000_metadata.csv (image_id, dx, age, sex, localisation, source) |

##  Prétraitement des données
- Redimensionnement des images  
- Normalisation des pixels  
- Augmentation de données :
  - rotation
  - zoom
  - flip horizontal
- Séparation des données :
  - Entraînement  
  - Validation  
  - Test  

---

#  Technologies utilisées

- Python  
- TensorFlow / Keras ou PyTorch  
- OpenCV  
- NumPy, Pandas  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

#  Résultats attendus

- Un modèle capable de classifier les lésions cutanées avec une **bonne précision**
- Une **interface simple** permettant de tester une image
- Des **métriques d’évaluation** :
  - Accuracy  
  - Precision  
  - Recall  
