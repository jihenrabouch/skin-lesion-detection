# advanced_augmentation_simple.py
"""
SOLUTION SIMPLIFIÉE D'AUGMENTATION ET ÉQUILIBRAGE POUR HAM10000
AVEC CHARGEMENT DES VRAIES IMAGES - VERSION CORRIGÉE
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

print("="*60)
print("🎯 SOLUTION SIMPLIFIÉE HAM10000 - VRAIES IMAGES")
print("="*60)

# ------------------------------------------------------------
# 1. TRANSFORMATIONS
# ------------------------------------------------------------
def get_augmentations(mode='train'):
    """Transformations pour lésions cutanées"""
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            # Augmentations géométriques
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomVerticalFlip(p=0.3),
            
            # Augmentations photométriques
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.05
            ),
            
            # Conversion
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

# ------------------------------------------------------------
# 2. DATASET ÉQUILIBRÉ - AVEC VRAIES IMAGES
# ------------------------------------------------------------
class BalancedSkinDataset(Dataset):
    """Dataset équilibré avec chargement des vraies images"""
    
    def __init__(self, target_samples=2000, mode='train', data_dir="data"):
        self.target_samples = target_samples
        self.mode = mode
        self.data_dir = data_dir
        
        print(f"\n🔧 Création dataset {mode}...")
        
        # ============= 1. CHARGER MÉTADONNÉES =============
        # Trouver le chemin ABSOLU du CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, data_dir, "HAM10000_metadata.csv")
        
        if not os.path.exists(csv_path):
            csv_path = os.path.join(current_dir, "data", "HAM10000_metadata.csv")
        
        self.df = pd.read_csv(csv_path)
        print(f"✅ Données chargées: {len(self.df)} images")
        print(f"📁 Fichier CSV: {csv_path}")
        
        # ============= 2. CLASSES =============
        self.classes = sorted(self.df['dx'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        print(f"📊 Classes: {self.classes}")
        
        # ============= 3. DISTRIBUTION ORIGINALE =============
        print("\n📈 DISTRIBUTION ORIGINALE:")
        self._show_distribution(self.df)
        
        # ============= 4. ÉQUILIBRER =============
        self.balanced_df = self._balance_data()
        
        # ============= 5. DISTRIBUTION ÉQUILIBRÉE =============
        print("\n📈 DISTRIBUTION ÉQUILIBRÉE:")
        self._show_distribution(self.balanced_df)
        
        # ============= 6. TROUVER LE DOSSIER DES IMAGES =============
        self.image_base_dir = self._find_image_directory()
        
        # ============= 7. TRANSFORMATIONS =============
        self.transform = get_augmentations(mode)
        
        print(f"\n✅ Dataset créé: {len(self.balanced_df)} images")
    
    def _show_distribution(self, df):
        """Affiche la distribution"""
        total = len(df)
        for cls in self.classes:
            count = len(df[df['dx'] == cls])
            pct = (count/total)*100
            print(f"  {cls}: {count} images ({pct:.1f}%)")
    
    def _balance_data(self):
        """Équilibre les données"""
        balanced = []
        
        for cls in self.classes:
            class_df = self.df[self.df['dx'] == cls]
            n_samples = len(class_df)
            
            if self.mode == 'train':
                if n_samples < self.target_samples:
                    # Oversampling
                    repeat = self.target_samples // n_samples
                    remainder = self.target_samples % n_samples
                    
                    duplicated = pd.concat([class_df] * repeat, ignore_index=True)
                    if remainder > 0:
                        extra = class_df.sample(n=remainder, replace=True, random_state=42)
                        duplicated = pd.concat([duplicated, extra], ignore_index=True)
                    
                    balanced.append(duplicated)
                else:
                    # Undersampling
                    sampled = class_df.sample(n=self.target_samples, random_state=42)
                    balanced.append(sampled)
            else:
                balanced.append(class_df)
        
        result = pd.concat(balanced, ignore_index=True)
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return result
    
    def _find_image_directory(self):
        """Trouve automatiquement le dossier contenant les images"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Chemins possibles pour le dossier images
        possible_dirs = [
            os.path.join(current_dir, "data", "images"),
            os.path.join(current_dir, "data"),
            os.path.join(os.path.dirname(current_dir), "data", "images"),
        ]
        
        for base_dir in possible_dirs:
            if os.path.exists(base_dir):
                # Chercher les sous-dossiers part_1 et part_2
                part1 = os.path.join(base_dir, "HAM10000_images_part_1")
                part2 = os.path.join(base_dir, "HAM10000_images_part_2")
                
                if os.path.exists(part1) and os.path.exists(part2):
                    print(f"📂 Dossier images trouvé: {base_dir}")
                    return base_dir
        
        print("⚠️ Dossier images non trouvé, utilisation du chemin par défaut")
        return os.path.join(current_dir, "data", "images")
    
    def get_sampler(self):
        """Crée un sampler pondéré"""
        if self.mode == 'train':
            # Calculer les poids
            class_weights = {}
            total = len(self.df)
            
            for cls in self.classes:
                count = len(self.df[self.df['dx'] == cls])
                class_weights[cls] = total / (len(self.classes) * count)
            
            # Poids par échantillon
            sample_weights = [
                class_weights[row['dx']]
                for _, row in self.balanced_df.iterrows()
            ]
            
            weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
            return WeightedRandomSampler(
                weights=weights_tensor,
                num_samples=len(self),
                replacement=True
            )
        return None
    
    def __len__(self):
        return len(self.balanced_df)
    
    # ============================================================
    # ✅ FONCTION __getitem__ CORRIGÉE - CHARGE LES VRAIES IMAGES
    # ============================================================
    def __getitem__(self, idx):
        """Récupère un échantillon avec la VRAIE image"""
        row = self.balanced_df.iloc[idx]
        label = self.class_to_idx[row['dx']]
        image_id = row['image_id']
        
        # Construction du nom de fichier
        img_filename = f"{image_id}.jpg"
        
        # ============= CHEMINS CORRIGÉS =============
        # 1. Essayer dans HAM10000_images_part_1
        img_path = os.path.join(self.image_base_dir, "HAM10000_images_part_1", img_filename)
        
        # 2. Si pas trouvé, essayer dans HAM10000_images_part_2
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_base_dir, "HAM10000_images_part_2", img_filename)
        
        # 3. Si toujours pas trouvé, essayer directement dans images/
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_base_dir, img_filename)
        
        # Charger l'image
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                print(f"⚠️ Image non trouvée: {image_id} à {img_path}")
                image = Image.new('RGB', (224, 224), color='gray')
        except Exception as e:
            print(f"❌ Erreur chargement {image_id}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ------------------------------------------------------------
# 3. FONCTION DE DIAGNOSTIC
# ------------------------------------------------------------
def diagnose_paths():
    """Diagnostique la structure des dossiers"""
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC - STRUCTURE DES DOSSIERS")
    print("="*60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Répertoire courant: {current_dir}")
    
    # Vérifier data/
    data_dir = os.path.join(current_dir, "data")
    print(f"\n📂 data/ existe: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        # Vérifier data/images/
        images_dir = os.path.join(data_dir, "images")
        print(f"📂 data/images/ existe: {os.path.exists(images_dir)}")
        
        if os.path.exists(images_dir):
            # Lister les sous-dossiers
            subdirs = [d for d in os.listdir(images_dir) 
                      if os.path.isdir(os.path.join(images_dir, d))]
            print(f"   Sous-dossiers: {subdirs}")
            
            # Vérifier les dossiers part_1 et part_2
            part1 = os.path.join(images_dir, "HAM10000_images_part_1")
            part2 = os.path.join(images_dir, "HAM10000_images_part_2")
            
            print(f"\n📂 HAM10000_images_part_1 existe: {os.path.exists(part1)}")
            if os.path.exists(part1):
                jpgs = [f for f in os.listdir(part1) if f.endswith('.jpg')]
                print(f"   {len(jpgs)} images JPG trouvées")
                if jpgs:
                    print(f"   Exemple: {jpgs[0]}")
            
            print(f"\n📂 HAM10000_images_part_2 existe: {os.path.exists(part2)}")
            if os.path.exists(part2):
                jpgs = [f for f in os.listdir(part2) if f.endswith('.jpg')]
                print(f"   {len(jpgs)} images JPG trouvées")
                if jpgs:
                    print(f"   Exemple: {jpgs[0]}")

# ------------------------------------------------------------
# 4. TEST SIMPLE
# ------------------------------------------------------------
def test_simple():
    """Test simple avec vérification des images"""
    print("\n🧪 TEST DE LA SOLUTION - VRAIES IMAGES")
    print("="*60)
    
    try:
        # Créer dataset
        dataset = BalancedSkinDataset(
            target_samples=300,
            mode='train',
            data_dir="data"
        )
        
        print(f"\n✅ RÉSULTATS:")
        print(f"   Images totales: {len(dataset)}")
        
        # Vérifier le chargement des images
        print("\n🖼️  TEST CHARGEMENT IMAGES:")
        for i in range(min(5, len(dataset))):
            img, label = dataset[i]
            print(f"   Image {i}: {dataset.idx_to_class[label]}, "
                  f"min={img.min():.2f}, max={img.max():.2f}")
        
        # Calculer les ratios
        original_counts = []
        balanced_counts = []
        
        for cls in dataset.classes:
            orig = len(dataset.df[dataset.df['dx'] == cls])
            bal = len(dataset.balanced_df[dataset.balanced_df['dx'] == cls])
            original_counts.append(orig)
            balanced_counts.append(bal)
        
        orig_ratio = max(original_counts) / min(original_counts)
        bal_ratio = max(balanced_counts) / min(balanced_counts)
        
        print(f"\n⚖️  COMPARAISON:")
        print(f"   Ratio original: {orig_ratio:.1f}x")
        print(f"   Ratio équilibré: {bal_ratio:.2f}x")
        print(f"   Amélioration: {orig_ratio/bal_ratio:.1f}x mieux!")
        
        print("\n🎉 SUCCÈS! Le dataset est prêt.")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Lancer le diagnostic d'abord
    diagnose_paths()
    # Puis le test
    test_simple()