# check_balance_simple.py
import sys
import os
import pandas as pd

print(f"🔍 VÉRIFICATION SIMPLE")

# Désactiver les avertissements temporairement
import warnings
warnings.filterwarnings('ignore')

# Chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print(f"Projet: {project_root}")

# 1. Vérifier la structure
print("\n📁 STRUCTURE:")
for item in ["src/data", "data"]:
    path = os.path.join(project_root, item)
    if os.path.exists(path):
        print(f"✅ {item}/")
        files = [f for f in os.listdir(path) if f.endswith('.py')]
        for f in files[:3]:  # Afficher 3 fichiers max
            print(f"   📄 {f}")
    else:
        print(f"❌ {item}/")

# 2. Analyser les données directement
print("\n📊 ANALYSE DES DONNÉES:")
metadata_path = os.path.join(project_root, "data", "HAM10000_metadata.csv")

if os.path.exists(metadata_path):
    df = pd.read_csv(metadata_path)
    print(f"✅ Metadata: {len(df)} entrées")
    
    if 'dx' in df.columns:
        counts = df['dx'].value_counts()
        print("\n📈 DISTRIBUTION:")
        total = len(df)
        for cls, count in counts.items():
            pct = (count/total)*100
            print(f"  {cls}: {count:5d} ({pct:5.1f}%)")
        
        ratio = counts.max() / counts.min()
        print(f"\n⚖️  Déséquilibre: {ratio:.1f}x")
    else:
        print("❌ Pas de colonne 'dx'")
else:
    print(f"❌ Metadata non trouvé: {metadata_path}")

print("\n" + "="*50)
print("Pour supprimer l'erreur Pylance:")
print("1. Créez .vscode/settings.json")
print("2. Ou ignorez l'avertissement (c'est juste l'éditeur)")