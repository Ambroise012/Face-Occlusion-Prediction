"""
Script d'analyse rapide du dataset pour comprendre sa structure,
sa répartition, et ses statistiques.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import IMAGE_DIR, TRAIN_CSV, TEST_CSV, VAL_SIZE

def load_and_clean_data():
    """Charge et nettoie les données."""
    df_train = pd.read_csv(TRAIN_CSV, delimiter=',')
    df_test = pd.read_csv(TEST_CSV, delimiter=',')

    # Supprimer les lignes avec des valeurs manquantes
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Split train/val
    df_val = df_train.loc[:VAL_SIZE].reset_index(drop=True)
    df_train = df_train.loc[VAL_SIZE:].reset_index(drop=True)

    return df_train, df_val, df_test

def print_basic_info(df, name):
    """Affiche les infos de base d'un DataFrame."""
    print(f"\n{'='*60}")
    print(f"📊 {name.upper()}")
    print(f"{'='*60}")
    print(f"Nombre d'échantillons : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")
    print(f"Colonnes : {list(df.columns)}")
    print(f"Types :\n{df.dtypes}")

def print_statistics(df, name):
    """Affiche les statistiques numériques."""
    print(f"\n📈 Statistiques numériques pour {name} :")
    print(df.describe())

def analyze_occlusion(df, name):
    """Analyse la colonne FaceOcclusion."""
    print(f"\n🎭 Analyse de FaceOcclusion ({name}) :")
    occlusion = df['FaceOcclusion']
    print(f"  - Moyenne : {occlusion.mean():.4f}")
    print(f"  - Médiane : {occlusion.median():.4f}")
    print(f"  - Min : {occlusion.min():.4f}")
    print(f"  - Max : {occlusion.max():.4f}")
    print(f"  - Écart-type : {occlusion.std():.4f}")
    print(f"  - Valeurs uniques : {occlusion.nunique()}")

    # Répartition en bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
              '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    occlusion_binned = pd.cut(occlusion, bins=bins, labels=labels, include_lowest=True)
    print(f"\n  Répartition par intervalles :")
    print(occlusion_binned.value_counts().sort_index())

def analyze_gender(df, name):
    """Analyse la colonne gender."""
    print(f"\n👫 Analyse du genre ({name}) :")
    gender = df['gender']
    print(f"  - Répartition :")
    print(gender.value_counts().sort_index())

    # Moyenne de FaceOcclusion par genre
    print(f"\n  - Moyenne de FaceOcclusion par genre :")
    print(df.groupby('gender')['FaceOcclusion'].mean())

def check_image_files(df, image_dir, name):
    """Vérifie l'existence des fichiers images (1000 premiers)."""
    print(f"\n🖼️ Vérification des images ({name}) :")
    filenames = df['filename'].tolist()
    missing_files = []
    existing_files = 0

    for filename in filenames[:1000]:  # Limité à 1000 pour éviter la lenteur
        file_path = Path(image_dir) / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            existing_files += 1

    print(f"  - Fichiers vérifiés : 1000")
    print(f"  - Fichiers existants : {existing_files}")
    print(f"  - Fichiers manquants : {len(missing_files)}")
    if missing_files:
        print(f"  - Exemples de fichiers manquants : {missing_files[:5]}")

def analyze_filename_patterns(df, name):
    """Analyse les motifs dans les noms de fichiers."""
    print(f"\n📁 Analyse des noms de fichiers ({name}) :")
    filenames = df['filename']
    print(f"  - Exemples : {filenames.head().tolist()}")
    print(f"  - Longueur moyenne des noms : {filenames.str.len().mean():.1f} caractères")
    extensions = filenames.str.split('.').str[-1]
    print(f"  - Extensions : {extensions.value_counts().to_dict()}")

def main():
    print("🔍 Analyse du dataset...")

    # Charger les données
    df_train, df_val, df_test = load_and_clean_data()

    # Infos de base
    print_basic_info(df_train, "Train")
    print_basic_info(df_val, "Validation")
    print_basic_info(df_test, "Test")

    # Statistiques numériques
    print_statistics(df_train, "Train")

    # Analyse de FaceOcclusion
    analyze_occlusion(df_train, "Train")
    analyze_occlusion(df_val, "Validation")
    analyze_occlusion(df_test, "Test")

    # Analyse du genre
    analyze_gender(df_train, "Train")
    analyze_gender(df_val, "Validation")
    analyze_gender(df_test, "Test")

    # Vérification des images
    check_image_files(df_train, IMAGE_DIR, "Train")

    # Analyse des noms de fichiers
    analyze_filename_patterns(df_train, "Train")

    # Répartition globale
    print(f"\n{'='*60}")
    print("📊 RÉPARTITION GLOBALE")
    print(f"{'='*60}")
    total = len(df_train) + len(df_val) + len(df_test)
    print(f"Train : {len(df_train)} échantillons ({len(df_train)/total*100:.1f}%)")
    print(f"Validation : {len(df_val)} échantillons ({len(df_val)/total*100:.1f}%)")
    print(f"Test : {len(df_test)} échantillons ({len(df_test)/total*100:.1f}%)")
    print(f"Total : {total} échantillons")

    print("\n✅ Analyse terminée !")

if __name__ == "__main__":
    main()