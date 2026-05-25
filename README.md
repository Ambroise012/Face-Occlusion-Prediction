# Face Occlusion Prediction

> Projet PyTorch pour prédire le niveau d'occlusion du visage (`FaceOcclusion`) à partir d'images.

---

## 📁 Structure
```bash
project/
├── src/          # Chemins et hyperparamètres
|    ├── dataset.py            # Script principal
|    ├── main.py            # Script principal
|    ├── main.py            # Script principal
|    ├── main.py            # Script principal
|    └── analyze_dataset.py # Analyse exploratoire des données
├── config.py          # Chemins et hyperparamètres
├── dataset.py         # Chargement des données et DataLoaders
├── model.py           # Architecture du modèle (MobileNetV3)
├── train.py           # Boucle d'entraînement
├── inference.py       # Validation et prédictions
├── metrics.py         # Métriques d'évaluation
├── main.py            # Script principal
└── analyze_dataset.py # Analyse exploratoire des données
```

---

## ⚙️ Prérequis
- Python ≥ 3.8
- PyTorch (`torch`, `torchvision`)
- Pandas, NumPy, PIL
- `prettytable`, `tqdm`

Installation :
```bash
pip install - r requirements.txt
```

---
## 🚀 Utilisation

### 1️⃣ **Analyser le dataset**
```bash
python analyze_dataset.py
```
Affiche la répartition des données, les statistiques sur `FaceOcclusion`, le genre, et vérifie les images.

### 2️⃣ **Entraîner le modèle et prédire**
```bash
python main.py
```
- Charge les données (`train.csv`, `test_students.csv`)
- Entraîne un **MobileNetV3** sur `FaceOcclusion`
- Sauvegarde les prédictions dans `test_predictions.csv`

---
## 📊 Données
| Dataset       | Source                          | Taille  | Labels                     |
|---------------|---------------------------------|---------|----------------------------|
| **Train**     | `occlusion_datasets/train.csv`  | 80,000+ | `filename`, `FaceOcclusion`, `gender` |
| **Validation**| (Split depuis Train)             | 20,000  | Idem                        |
| **Test**      | `occlusion_datasets/test_students.csv` | X       | `filename` (pas de label)   |
| **Images**    | `../crops/Crop_224_5fp_100K/`   | 224x224 | Format `.jpg`/`.png`        |

- `FaceOcclusion` : **Float entre 0 et 1** (0 = pas d'occlusion, 1 = occlusion totale).
- `gender` : `0` (femme) ou `1` (homme).

---
## 🎯 Métriques
- **Erreur pondérée** : `error_fn` (poids = `1/30 + ground_truth`)
- **Métrique finale** : Moyenne des erreurs (H/F) + différence absolue entre les genres.

---
## 🔧 Configuration
Modifiez `config.py` pour ajuster :
- `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`
- Chemins des fichiers (`IMAGE_DIR`, `TRAIN_CSV`, etc.)
- Taille de la validation (`VAL_SIZE`)

---
## 📂 Sorties
- **`test_predictions.csv`** : Prédictions `FaceOcclusion` pour le test set.
```

---
**✅ Tout est prêt dans `/home/user/project/` !** 🎉