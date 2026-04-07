# 🧠 Dépistage Préliminaire du Trouble du Spectre de l'Autisme par Deep Learning

> **Projet Tutoré — Semestre S6**  
> Filière : Sciences Mathématiques et Informatique (SMI)  
> Faculté des Sciences Dhar El Mahraz — USMBA, Fès  
> Année universitaire : 2024/2025

---

## 👥 Auteurs

| Nom | Rôle |
|-----|------|
| **Jaouad EL MORABIT** | Réalisateur |
| **Mohamed ZAARI** | Réalisateur |

**Encadrant :** Pr. Ilyasse ABOUSSALEH  
**Soutenu le :** Samedi 24 mai 2025

---

## 📋 Description du projet

Ce projet explore le potentiel des **réseaux de neurones convolutifs (CNN)** pour le dépistage préliminaire du **Trouble du Spectre de l'Autisme (TSA)** chez les enfants, à partir d'images faciales. Face aux défis d'accès au diagnostic précoce, particulièrement dans le contexte marocain, cette recherche vise à établir les fondements techniques d'un futur outil de dépistage accessible via dispositif mobile.

---

## 🎯 Objectifs

- Développer et évaluer des modèles CNN capables d'identifier des marqueurs visuels potentiels du TSA
- Optimiser les performances via **transfer learning** et **fine-tuning**
- Implémenter une stratégie d'**ensemble learning** pour une meilleure robustesse
- Atteindre une précision de classification supérieure à **85%** avec un bon équilibre sensibilité/spécificité
- Poser les bases techniques d'une future application mobile de dépistage accessible

---

## 🗂️ Structure du projet

```
.
├── data/
│   ├── train/
│   │   ├── Autistic/          # 1280 images
│   │   └── Non_Autistic/      # 1280 images
│   ├── val/
│   │   ├── Autistic/          # 50 images
│   │   └── Non_Autistic/      # 50 images
│   └── test/
│       ├── Autistic/          # 150 images
│       └── Non_Autistic/      # 150 images
├── models/
│   ├── base/                  # Modèles sans fine-tuning
│   └── fine_tuned/            # Modèles après fine-tuning
├── notebooks/
│   └── image-classification-autistic-children.ipynb
├── requirements.txt
└── README.md
```

---

## 🏗️ Architectures utilisées

| Architecture | Paramètres | Particularité |
|---|---|---|
| **MobileNetV2** | 3,5 M | Optimisé pour mobile, meilleur rapport performance/efficience |
| **DenseNet121** | 8,1 M | Connexions denses entre toutes les couches |
| **InceptionV3** | 23,9 M | Traitement multi-échelle en parallèle |
| **VGG19** | 20,5 M | Architecture profonde à filtres 3×3 |

---

## ⚙️ Méthodologie

### 1. Prétraitement des données
- Redimensionnement uniforme à **224×224 pixels**
- Normalisation des valeurs de pixels dans `[0, 1]`
- **Data augmentation** : rotations (±10°), translations, zooms, variations de luminosité, retournements horizontaux

### 2. Transfer Learning
Réutilisation des poids pré-entraînés sur **ImageNet** — les premières couches sont gelées, seules les couches supérieures sont adaptées à la détection du TSA.

### 3. Fine-tuning
- Dégel ciblé des couches supérieures
- Taux d'apprentissage différenciés : `0.0001` (couches gelées dégelées) / `0.001` (couches de classification)
- Régularisation via **Dropout (50%)**, **Early Stopping** et **ReduceLROnPlateau**

### 4. Ensemble Learning (Soft Voting)
Combinaison des probabilités prédites par plusieurs modèles :

$$P_{\text{ensemble}}(y=1|x) = \frac{1}{n} \sum_{i=1}^{n} P_i(y=1|x)$$

Deux ensembles évalués :
- Ensemble des **modèles de base**
- Ensemble des **modèles fine-tunés** ← meilleurs résultats

---

## 📊 Résultats

### Modèles de base

| Modèle | Accuracy | F1-Score | AUC |
|--------|----------|----------|-----|
| MobileNetV2 | 83,3% | 0,825 | 0,906 |
| DenseNet121 | 81,3% | 0,801 | 0,895 |
| VGG19 | 78,3% | 0,783 | 0,863 |
| InceptionV3 | 76,3% | 0,758 | 0,849 |

### Après Fine-tuning

| Modèle | Accuracy | F1-Score | AUC |
|--------|----------|----------|-----|
| **MobileNetV2** | **87,7%** | **0,875** | **0,949** |
| InceptionV3 | 86,7% | 0,872 | 0,928 |
| DenseNet121 | 85,0% | 0,842 | 0,926 |
| VGG19 | 81,3% | 0,776 | 0,933 |

### Ensemble Learning

| Approche | Accuracy | F1-Score | AUC |
|----------|----------|----------|-----|
| Ensemble modèles de base | 82,7% | 0,817 | 0,917 |
| **Ensemble modèles fine-tunés** | **89,0%** | **0,885** | **0,955** |

> 🏆 **Meilleur résultat :** Ensemble soft voting des modèles fine-tunés — **89% d'accuracy**, **AUC de 95,5%**, **sensibilité de 93,3%**

---

## 🛠️ Environnement technique

| Outil | Version | Usage |
|-------|---------|-------|
| **Python** | 3.11 | Langage principal |
| **TensorFlow** | 2.12.0 | Framework deep learning |
| **Keras** | (intégré) | API haut niveau, modèles pré-entraînés |
| **Kaggle** | — | GPU NVIDIA Tesla T4 (16 GB), 30h/semaine |
| NumPy, Pandas | — | Manipulation des données |
| Matplotlib, Seaborn | — | Visualisation |
| scikit-learn | — | Métriques d'évaluation |

---

## 🚀 Installation et utilisation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/<nom-du-repo>.git
cd <nom-du-repo>

# Installer les dépendances
pip install -r requirements.txt
```

Le notebook principal est disponible sur Kaggle :  
🔗 [image-classification-autistic-children](https://www.kaggle.com/code/jaouadelmorabit/image-classification-autistic-children)

### Configuration minimale recommandée
- RAM : 16 GB minimum
- GPU : Support CUDA 11.2
- Espace disque : ~5 GB pour les données et les modèles

---

## 📈 Métriques d'évaluation

- **Accuracy** — proportion de prédictions correctes
- **F1-Score** — moyenne harmonique précision/rappel
- **AUC-ROC** — capacité discriminative du modèle
- **Sensibilité (Recall)** — taux de vrais positifs (cas autistiques correctement détectés)
- **Spécificité** — taux de vrais négatifs (cas non-autistiques correctement écartés)

---

## ⚠️ Limitations

- Jeu de données relativement limité (2560 images d'entraînement)
- Absence de validation clinique formelle
- Application mobile non développée dans le cadre de ce projet académique
- Performances à valider sur des populations plus diversifiées

---

## 🔭 Perspectives

- Développement d'une **application mobile** (iOS/Android) via TensorFlow Lite
- Intégration de données **multimodales** (images, vidéo, audio)
- Collaboration avec des **professionnels de santé** pour validation clinique
- Extension à des bases de données plus larges et diversifiées
- Ajustement du seuil de décision selon les priorités cliniques (sensibilité vs spécificité)

---

## ⚖️ Considérations éthiques

Cet outil est conçu comme un **aide au dépistage préliminaire** et ne constitue en aucun cas un diagnostic médical. Toute utilisation clinique future devra :
- Recueillir un **consentement éclairé** des utilisateurs
- Garantir la **protection des données** conformément au RGPD
- S'accompagner d'une **validation par des professionnels de santé qualifiés**

---

## 📚 Références clés

- Goodfellow, Bengio & Courville (2016). *Deep Learning*. MIT Press.
- Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
- Huang et al. (2017). *Densely Connected Convolutional Networks*. CVPR.
- Szegedy et al. (2016). *Rethinking the Inception Architecture*. CVPR.
- Simonyan & Zisserman (2014). *Very Deep Convolutional Networks*. arXiv:1409.1556.
- Zeidan et al. (2022). *Global prevalence of autism: A systematic review update*. Autism Research.

---

## 🏛️ Institution

**Faculté des Sciences Dhar El Mahraz (FSDM)**  
Université Sidi Mohamed Ben Abdellah (USMBA) — Fès, Maroc

---

*Réalisé par Jaouad EL MORABIT & Mohamed ZAARI — FSDM/USMBA, 2024/2025*
