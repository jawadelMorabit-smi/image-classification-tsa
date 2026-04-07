# Image Classification for Autistic Children (Détection du TSA)

This project develops a Deep Learning ecosystem for the early screening of Autism Spectrum Disorder (TSA - Trouble du Spectre de l'Autisme) using facial images of children. The overarching goal is to provide a robust, lightweight model (specifically with mobile-first architectures) suitable for deployment on resource-limited mobile devices. This serves as a complementary mechanism to traditional clinical methods in regions lacking specialized diagnostic expertise.

## Dataset Details
The **AutismDataset** comprises images structurally separated into two distinct binary classes: `Autistic` and `Non_Autistic`. 
- **Training set:** 1280 images per class
- **Validation set:** 50 images per class
- **Testing set:** 150 images per class

## Methodology
The project largely leverages **Transfer Learning** and **Fine-Tuning** to overcome the limited availability of medically labeled data (utilizing ImageNet base knowledge). 
    
### Preprocessing & Data Augmentation
- **Resizing:** Images are uniformly scaled to 224x224.
- **Normalization:** Pixel values rescaled to [0, 1].
- **Data Augmentation:** Geometric (rotations, translations, zoom) and photometric (brightness, reflections) transformations improve model robustness and avoid overfitting without distorting clinically relevant markers.

### Architectures (CNNs)
Four pre-trained backbone architectures were studied and fine-tuned:
- **VGG19**
- **MobileNetV2** (highly relevant for targeted mobile deployment)
- **InceptionV3**
- **DenseNet121**

### Ensemble Learning
To maximize diagnostic reliability, an **Ensemble Learning** approach based on *Soft Voting* was implemented. By averaging the probability predictions of the individual optimized models, the ensemble reduces variance, improves model robustness, and captures a wider array of visual features associated with ASD. Both base and fine-tuned versions of the models were combined into respective ensembles.

## Getting Started

1. Ensure you have the dataset available (expected at `/kaggle/input/autistic-childrens/AutismDataset`).
2. Install the necessary requirements: `tensorflow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
3. Run the Jupyter notebook (`image-classification-autistic-children.ipynb`) sequentially to apply preprocessing, train the base models, apply fine-tuning, evaluate confusion matrices, and formulate the ensemble model.

---
**Authors:** Jawad El Motabit & MZ

