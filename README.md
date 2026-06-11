
# AdvRobust-Stats: Robustness Evaluation and Statistical Feature-Reduction Defense for Deep Neural Networks

`AdvRobust-Stats` is an open-source PyTorch-based framework designed to evaluate the vulnerability of Deep Neural Networks (DNNs) under adversarial attacks and explore lightweight statistical dimensionality-reduction techniques as a defense mechanism.

The core research objective of this project is to analyze how high-dimensional adversarial perturbations propagate through deep representations, and to investigate whether statistical reconstruction methods (such as Principal Component Analysis, PCA) can filter out adversarial noise in latent spaces without incurring high retraining costs.

---

## 1. Project Overview & Methodology

Adversarial attacks, such as FGSM and PGD, introduce imperceptible high-dimensional perturbations that easily mislead deep classifiers. While adversarial training is effective, it is computationally expensive.

This project implements a lightweight defense pipeline:

1. **Feature Extraction**: Extracts latent high-dimensional feature maps from intermediate layers of a target DNN (e.g., `ResNet-50` or `MobileNetV2`).
2. **Statistical Dimensionality Reduction**: Projects the latent features into a lower-dimensional principal component subspace using **PCA** or **Kernel PCA**.
3. **Reconstruction & Denoising**: Reconstructs the features back to the original dimension. The reconstruction process naturally discards minor, high-frequency adversarial perturbations (acting as a statistical low-pass filter).
4. **Classification**: Feeds the reconstructed, denoised features into the remaining classification layers.

```
[Input Image (Clean/Adv)] -> [Backbone Layers] -> [High-Dim Latent Features] 
                                                        |
                                            [PCA Dimensionality Reduction]
                                                        |
                                            [Statistical Reconstruction] 
                                                        |
                                                        v
                                          [Classification Head] -> [Prediction]
```

---

## 2. Key Features & Specifications

This repository defines and implements the following requirements:

### A. Core Model & Dataset Support

* **Dataset**: Support for standard benchmarks including `CIFAR-10` and `MNIST`.
* **Target Models**: Implementations of `ResNet-18/50` and `VGG-16` optimized for classification tasks.

### B. Adversarial Attack Suite

* **White-box Attacks**: Fast Gradient Sign Method (**FGSM**) and Projected Gradient Descent (**PGD**).
* **Configuration**: Configurable perturbation magnitude ($\epsilon$) and step size ($\alpha$) to analyze the model's degradation curves.

### C. Statistical Defense Layer

* **Subspace Defense**: Integrating scikit-learn-based `PCA` / `Kernel-PCA` into PyTorch's forward propagation pass.
* **Adversarial Training**: Standard adversarial training as a baseline for comparative analysis.

### D. Comprehensive Evaluation Metrics

* **Robustness Curve**: Classification accuracy plotted against perturbation budgets ($\epsilon$).
* **Variance Analysis**: Analysis of the cumulative explained variance of clean vs. adversarial features in the latent space.

---

## 3. Repository Structure

```directory
├── data/                   # Directory to store CIFAR-10 / MNIST
├── models/
│   ├── __init__.py
│   └── classifiers.py      # Target DNN definitions (ResNet, VGG, etc.)
├── defense/
│   ├── __init__.py
│   └── statistical_pca.py  # Latent space PCA projection and reconstruction
├── utils/
│   └── attack.py           # FGSM and PGD attack implementations
├── main.py                 # Main pipeline runner (Training, Attacking, Defending)
├── evaluate.py             # Evaluation scripts and visualization generation
├── requirements.txt        # Package dependencies
└── README.md
```

---

## 4. Installation & Setup

### Requirements

* Python 3.8+
* PyTorch 1.12+ (CUDA compatible)
* Torchattacks
* Scikit-Learn
* Matplotlib & Seaborn

To install dependencies:

```bash
pip install -r requirements.txt
```

---

## 5. Usage Guide

### Step 1: Train the Target Base Model

Train a standard ResNet-18 classifier on CIFAR-10:

```bash
python main.py --mode train --model resnet18 --dataset cifar10 --epochs 20
```

### Step 2: Extract Latent Features and Fit the Defense Layer

Fit the PCA defense module using clean training representations:

```bash
python main.py --mode fit_pca --model resnet18 --dataset cifar10 --n_components 128
```

### Step 3: Run Evaluation Under Attacks

Evaluate clean accuracy, attacked accuracy, and accuracy under PCA defense across different perturbation levels ($\epsilon \in [0.01, 0.08]$):

```bash
python evaluate.py --model resnet18 --attack pgd --epsilons 0.01 0.02 0.04 0.08
```

---

## 6. Expected Results

The evaluation script generates the following outputs:

1. **Robustness Performance Plot (`robustness_curve.png`)**:
   A line plot comparing:

   * *Baseline Model* (unprotected) under attack.
   * *Adversarial Training Model* under attack.
   * *Our Statistical Defense (PCA reconstructed)* under attack.
2. **Explained Variance Distribution (`variance_distribution.png`)**:
   A visualization of how the eigenvalues distribution differs between clean latent vectors and adversarially perturbed latent vectors.

---

## 7. License

This project is licensed under the MIT License - see the LICENSE file for details.
