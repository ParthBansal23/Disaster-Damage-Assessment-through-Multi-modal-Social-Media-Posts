# Cross-Platform Multi-Modal Disaster Post Classification

A research project on **cross-platform classification of multi-modal disaster-related social media posts using unsupervised domain adaptation**.

> **Research Note:** This repository accompanies research that is currently under review. The README intentionally provides a high-level description of the methodology and does not disclose the complete proposed formulation, detailed experimental configuration, or final research results.

---

## Overview

Social media platforms generate large volumes of information during disaster events. However, data distributions can vary significantly across platforms due to differences in user behavior, content characteristics, and platform-specific patterns. Furthermore, labeled disaster-related data is often limited.

This project investigates **unsupervised domain adaptation (UDA)** for transferring knowledge from a labeled source domain to an unlabeled target domain for **multi-modal disaster post classification**.

Each social media post is represented using both **text and associated visual information**, and the task is to classify the complete post into one of two categories:

- **Informative**
- **Non-informative**

The project uses **CrisisMMD** as the labeled source domain and a team-curated **Southern California Wildfires 2025 (SCW2025)** dataset from **Bluesky** as the target domain.

---

## Research Motivation

During disaster events, social media can provide valuable information for:

- Disaster assessment
- Situational awareness
- Emergency response
- Crisis information monitoring
- Recovery efforts

However, information characteristics can differ substantially between social media platforms. A model trained on labeled data from one platform may therefore experience a **domain shift** when applied to another platform.

This project explores whether **unsupervised domain adaptation** can improve cross-platform generalization by learning representations that are less sensitive to differences between source and target domains.

---

## Problem Definition

The task is formulated as a **binary multi-modal classification problem**.

Given a social media post consisting of:

- Text
- Image

the model predicts whether the complete post is:

- **Informative**
- **Non-informative**

### Source Domain

- **Dataset:** CrisisMMD
- **Labels:** Available
- **Role:** Labeled source domain

### Target Domain

- **Dataset:** Southern California Wildfires 2025 (SCW2025)
- **Platform:** Bluesky
- **Labels:** Unlabeled during domain adaptation
- **Role:** Target domain for cross-platform adaptation

---

## High-Level Methodology

The project combines **multi-modal representation learning** with **unsupervised domain adaptation**.

The overall pipeline can be summarized as:

```mermaid
flowchart TD
    S["Labeled Source Domain<br/>CrisisMMD"]
    T["Unlabeled Target Domain<br/>SCW2025"]

    S --> ST["Text + Image"]
    T --> TT["Text + Image"]

    ST --> CLIP["CLIP ViT-L/14<br/>Multi-Modal Representation"]
    TT --> CLIP

    CLIP --> FG["Shared Feature Generator"]

    FG --> DA["Domain Alignment"]
    FG --> CD["Class Discrimination"]

    DA --> DB["Dynamic Objective Balancing"]
    CD --> DB

    DB --> C["Post Classification"]

    C --> I["Informative"]
    C --> NI["Non-informative"]


### Block 2 — Continue immediately after Block 1

```markdown
The image and text components of each post are encoded independently using a pretrained **CLIP ViT-L/14** model. The resulting representations are combined to obtain a joint multi-modal representation.

The shared representation is optimized using complementary objectives:

1. **Domain Alignment** — reducing differences between source and target feature distributions.
2. **Class Discrimination** — maintaining meaningful class boundaries during domain adaptation.
3. **Dynamic Objective Balancing** — adaptively balancing the domain-alignment and class-discrimination objectives during training.

The detailed formulation of these objectives is intentionally omitted while the associated research paper is under review.

---

## Multi-Modal Representation

Each post is treated as an **image-text pair**, allowing the model to exploit complementary information from both modalities.

A pretrained **CLIP ViT-L/14** model is used to generate modality-specific representations:

    Text  ──> CLIP Text Encoder  ──> Text Embedding
                                      │
                                      │
    Image ──> CLIP Image Encoder ──> Image Embedding
                                      │
                                      ▼
                               Joint Representation

The resulting image and text representations are combined before being passed to the domain adaptation framework.

---

## Unsupervised Domain Adaptation

The project follows an **unsupervised domain adaptation** setting in which the source domain provides labeled examples while the target domain remains unlabeled during adaptation.

    Labeled Source
       CrisisMMD
           │
           │ Knowledge Transfer
           ▼
    ┌─────────────────┐
    │     Domain      │
    │    Adaptation   │
    │    Framework    │
    └─────────────────┘
           ▲
           │
           │
    Unlabeled Target
       SCW2025

The framework uses adversarial learning to encourage domain-invariant representations while incorporating class-discriminative learning to preserve useful decision boundaries.

A dynamic weighting mechanism is used to balance these objectives during training.

The complete mathematical formulation and detailed optimization procedure are intentionally not included in this README while the research paper is under review.

---

## SCW2025 Dataset

**Southern California Wildfires 2025 (SCW2025)** is a multi-modal disaster social media dataset **curated by our team** from posts collected from **Bluesky** during the Southern California wildfires in 2025.

The dataset contains:

- **20K+ social media posts**
- Textual content
- Images associated with posts
- Post metadata

SCW2025 serves as the **unlabeled target domain** in the cross-platform domain adaptation experiments.

For target-domain evaluation, a randomly sampled subset of **128 image-text pairs** was manually annotated as:

- Informative
- Non-informative

These annotations were used for target-domain evaluation and were not used during the domain adaptation training process.

> The SCW2025 dataset is not redistributed in this repository.

---

## Preliminary Experiments

Before adopting the domain-adaptation approach, preliminary experiments were conducted using conventional modality-specific representation models.

These included:

- **ResNet** for image representation
- **BERT** for text representation

These experiments served as initial baselines and motivated further investigation into joint multi-modal representations and cross-platform domain adaptation.

The corresponding exploratory implementations are retained in the repository.

---

## Experimental Configurations

The research implementation investigates cross-platform performance under several configurations:

### 1. Source Baseline

A model trained using labeled source-domain data without applying domain adaptation.

### 2. Domain-Adversarial Neural Network (DANN)

A conventional domain-adaptation baseline using adversarial domain alignment.

### 3. Static Balance Factor

The proposed domain-adaptation framework using a fixed balance between domain alignment and class-discriminative learning.

### 4. Dynamic Balance Factor

The proposed framework with an adaptive mechanism for balancing the domain-alignment and class-discrimination objectives.

Detailed experimental comparisons and quantitative results are intentionally not included in this README while the associated research work is under review.

---

## Implementation

The experiments were implemented using **PyTorch**.

### Core Components

- **CLIP ViT-L/14** for multi-modal representation
- MLP-based feature generator
- Adversarial domain discriminator
- Multiple classification heads for class-discriminative learning
- Dynamic objective balancing
- Adam optimizer

The feature generation and classification components are implemented using multilayer perceptrons with nonlinear activation and regularization.

---

## Repository Structure

    Disaster-Damage-Assessment-through-Multi-modal-Social-Media-Posts/
    │
    ├── Domain_Adaptation.ipynb
    ├── RESNET_for_post_image.ipynb
    ├── bert_with_layers.ipynb
    └── README.md

### `Domain_Adaptation.ipynb`

Contains the primary implementation of the cross-platform multi-modal domain adaptation experiments.

### `RESNET_for_post_image.ipynb`

Contains preliminary experiments using **ResNet-based image representations**.

### `bert_with_layers.ipynb`

Contains preliminary experiments using **BERT-based text representations**.

---

## Datasets

### CrisisMMD

**CrisisMMD** is used as the labeled **source domain** for the cross-platform experiments.

It provides multi-modal disaster-related social media data with labels used for supervised source-domain learning.

### Southern California Wildfires 2025 (SCW2025)

**SCW2025** is the team-curated **target-domain** dataset collected from Bluesky.

It contains **20K+ posts**, associated visual information, and post metadata.

The target domain is treated as unlabeled during domain adaptation.

The datasets are not included in this repository.

---

## Applications

The broader motivation of this work is to explore how heterogeneous and largely unlabeled social media data can support disaster-related information analysis.

Potential applications include:

- Disaster assessment
- Situational awareness
- Emergency response coordination
- Crisis information monitoring
- Recovery and post-disaster analysis

---

## Research Status

This work is part of an ongoing research study, and the associated **research paper is currently under review**.

For this reason, detailed mathematical formulations, complete experimental results, and certain implementation details are intentionally not reproduced in this repository at this stage.

The repository is primarily intended to document the project and its experimental implementation.

---

## Disclaimer

This repository contains research code and experimental implementations. It should not be interpreted as a production-ready disaster response system.

The datasets used in the experiments are subject to their respective licenses and usage conditions and are not included in this repository.

---

## Acknowledgements

This project was carried out as a research project by a team of three members.
