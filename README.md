# Semantic Segmentation Projects using Deep Learning

## Overview

This repository focuses on **semantic segmentation**, a fundamental computer vision task where **each pixel in an image is assigned a semantic class** (for example: road, building, vehicle, pedestrian, background).  
Unlike image classification, which produces a single label per image, semantic segmentation provides **dense, pixel-level understanding** of visual scenes.

These notebooks are part of a **Deep Learning with TensorFlow and Keras** learning track and demonstrate a progression from **classical segmentation architectures** to **modern architectures and foundation models**.

---

## Why Semantic Segmentation Matters

Semantic segmentation is widely used in real-world applications such as:
- Autonomous driving (road, lane, vehicle, pedestrian detection)
- Medical imaging (organ and tumor segmentation)
- Robotics and navigation
- Satellite and aerial image analysis

The projects here emphasize both **model architecture design** and **practical training considerations**, including loss functions and data pipelines.

---

## 1) Fully Convolutional Network (FCN)

**Notebook:** `SemanticSegmentation_using_FCN.ipynb`

Fully Convolutional Networks (FCNs) were among the first deep learning models designed specifically for semantic segmentation.  
The key idea is to **replace fully connected layers with convolutional layers**, allowing the network to output a prediction map with the same spatial structure as the input.

### Why FCN works
- Preserves spatial information
- Enables end-to-end pixel-wise prediction
- Serves as the foundation for later segmentation models

### Code snippet
```python
x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
x = Conv2D(num_classes, (1,1), activation='softmax')(x)
```

---

## 2) U-Net Architecture

**Notebook:** `UNet.ipynb`

U-Net improves on FCN by using an **encoder–decoder architecture with skip connections**.  
The encoder captures high-level semantic information, while the decoder restores spatial resolution.

### Why skip connections are important
- Recover fine-grained spatial details
- Improve boundary and edge prediction
- Especially effective for small or thin objects

### Code snippet
```python
c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
p1 = MaxPooling2D((2,2))(c1)
```

---

## 3) U-Net on CamVid Dataset

**Notebook:** `UNet_CamVidDataset.ipynb`

This notebook applies U-Net to the **CamVid road-scene dataset**, which contains multiple semantic classes such as road, sky, buildings, and vehicles.

### What this demonstrates
- Multi-class segmentation
- Dataset-specific preprocessing
- Training and evaluation on real-world data

### Code snippet
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 4) DeepLabV3 with Dice Loss

**Notebook:** `DeepLabv_CamVid_Dice_Loss.ipynb`

DeepLabV3 is an advanced segmentation architecture that uses **atrous (dilated) convolutions** to capture multi-scale context.

This notebook uses **Dice Loss**, which is particularly useful when dealing with **class imbalance**.

### Dice Loss intuition
Dice Loss measures overlap between predicted and ground-truth masks and emphasizes foreground regions.

### Code snippet
```python
dice = (2 * intersection) / (union + smooth)
```

---

## 5) DeepLabV3 with Cross-Entropy Loss

**Notebook:** `DeepLabV3_onRoad_CELoss.ipynb`

This notebook trains DeepLabV3 using **Categorical Cross-Entropy Loss**, a standard and stable loss for multi-class segmentation.

### Comparison with Dice Loss
- Cross-Entropy: stable, widely used
- Dice Loss: better for imbalanced datasets

### Code snippet
```python
loss = tf.keras.losses.CategoricalCrossentropy()
```

---

## 6) Custom Data Loader for Segmentation

**Notebook:** `Segmentation_CustomDataLoader.ipynb`

Segmentation tasks require paired **image–mask datasets**.  
This notebook builds a **custom data loader** to correctly load, preprocess, and batch these pairs.

### Why custom loaders are needed
- Masks require different preprocessing than images
- Enables flexible dataset organization
- Supports large-scale training pipelines

### Code snippet
```python
dataset = dataset.map(load_image_mask).batch(batch_size)
```

---

## 7) Segment Anything Model (SAM) Inference

**Notebook:** `SAM_Inference.ipynb`

This notebook demonstrates inference using the **Segment Anything Model (SAM)**, a foundation model for segmentation.

### Key idea behind SAM
- No task-specific training required
- Uses prompts (points or boxes)
- Enables zero-shot segmentation

### Code snippet
```python
masks = predictor.predict(
    point_coords=points,
    point_labels=labels
)
```

---

## How These Models Progress

The notebooks follow a logical progression:
1. FCN → basic pixel-wise prediction  
2. U-Net → improved spatial detail via skip connections  
3. DeepLabV3 → multi-scale context modeling  
4. Custom loaders → real-world data handling  
5. SAM → foundation models and zero-shot segmentation  

---


## Requirements

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

---


## Author

**Manasa Vijayendra Gokak**

Graduate Student – Data Science  
