# Image Segmentation using PASCAL VOC-2012

## Overview
This project focuses on image segmentation using the PASCAL VOC-2012 dataset, a widely used dataset in computer vision containing 20 object classes. The goal is to perform foreground vs background segmentation, correctly identifying and separating the main subject of an image from its background.

Two model architectures are built and compared:
- A **U-Net** based segmentation model
- A **Modified U-Net** that replaces Conv2DTranspose upsampling with bilinear interpolation using UpSampling2D followed by standard convolution

The project was completed as part of a deep learning course and run on Google Colab using GPU support.

## Code
The full implementation can be found in the notebook: [`Neural_Network_Image_Segmentation_.ipynb`](Neural_Network_Image_Segmentation_.ipynb)

---

## Approach

### 1. Data Preparation
- Downloaded and explored the PASCAL VOC-2012 dataset
- Visualised sample images alongside their corresponding segmentation masks
- Images and masks resized to a fixed dimension and masks converted to binary format for foreground vs background classification
- Dataset split into training, validation, and test sets
- Efficient TensorFlow tf.data.Dataset pipeline created for optimised data loading and processing during training

### 2. Model Architecture — U-Net
A U-Net architecture was built for the segmentation task. U-Net is a popular architecture for image segmentation due to its encoder-decoder structure with skip connections, which helps preserve spatial information during upsampling.

### 3. Training
- Both models trained for 20 epochs
- Performance evaluated on the test set using accuracy and loss
- Segmentation outputs visualised and compared against ground truth masks

---

## Results

### Original U-Net Model
| Metric | Value |
|---|---|
| Test Accuracy | 80.99% |
| Test Loss | 0.4031 |

- Training and validation accuracy both showed an upward trend throughout training
- Validation accuracy plateaued at around 78% by the end of training
- Some overfitting observed — training accuracy consistently higher than validation accuracy
- Predicted masks successfully captured the general shape and location of foreground elements, though boundary precision showed room for improvement

### Modified U-Net Model (UpSampling2D + Conv2D)
The Conv2DTranspose layers in the decoder were replaced with UpSampling2D using bilinear interpolation followed by a standard Conv2D layer.

| Metric | Original U-Net | Modified U-Net |
|---|---|---|
| Test Accuracy | 80.99% | ~78% (peak) |
| Validation Stability | Stable | Less stable |
| Boundary Sharpness | Softer edges | Sharper boundaries |

- The modified model learned quickly in early epochs but showed instability in later epochs
- Validation loss increased towards the end of training, suggesting increased susceptibility to overfitting
- Visually, the modified model produced sharper boundary outlines, while the original model showed higher overall confidence in its predictions
- Both models handled the background well with minimal false positives

---

## Key Observations
- The original U-Net model showed more stable training and better generalisation overall
- The modified model produced visually sharper segmentation boundaries but was less stable in later training epochs
- Both models performed well on the foreground vs background task but showed room for improvement in edge detection and segmentation clarity
- The choice between the two architectures depends on whether stability or boundary sharpness is the priority for the specific use case

---

## Dataset
[PASCAL VOC-2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 
- 20 object classes
- Task: Binary foreground vs background segmentation

---

## Tools & Libraries
- **Language:** Python
- **Framework:** TensorFlow, Keras
- **Environment:** Google Colab (GPU)

---
