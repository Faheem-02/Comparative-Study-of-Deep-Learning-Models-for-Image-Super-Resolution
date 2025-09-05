

  <h1>Comparative Study of Deep Learning Models for Image Super-Resolution</h1>
  <p><strong>Performance Evaluation of SRCNN and SRGAN for Geospatial Image Super-Resolution: A Comparative Study on Metrics and Data Preparation Approaches</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Repo Size](https://img.shields.io/github/repo-size/Faheem-02/Comparative-Study-of-Deep-Learning-Models-for-Image-Super-Resolution)](https://github.com/Faheem-02/Comparative-Study-of-Deep-Learning-Models-for-Image-Super-Resolution)
  
<p align="center">
  <em>Enhancing geospatial imagery with SRCNN & SRGAN — from low-res to high-res masterpieces 🚀🛰️</em>
</p>


## 🌟 Overview
This repository presents a comparative analysis of two pioneering deep learning models for image super-resolution (SR) applied to geospatial imagery: **SRCNN** (Super-Resolution Convolutional Neural Network) and **SRGAN** (Super-Resolution Generative Adversarial Network). We evaluate their performance on metrics like PSNR, SSIM, and MSE, using augmented datasets of low-resolution (LR) and high-resolution (HR) satellite images.

The study focuses on upscaling geospatial images (e.g., from 64x64 to 256x256) for applications in remote sensing, urban planning, and environmental monitoring. Key insights include how SRGAN's adversarial training produces more perceptually realistic results compared to SRCNN's MSE-based approach.

This project is inspired by research on super-resolution of Indian Remote Sensing (IRS) satellite imagery, adapting techniques to enhance medium-resolution (5m/pixel) images to high-resolution (1m/pixel) outputs for better analysis in land monitoring and disaster management.

_Note:_Due to limited resources we trained both the models for 80epochs, To achieve better results we need to train the models for more epochs as claimed by the original research papers, especially for SRGAN model which requires a training for days and weeks

**Dataset Sources:**
The dataset used in this project cannot be shared due to security reasons

## ✨ Features- 
-**Model Implementations:**
  - [SRCNN](src/srcnn.py)
  - [SRGAN Final](src/srgan-final.py)
  - [SRGAN Variant](src/srgan-f1.py)
- **Evaluation Metrics:** PSNR, SSIM, MSE with visualizations (histograms, error maps, bar charts).
- **Data Preparation:** Image loading, augmentation, and RGB/BGR handling for VGG compatibility.
- **Training Loops:** Custom learning rate schedules, adversarial training for SRGAN.
- **Comparative Analysis:** Side-by-side performance on original vs. new datasets.
- **Visualizations:** Sample comparisons, loss curves, metric distributions, and error analysis.

## 🛠️ Installation
   ```bash
   git clone https://github.com/Faheem-02/Comparative-Study-of-Deep-Learning-Models-for-Image-Super-Resolution.git
   cd Comparative-Study-of-Deep-Learning-Models-for-Image-Super-Resolution
