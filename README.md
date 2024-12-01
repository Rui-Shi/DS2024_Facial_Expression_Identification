# DS2024_facial_expression
Data Science Project for Erdos institution (fall 2024)

# Wiki:
*Folder:*
- raw_data: the original training and testing images.
- train_balanced: the training data set after data augmentation, making sure the trainning data set is balanced.
- raw_data_to_csv: a csv containing the greyscale of each image.
- eigenface: the classifiers using eigenfaces for data preprocessing.
- Gabor Filter: the classifiers using Gabor filter for image processing.
- MediaPipe: the classifiers using mediapipe for feature extraction.

# Facial Emotion Recognition (FER) Project

## Background and Introduction

### Motivation
Facial expressions are a universal and powerful form of non-verbal communication. Recognizing these expressions has significant implications in various domains, including:
- Enhancing **human-computer interaction** for more natural and intuitive interfaces.
- Improving **accessibility technologies**, enabling inclusive systems for individuals with communication challenges.
- Powering impactful applications such as:
  - **Mental health monitoring**, where early detection of emotional states can inform timely interventions.
  - **Customer sentiment analysis**, aiding businesses in understanding user behavior.

Despite recent advances in machine learning and computer vision, developing an **accurate and computationally efficient Facial Emotion Recognition (FER) system** remains challenging. Key obstacles include:
- **Imbalanced datasets**: Certain emotions are underrepresented in available datasets.
- **Real-world variability**: Variations in lighting, pose, and occlusions complicate recognition.
- **Resource constraints**: Time and memory consumption for large-scale FER models pose practical challenges.

---

### Dataset
This project leverages the **FER-2013 dataset**, a benchmark dataset sourced from Kaggle. The dataset consists of:
- **48x48 pixel grayscale facial images**, classified into seven emotion categories:
  - **Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral**
- **Data split**:
  - **28,709 training images**
  - **7,178 testing images**

FER-2013 is widely used in the FER domain, providing a diverse and challenging dataset for evaluating models under realistic conditions.

---

### Goal
The goal of this project is to develop a **comprehensive Facial Emotion Recognition system** that combines:
1. **Traditional machine learning** approaches (e.g., feature-based methods).
2. **Deep learning** techniques using state-of-the-art neural network architectures.

The aim is to produce **well-tuned, efficient models** for deployment in real-world applications.

---

### Key Performance Indicators (KPIs)
The project will evaluate the performance of the FER system using the following metrics:
1. **Accuracy**: The overall proportion of correctly predicted samples.
2. **F1-Score**: The harmonic mean of precision and recall, measuring the model's robustness across all classes.
3. **Confusion Matrix**: A detailed breakdown of classification performance for each emotion category.
4. **Time/Space Complexity**: Evaluating the system's computational efficiency, including **inference time** and **memory consumption**.

---

### Related Work
The **FER-2013 dataset** has inspired extensive research in facial emotion recognition. Notable prior work includes:
- **Feature-based simple classifiers**: Logistic regression and support vector machines, leveraging handcrafted features such as Histogram of Oriented Gradients (HOG) or Local Binary Patterns (LBP).
- **Neural networks**: Baseline deep learning approaches such as Convolutional Neural Networks (CNNs).

This project seeks to expand on these baselines by:
- Introducing **advanced data preprocessing** and augmentation techniques.
- Exploring **ensemble methods** for enhanced model accuracy.
- Integrating **feature extraction methods** with traditional and deep learning pipelines to provide a comprehensive comparison.
