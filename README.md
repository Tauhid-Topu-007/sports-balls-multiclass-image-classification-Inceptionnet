# Sports Ball Multi-Class Image Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/sports_ball_classification.ipynb)

A deep learning project implementing **GoogLeNet (Inception v1)** architecture for multi-class classification of sports balls. The model classifies images of 15 different types of sports balls including cricket balls, footballs, golf balls, hockey balls, hockey pucks, rugby balls, shuttlecocks, table tennis balls, tennis balls, volleyballs, and more.

## 📊 Dataset

The dataset used is the **Sports Balls Multi-class Image Classification** dataset from Kaggle:
- **Source**: [Sports Balls Multi-class Image Classification](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification)
- **License**: CC0-1.0
- **Total Images**: 7,328 images
- **Number of Classes**: 15 sports ball categories
- **Train/Validation Split**: 80/20 (5,863 training / 1,465 validation)

### Dataset Classes
The dataset contains the following sports ball categories:
- Cricket Ball
- Football
- Golf Ball
- Hockey Ball
- Hockey Puck
- Rugby Ball
- Shuttlecock
- Table Tennis Ball
- Tennis Ball
- Volleyball
- (And 5 additional sports balls)

## Model Architecture

This project implements **GoogLeNet (Inception v1)** architecture, which introduced the concept of inception modules. The architecture features:

### Key Components:
- **Inception Modules**: Efficient multi-scale feature extraction using parallel convolution operations (1×1, 3×3, 5×5) and max pooling
- **Auxiliary Classifiers**: Two auxiliary classifiers added to combat the vanishing gradient problem and provide regularization
- **Global Average Pooling**: Replaces fully connected layers to reduce parameters and prevent overfitting

### Model Parameters:
- **Total Parameters**: 10.3 million (trainable)
- **Input Size**: 224 × 224 × 3 (RGB images)
- **Output Classes**: 15

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow numpy matplotlib seaborn pandas
