# Developing an Image Classifier with Deep Learning

This project implements an **image classification application** using **PyTorch**. The application trains a deep learning model on a dataset of flower images, then uses the trained model to classify new images. It demonstrates how to build, train, and deploy a neural network both in a Jupyter Notebook and as a command-line application.

---

## 📘 Project Overview

In this project, I built a **convolutional neural network (CNN)** capable of classifying flower images into different categories. The project consists of two main parts:

1. **Model Development in Jupyter Notebook**  
   - Build and train an image classifier using a deep neural network in PyTorch.  
   - Save the trained model as a checkpoint for later use.

2. **Command-Line Application**  
   - Convert the notebook implementation into two Python scripts:
     - `train.py` — trains the model and saves the checkpoint.  
     - `predict.py` — uses a trained model to predict the class of an input image.  


## ⚙️ Project Structure
```
├── train.py # Trains a new model on a dataset and saves it as a checkpoint
├── predict.py # Loads a trained model and predicts image classes
├── cat_to_name.json # Mapping of category labels to real flower names
├── checkpoint.pth # Saved model checkpoint
├── Classifier_Part1.ipynb # Jupyter notebook version
└── README.md # Project documentation
```

---
## 📂 Dataset
This project uses the [Flower Classification dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).  
Download the dataset and place it in a folder named `flowers/` in the project directory before training.

---

## 🧠 Features

- Train a deep neural network using transfer learning (e.g., VGG or AlexNet).  
- Save and load model checkpoints.  
- Predict top-k most probable classes for an image.  
- Support for GPU training and inference.  
- Command-line arguments for flexible configuration.  

---

## 🖥️ Usage

### 1. Train the Model

```bash
python train.py data_directory
```
Options:

- --save_dir save_directory         # Directory to save checkpoints
- --arch "vgg13"                    # Choose model architecture (e.g., vgg13, alexnet)
- --learning_rate 0.01              # Set learning rate
- --hidden_units 512                # Set number of hidden units
- --epochs 20                       # Set number of training epochs
- --gpu                             # Use GPU for training

Example:
```bash
python train.py flowers --arch vgg13 --epochs 10 --gpu
```

### 2. Predict Image Class
```bash
python predict.py /path/to/image checkpoint
```

Options:

- --top_k 3                         # Return top K most likely classes
- --category_names cat_to_name.json # Use mapping of categories to real names
- --gpu                             # Use GPU for inference


Example:
```bash
python predict.py lotus.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```
## 🧩 Technologies Used

- Python 3
- PyTorch
- NumPy
- Pandas
- Matplotlib
- argparse (for command-line interface)
- PIL (for image processing)

The trained model checkpoint (checkpoint.pth) is not included in this repository due to GitHub’s file size limitations.
You can download the model from the link below and place it in the project directory to enable inference and evaluation.
https://drive.google.com/drive/folders/1EaXMxZbR7n46aMgCnAQMh1Wnc-6723Yu?usp=drive_link
👉 Download checkpoint.pth

Once downloaded, ensure the file is located in the same directory as predict.py before running predictions

## 📈 Results
- Successfully trained a convolutional neural network achieving high accuracy on the flower dataset.
- Implemented full ML workflow from training to inference.
- Demonstrated command-line application development with customizable hyperparameters.