# Neural Network From Scratch — Handwritten Digit Recognition

A fully connected artificial neural network implemented from scratch in Python using NumPy, achieving 96.60% test accuracy on the MNIST handwritten digit dataset.

## Project Overview

This project explores the mathematics and implementation of artificial neural networks without relying on machine-learning frameworks such as TensorFlow or PyTorch.

Starting with a simple XOR classifier, the project develops into a handwritten digit recognition system with a custom graphical interface.

## Features

- Neural network built from scratch using NumPy
- Forward propagation and backpropagation
- Gradient descent and mini-batch training
- ReLU and softmax activation functions
- 96.60% accuracy on the MNIST test set
- Interactive handwritten digit recognition application
- Real-time predictions and confidence visualisation
- Trained model saving and loading
- Training accuracy visualisation

## Neural Network Architecture

 Layer  Neurons 
------
 Input  784 (28 × 28 image pixels) 
 Hidden  128 (ReLU activation) 
 Output  10 (softmax activation) 

The model was trained on 60,000 MNIST training images and evaluated on 10,000 separate test images.

## Technologies

Python, NumPy, Matplotlib, Pillow, Tkinter

## Installation

Clone the repository and install dependencies

```bash
git clone httpsgithub.com/Isaacbakerr/neural-network-from-scratch.git
cd neural-network-from-scratch
py -m pip install -r requirements.txt
```

## Running the Application

```bash
py digit_recogniserdraw_digits.py
```

Draw a number from 0 to 9 and see the network's prediction.

## Training the Model

```bash
py digit_recogniserdigit_network.py
```

This trains the network and saves the weights.

## Results

MNIST Test Accuracy 96.60%

This result was achieved using a fully connected neural network with one hidden layer, trained with NumPy.

## Future Improvements

- Additional hidden layers and architecture experiments
- More systematic evaluation and benchmarking
- Improved generalisation to mouse-drawn digits
- Unit tests and gradient checking
- Further interactive visualisations

## Author

Isaac Louis Baker  
BSc (Hons) Computer Science — University of Liverpool