# Intelligent-Waste-Segregator

This repository contains code for classifying various types of solid waste materials into different categories using Convolutional Neural Networks (CNNs). 

Requirements for this code: 1)Tensorflow 2)Keras 3)OpenCv 4)Numpy

Around 300 - 400 images for each class were used for training and 100 images for each class were used for validation. The validation accuracy achieved was roughly 85%-90%.

Number of class = 4 

Class Names - Metal, Plastic, Glass, Paper

Implementation Video - https://www.youtube.com/watch?v=RjB2AAkHBlY

Research Paper Link - https://www.irjet.net/archives/V5/i4/IRJET-V5I4756.pdf

In order to reduce time for training, CUDA was used. The GPU used was Nvidia GTX 1050 Ti.

In this repository, both codes for Tensorflow and Keras have been included. Keras showed better results than Tensorflow and hence Keras was used to train our classifier.
