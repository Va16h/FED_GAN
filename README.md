# Federated Generative Adversarial Networks for Debugging a Primary ML Model

## Introduction

This repository contains the source code and instructions for reproducing the results from the paper titled "Generative Models for Effective ML on Private, Decentralized Datasets." The paper discusses a method for debugging a primary machine learning (ML) model deployed within a mobile app, with a specific focus on a handwriting image classifier used in a banking app. The data for this model is private and decentralized, residing on users' mobile devices. To model this decentralized data, the paper uses the Federated EMNIST dataset.

## Prerequisites

Before you begin, make sure you have the following Python packages installed:

```bash
pip install absl-py
pip install attr
pip install dm-tree
pip install numpy
pip install Pillow
pip install tensorboard
pip install tensorflow
pip install tensorflow-federated
pip install tensorflow-gan
pip install tensorflow-privacy
```
## EMNIST Classifier (Primary Model)
The EMNIST classifier, which is the primary model, is defined as a Keras model. You can find a pre-trained Keras model instance in the ```experiments/emnist/classifier:emnist_classifier_model``` library. If you wish to retrain the EMNIST classifier from scratch, use the ```experiments/emnist/classifier:train_emnist_classifier_model``` script.

## Bug: Pixel Intensity Inversion
A bug has been introduced in the application, causing the inversion of pixel intensities (black and white flip) in a fraction of user data. To observe the impact of this bug on classification accuracy, you can run the ```experiments/emnist/classifier:measure_misclassification_of_users``` script.

## Federated GAN (Auxiliary Model)
Federated Generative Adversarial Networks (GANs) can be trained to reproduce data examples for either the normal or buggy data. Comparing the generated content of a GAN trained on normal data with that of a GAN trained on buggy data can help identify the bug.

## Preprocessing - Data Filtering for Debugging
In this section, we discuss the preprocessing steps necessary for filtering user and example data in order to identify and debug issues related to a bug within the mobile app's primary machine learning model. The bug in question involves the inversion of pixel intensities (black and white flip) in a fraction of user data. Since user data is non-inspectable, we need to approximate the separation of 'normal' data from 'buggy' data based on factors that reasonably correlate with the bug's presence. Two filtering methods are supported:

### Filtering by User
In the filtering method "by_user," we set user accuracy thresholds. Users with EMNIST classification accuracy above the threshold are considered to have 'normal' data, while users with accuracy below the threshold are considered to have 'buggy' data. This method is employed in the main body of the paper, specifically in Section 6.

### Filtering by Example
In the "by_example" filtering method, examples are classified. Examples that classify correctly are considered 'normal,' while those that misclassify are considered 'buggy.' The results obtained using this filtering method are presented in the Appendix.

### Data Filtering Workflow
To perform the data filtering, we use scripts that apply the bug to a given percentage of users. Subsequently, we create sets of users/examples based on the chosen filtering method, either by_user or by_example. To save computation time when conducting repeated experiments, the results of this filtering process are saved in CSV files. This allows experiments to load which users are part of a training population (if filtered by_user) or which examples for each user are included in the training set (if filtered by_example).

### Using Precomputed Filtered Data
To access the precomputed filtered data and utilize it in your experiments, you can leverage the utilities provided in the ```experiments/emnist/preprocessing:filtered_emnist_data_utils``` library. If you wish to redo the precomputation and generate new CSV files, follow the instructions below:

### Filtering Users
To filter users and generate new CSV files, perform the following steps:

Build and run the ```experiments/emnist/preprocessing:filter_users``` script.

This script will apply the bug to a specified percentage of users and filter them based on the chosen user accuracy thresholds.

The resulting filtered user data will be saved in CSV files for later use.

### Filtering Examples
To filter examples and generate new CSV files, follow these steps:

Build and run the ```experiments/emnist/preprocessing:filter_examples``` script.

This script will apply the bug to a specified percentage of users and classify their examples as 'normal' or 'buggy.'

The filtered example data will be saved in CSV files for subsequent experiments.

By following these preprocessing steps, you can effectively separate 'normal' and 'buggy' data and use them for debugging your primary machine learning model deployed within the mobile app.

## Training
The ```experiments/emnist:train``` script is used to train a federated GAN on EMNIST data. You can specify flags to select the subset of data to train on (filtered by_user, by_example, or not filtered), set training hyperparameters (including differential privacy hyperparameters for clipping/noising), and determine how often to save model checkpoints or generated images.
