# **Web Table Classification**

This repository contains the code and data used in my thesis on automatic identification and classification of web tables.
The work combines feature-based machine learning models and image-based deep learning models (CNNs) to classify tables based on their header locations.

# Abstract

Tables are one of the most common ways of presenting structured and complex information. However, the wide variety of web table formats and styles makes it difficult to ensure clear presentation for all users—especially those with visual disabilities who rely on speech-based access.

This project explores the automatic classification of web tables into nine categories based on header location. Unlike previous methods that rely solely on raw HTML features, this work uses both rendered features and rendered images of tables, capturing how they appear to users in a browser.

A combined dataset of 5,437 tables (existing and manually collected/labeled) was used to train and evaluate models. The experiments demonstrate strong performance with:


The primary contribution of this work is advancing automated web table mining and improving accessibility for visually impaired users.

# Repository Structure
**1. Data**

Contains everything related to dataset creation and preprocessing.

Feature Extraction & Collection: Scripts for rendering and extracting table features, along with collected screenshots.

Labeling Tool: A custom annotation tool used by human annotators to assign classes to tables.

Preprocessing: Scripts for cleaning and preparing the labeled dataset for model training.

**2. Classification**

Contains model implementations and experiments for table classification.

Feature-Based Models (ML): Classical machine learning models (Random Forest, SVM, etc.) trained on rendered table features.

Image-Based Models (DL): Deep learning models (CNNs) trained on rendered table images.
