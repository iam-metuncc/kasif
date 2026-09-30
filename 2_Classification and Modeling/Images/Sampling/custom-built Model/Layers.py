#----------------------------------------------------------#
# Custom CNN (layers), new experiments, hybrid sampling #
#----------------------------------------------------------#

import os
import gc
import sys
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten,Conv2D, MaxPool2D, BatchNormalization)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, classification_report)

IMG_SIZE = 100 # 75

# List of categories
CAT = ['no_header', 'value_attr', 'col_0', 'row_0_col_0', 'row_01_col_0','row_012', 'row_01', 'row_0', 'others']
       
NUM_CLASSES = 9

class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


#=========================================================================================#

# This function rints how many samples each class has, plus the shape of the data."""
def display_class_distribution(features, labels):
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    max_len = max(len(item) for item in CAT)

    print("Class\t\tClass Digit\t\t\tcount")
    for index, count in counts.items():
        item = CAT[index]
        print("{:<{width}}\t\t{}\t\t\t{}".format(item, index, count, width=max_len))
    print("---------")
    print("Features dataset: ", features.shape)
    print("Targets dataset: ", labels.shape)
    print("-----------------------------\n")

# Function to load and preprocess images
def load_and_preprocess_images(directory):
    training_data = []
    for item in CAT: # Looping through each category
        path = os.path.join(directory, item)   # Getting the path for each category
        class_num = CAT.index(item)   # Getting the index of the category
        for img in os.listdir(path): # Looping through each image in the category
            img_array = cv2.imread(os.path.join(path, img))  # Reading the image using OpenCV
            
            if img_array.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # This forces the size on the img 
            training_data.append([img_array, class_num]) # Appending image array and class number to training data

    X = np.array([f for f, _ in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) # Looping through each image array and class number and Reshaping X to desired format
    y = np.array([l for _, l in training_data])
    X = X.astype(np.float32) / 255.0 # Normalizing pixel values of X to the range [0, 1]
    return X, y



def sample_training_data(X_train, y_train):
 
    # Flatten each image into a single row of pixels
    X_flat = X_train.reshape(X_train.shape[0], -1)

    
    print("Class Distribution BEFORE sampling (this fold's training data):")
    display_class_distribution(X_flat, y_train)

    # Random undersampling
    rus = RandomUnderSampler(sampling_strategy={7: 800})
    X_flat, y_train = rus.fit_resample(X_flat, y_train)

    
    print("Class Distribution AFTER undersampling:")
    display_class_distribution(X_flat, y_train)

    #SMOTE oversampling of the minority class
    smote = SMOTE(sampling_strategy={0: 800, 1: 800, 2: 800, 3: 800, 4: 800, 5: 800, 6: 800, 8: 800})
    X_flat, y_train = smote.fit_resample(X_flat, y_train)

    
    print("Class Distribution AFTER oversampling (final training set):")
    display_class_distribution(X_flat, y_train)

    # Reshape features back to original image dimensions
    X_train = X_flat.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    return X_train, y_train



# Function to create and train a dummy classifier
def baseline_classifier(X_train, y_train, X_test, y_test):

    # DummyClassifier expects 2D features so we flatten.
    Xtr = X_train.reshape(X_train.shape[0], -1)
    Xte = X_test.reshape(X_test.shape[0], -1)

    dummy_clf = DummyClassifier(strategy='stratified', random_state=42) # creating the dummy classifier 
    dummy_clf.fit(Xtr, y_train) # fitting 
    y_pred = dummy_clf.predict(Xte) # predict 
    
    # evaluations 
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    return (accuracy, f1_macro, recall_macro, precision_macro)


# Function to build the model architecture
def build_model():
    channels = 3  # Channels represent the color components of the image (RGB).
    inputShape = (IMG_SIZE, IMG_SIZE, channels)
    chanDim = -1 # Specify the channel dimension. In this case, -1 indicates the last dimension (channels).
    
    # Initialize the sequential model

    model = Sequential()

    # Add the first convolutional layer
    # Conv2D layer performs 2D convolution on the input image.
    # 256 is the number of filters/kernels to be applied in the layer.
    # (3,3) specifies the size of the convolutional kernel.
    model.add(Conv2D(256, (3, 3), input_shape=inputShape))
    model.add(Activation("relu"))  #Apply ReLU activation function to introduce non-linearity.
    model.add(MaxPool2D(pool_size=(2, 2))) # Apply max pooling to reduce spatial dimensions and extract dominant features.
    model.add(BatchNormalization(axis=chanDim)) # Apply batch normalization to stabilize and speed up training.
    model.add(Dropout(0.2)) # Apply dropout regularization 


    # Add additional convolutional layers with similar configuration
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.2))
    
    # Flatten the feature maps and add dense layers for classification
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax')) # Output layer with softmax activation for multi-class classification.

    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model


# Function to train the model
def run_cv(model, features, labels):
    repeat_accuracy = []
    repeat_f1_macro = []
    repeat_recall_macro = []
    repeat_precision_macro = []

    baseline_repeat_accuracy = []
    baseline_repeat_f1_macro = []
    baseline_repeat_recall_macro = []
    baseline_repeat_precision_macro = []

    for repeat_idx in range(5):
        fold_accuracy = []
        fold_f1_macro = []
        fold_recall_macro = []
        fold_precision_macro = []

        baseline_fold_accuracy = []
        baseline_fold_f1_macro = []
        baseline_fold_recall_macro = []
        baseline_fold_precision_macro = []

       
        rskf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)

        for fold_idx, (train_index, test_index) in enumerate(rskf.split(features, labels)):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # sampling happens here, inside the fold, on training data only. The test set stays untouched and imbalanced.
            X_train_resampled, y_train_resampled = sample_training_data(X_train, y_train)

            # One-hot encode labels for categorical_crossentropy loss.
            y_train_ohe = tf.keras.utils.to_categorical(y_train_resampled, num_classes=NUM_CLASSES)
            y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)


            print(f"\n--- Repeat {repeat_idx + 1}, Fold {fold_idx} ---")
            model.fit(X_train_resampled, y_train_ohe, epochs=20,batch_size=32,callbacks=[GarbageCollectorCallback()])
               
            # Predictions
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            acc = accuracy_score(y_test, y_pred)
            f1_m = f1_score(y_test, y_pred, average='macro')
            rec_m = recall_score(y_test, y_pred, average='macro')
            prec_m = precision_score(y_test, y_pred, average='macro')

            print(
                f"Repeat {repeat_idx + 1}, Fold {fold_idx}, "
                f"Accuracy: {acc:.4f}, "
                f"F1 macro: {f1_m:.4f}, "
                f"Recall macro: {rec_m:.4f}, "
                f"Precision macro: {prec_m:.4f}"
            )

            print('\nClassification Report\n')
            print(classification_report(y_test, y_pred, target_names=CAT))

            fold_accuracy.append(acc)
            fold_f1_macro.append(f1_m)
            fold_recall_macro.append(rec_m)
            fold_precision_macro.append(prec_m)

            # Baseline claissifier
            (b_acc, b_f1_m, b_rec_m, b_prec_m) = baseline_classifier(X_train_resampled, y_train_resampled, X_test, y_test)
            baseline_fold_accuracy.append(b_acc)
            baseline_fold_f1_macro.append(b_f1_m)
            baseline_fold_recall_macro.append(b_rec_m)
            baseline_fold_precision_macro.append(b_prec_m)

            del X_train, X_test, y_train, y_test
            del X_train_resampled, y_train_resampled, y_train_ohe, y_test_ohe
            del y_pred_probs, y_pred
            gc.collect()

            print("------------------")
        print("---------------------------------------------------------------")

        # Averages across the 10 folds of this repeat.
        repeat_accuracy.append(np.mean(fold_accuracy))
        repeat_f1_macro.append(np.mean(fold_f1_macro))
        repeat_recall_macro.append(np.mean(fold_recall_macro))
        repeat_precision_macro.append(np.mean(fold_precision_macro))

        baseline_repeat_accuracy.append(np.mean(baseline_fold_accuracy))
        baseline_repeat_f1_macro.append(np.mean(baseline_fold_f1_macro))
        baseline_repeat_recall_macro.append(np.mean(baseline_fold_recall_macro))
        baseline_repeat_precision_macro.append(np.mean(baseline_fold_precision_macro))

        print(f"\nRepeat {repeat_idx + 1} summary:")
        print(f"  Mean Accuracy:            {repeat_accuracy[-1]:.4f}")
        print(f"  Mean F1 (macro):          {repeat_f1_macro[-1]:.4f}")
        print(f"  Mean Recall (macro):      {repeat_recall_macro[-1]:.4f}")
        print(f"  Mean Precision (macro):   {repeat_precision_macro[-1]:.4f}")
        print("=============================================")

    # Overall means across all 5 repeats.
    print("\n***************************************************")
    print("Overall Results (5 repeats x 10 folds):")
    print("***************************************************")
    print(f"Overall Mean Accuracy:            {np.mean(repeat_accuracy):.4f}")
    print(f"Overall Mean F1 (macro):          {np.mean(repeat_f1_macro):.4f}")
    print(f"Overall Mean Recall (macro):      {np.mean(repeat_recall_macro):.4f}")
    print(f"Overall Mean Precision (macro):   {np.mean(repeat_precision_macro):.4f}")

    print("\n------------")
    print("Baseline (Dummy Classifier):")
    print(f"Baseline Overall Mean Accuracy:            {np.mean(baseline_repeat_accuracy):.4f}")
    print(f"Baseline Overall Mean F1 (macro):          {np.mean(baseline_repeat_f1_macro):.4f}")
    print(f"Baseline Overall Mean Recall (macro):      {np.mean(baseline_repeat_recall_macro):.4f}")
    print(f"Baseline Overall Mean Precision (macro):   {np.mean(baseline_repeat_precision_macro):.4f}")
    print("***************************************************")



def main():

    folder = f'/content/drive/MyDrive/PaperNewExperiments/images_resized_{IMG_SIZE}'

    print(f"Experiment: Custom CNN with hybrid sampling")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Loading images from: {folder}")
    
    features, labels = load_and_preprocess_images(folder)
    print(f"Loaded {len(features)} images of shape {features.shape[1:]}")

    print("\nRaw class distribution:")
    display_class_distribution(features.reshape(features.shape[0], -1), labels)
    
    # Build the model
    model = build_model()
    run_cv(model, features, labels)

if __name__ == "__main__":
    main()
