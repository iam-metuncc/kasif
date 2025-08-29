#----------------------------------------------------------#
# Full new dataset, Sampling with VGG16                #
#----------------------------------------------------------#


import numpy as np  # Importing numpy library for numerical operations
import os  # Importing os library for operating system functionalities
import cv2  # Importing OpenCV library for image processing
import pandas as pd  # Importing pandas library for data manipulation
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras.src.layers import Rescaling
from sklearn.model_selection import train_test_split, KFold, \
    RepeatedStratifiedKFold, StratifiedKFold  # Importing train_test_split and KFold for data splitting
import tensorflow as tf  # Importing TensorFlow library for deep learning
from sklearn.dummy import DummyClassifier  # Importing DummyClassifier for creating a baseline classifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report  # Importing metrics for evaluation
from tensorflow.keras.applications import VGG16  # Importing VGG16 pre-trained model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization
import gc
from tensorflow.keras.callbacks import EarlyStopping

# List of categories
CAT = ['no_header', 'value_attr', 'col_0', 'row_0_col_0', 'row_01_col_0', 'row_012', 'row_01', 'row_0', 'others']


class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def display_class_distribution(features, labels):
    counts = pd.Series(labels).value_counts()
    # Sort the counts
    sorted_counts = counts.sort_values(ascending=False)
    max_len = max(len(item) for item in CAT)
    print("Class\t\tClass Digit\t\t\tcount")
    for index, count in sorted_counts.items():
        item = CAT[index]
        print("{:<{width}}\t\t{}\t\t\t{}".format(item, index, count, width=max_len))
    print("---------")
    print("Features and targets datasets")
    print("Features dataset: ", features.shape)
    print("Targets dataset: ", labels.shape)
    print("Number of features (columns): ", features.shape[1])
    print("Number of samples (rows): ", features.shape[0])
    print("-----------------------------\n")


# Function to load and preprocess images
def load_and_preprocess_images(directory):
    training_data = []
    for item in CAT:  # Looping through each category
        path = os.path.join(directory, item)  # Getting the path for each category
        class_num = CAT.index(item)  # Getting the index of the category
        for img in os.listdir(path):  # Looping through each image in the category
            img_array = cv2.imread(os.path.join(path, img))  # Reading the image using OpenCV
            training_data.append([img_array, class_num])  # Appending image array and class number to training data

    X = []  # List to store image arrays
    y = []  # List to store class numbers
    for feature, label in training_data:  # Looping through each image array and class number
        X.append(feature)  # Appending image array to X
        y.append(label)  # Appending class number to y

    X = np.array(X).reshape(-1, 75, 75, 3)  # Reshaping X to desired format
    X = X / 255.0  # Normalizing pixel values of X to the range [0, 1]

    return X, np.array(y)  # Returning preprocessed images and labels


def sampling(features, labels):
    features_flat = features.reshape(features.shape[0], -1)
    print("------------------")
    print("Class Distribution before Sampling:")
    display_class_distribution(features_flat, labels)

    # Random undersampling
    rus = RandomUnderSampler(sampling_strategy={0: 283, 1: 380, 2: 684, 3: 242, 4: 48, 5: 34, 6: 263, 7: 800, 8: 264})
    features, labels = rus.fit_resample(features_flat, labels)
    print("Class Distribution after UnderSampling:")
    display_class_distribution(features, labels)

    # SMOTE oversampling of the minority class
    smote = SMOTE(sampling_strategy={0: 800, 1: 800, 2: 800, 3: 800, 4: 800, 5: 800, 6: 800, 7: 800,
                                     8: 800})  # Specify ratios for each class
    print("Class Distribution after OverSampling:")
    features, labels = smote.fit_resample(features, labels)
    display_class_distribution(features, labels)


    # Reshape features back to original image dimensions
    features = features.reshape(-1, 75, 75, 3) # Reshape to (num_samples, 75, 75, 3)

    return features, labels


# Function to create and train a dummy classifier
def baseline_classifier(x_train, y_train, x_test, y_test):
    dummy_clf = DummyClassifier(strategy='stratified', random_state=42)  # Creating a baseline dummy classifier
    dummy_clf.fit(x_train, y_train)  # Fitting the dummy classifier on training data
    y_pred = dummy_clf.predict(x_test)  # Making predictions on test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculating accuracy
    f1Score = f1_score(y_test, y_pred, average='weighted')  # Calculating F1-score
    precision = precision_score(y_test, y_pred, average='weighted')  # Calculating precision
    recall = recall_score(y_test, y_pred, average='weighted')  # Calculating recall
    print("Baseline Classifier Accuracy:", accuracy)  # Printing accuracy
    print("Baseline Classifier f1_score:", f1Score)  # Printing F1-score
    print("Baseline Classifier Precision:", precision)  # Printing precision
    print("Baseline Classifier Recall:", recall)  # Printing recall




# Function to build the model architecture
def build_model():
    vgg16 = VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=(75, 75, 3),
        pooling=None,
    )

    model = Sequential()

    model.add(Rescaling(1. / 255, input_shape=(75, 75, 3)))

    for layers in vgg16.layers[1:-1]:
        model.add(layers)
    model.add(Dense(9, activation='softmax'))

    for layers in model.layers[1:-1]:
        layers.trainable = False

    model.summary()

    # config model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



# Function to train the model
def train_model(features, labels):

    # Initialize lists to store the evaluation metrics

    model = build_model()

    repeat_f1_scores = []
    repeat_accuracy_scores = []
    repeat_recall_scores = []
    repeat_precision_scores = []
    repeat_loss_scores = []

    repeat_training_losses = []
    repeat_training_accuracies = []
    repeat_training_f1Scores = []

    baseline_repeat_f1_scores = []
    baseline_repeat_accuracy_scores = []
    baseline_repeat_recall_scores = []
    baseline_repeat_precision_scores = []
    # callback


    early_stopping = [EarlyStopping(monitor='val_acc', patience=10)]

    rskf = StratifiedKFold(n_splits=10, random_state=np.random.randint(1, 1000), shuffle=True)

    for repeat_idx in range(5):
        fold_accuracy_scores = []
        fold_f1_scores = []
        fold_recall_scores = []
        fold_precision_scores = []
        fold_loss_scores = []

        folds_training_losses = []
        folds_training_accuracies = []
        folds_training_f1Scores = []

        baseline_fold_accuracy_scores = []
        baseline_fold_f1_scores = []
        baseline_fold_recall_scores = []
        baseline_fold_precision_scores = []
        for fold_idx, (train_index, test_index) in enumerate(rskf.split(features, labels), start=1):
            # Split the data
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Scale the data
            # X_train_scaled = scaler.fit_transform(X_train)
            # X_test_scaled = scaler.transform(X_test)

            # train
            # Train the model with early stopping callback
            history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                                callbacks=[GarbageCollectorCallback(), early_stopping])


            # Print training metrics
            train_loss = history.history['loss']
            train_accuracy = history.history['accuracy']
            train_precision = history.history['precision']
            train_recall = history.history['recall']
            train_f1_score = history.history['f1_scoree']

            print("For this Fold, for all epochs:")
            fold_average_train_loss = sum(train_loss) / len(train_loss)
            print("Average Train Loss:", fold_average_train_loss)
            fold_average_train_accuracy = sum(train_accuracy) / len(train_accuracy)
            print("Average Train Accuracy:", fold_average_train_accuracy)
            fold_average_train_precision = sum(train_precision) / len(train_precision)
            print("Average Train Precision:", fold_average_train_precision)
            fold_average_train_recall = sum(train_recall) / len(train_recall)
            print("Average Train Recall:", fold_average_train_recall)
            fold_average_train_f1Score = sum(train_f1_score) / len(train_f1_score)
            print("Average Train F1-Score:", fold_average_train_f1Score)
            print("---------------------------------------")

            # Predict the labels for the testing data
            y_pred_ohe = model.predict(X_test)  # Predict probabilities for each class.
            y_pred_labels = np.argmax(y_pred_ohe, axis=1)  # Convert probabilities to class labels.
            y_test_ohe = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels.

            print("Testing Results:\n")
            print("=========================")
            print('\nClassification Report\n')
            print(classification_report(y_test_ohe, y_pred_labels, target_names=CAT))
            # ---------------------#

            lossScore, accuracyScore = model.evaluate(X_test, batch_size=32)

            f1Score = f1_score(y_test_ohe, y_pred_labels, average='weighted')
            recallScore = recall_score(y_test_ohe, y_pred_labels, average='macro')
            precisionScore = precision_score(y_test_ohe, y_pred_labels, average='weighted')
            # Print the fold number and its score
            print(
                f"Repeat {repeat_idx + 1}, Fold {fold_idx}, Loss Score: {lossScore:.4f}, Accuracy Score: {accuracyScore:.4f}, f1 Score: {f1Score:.4f},  Recall Score: {recallScore:.4f},  Precision Score: {precisionScore:.4f}")

            # ---------------------#


            print("---------------------------")

            # foldsResults[i] = [fold_average_train_loss, fold_average_train_accuracy, fold_average_train_f1Score, loss, accuracy, f1_scoree]

            folds_training_losses.append(fold_average_train_loss)
            folds_training_accuracies.append(fold_average_train_accuracy)
            folds_training_f1Scores.append(fold_average_train_f1Score)


            fold_accuracy_scores.append(accuracyScore)
            fold_f1_scores.append(f1Score)
            fold_recall_scores.append(recallScore)
            fold_precision_scores.append(precisionScore)
            fold_loss_scores.append(lossScore)


            # ------------------------------------------------------------------#
            # Dummy classifier #
            baseline_accuracy, baseline_f1, baseline_recall, baseline_precision = baseline_classifier(X_train, y_train,
                                                                                                      X_test, y_test)
            baseline_fold_accuracy_scores.append(baseline_accuracy)
            baseline_fold_f1_scores.append(baseline_f1)
            baseline_fold_recall_scores.append(baseline_recall)
            baseline_fold_precision_scores.append(baseline_precision)

            print("------------------")
        print("---------------------------------------------------------------")
        # Calculate and print the mean score for the repeat
        # For training #
        mean_train_accuracy_score = np.mean(folds_training_accuracies)
        print(f"Mean training Accuracy score for Repeat {repeat_idx + 1}: {mean_train_accuracy_score:.4f}")
        repeat_training_accuracies.append(mean_train_accuracy_score)

        mean_train_f1_score = np.mean(folds_training_f1Scores)
        print(f"Mean training f1 score for Repeat {repeat_idx + 1}: {mean_train_f1_score:.4f}")
        repeat_training_f1Scores.append(mean_train_f1_score)

        mean_train_loss_score = np.mean(folds_training_losses)
        print(f"Mean training loss score for Repeat {repeat_idx + 1}: {mean_train_loss_score:.4f}")
        repeat_training_losses.append(mean_train_loss_score)

        print("--------------------")
        # for testing #

        mean_accuracy_score = np.mean(fold_accuracy_scores)
        print(f"Mean Accuracy score for Repeat {repeat_idx + 1}: {mean_accuracy_score:.4f}")
        repeat_accuracy_scores.append(mean_accuracy_score)

        mean_f1_score = np.mean(fold_f1_scores)
        print(f"Mean f1 score for Repeat {repeat_idx + 1}: {mean_f1_score:.4f}")
        repeat_f1_scores.append(mean_f1_score)

        mean_recall_score = np.mean(fold_recall_scores)
        print(f"Mean recall score for Repeat {repeat_idx + 1}: {mean_recall_score:.4f}")
        repeat_recall_scores.append(mean_recall_score)

        mean_precision_score = np.mean(fold_precision_scores)
        print(f"Mean precision score for Repeat {repeat_idx + 1}: {mean_precision_score:.4f}")
        repeat_precision_scores.append(mean_precision_score)

        mean_loss_score = np.mean(fold_loss_scores)
        print(f"Mean loss score for Repeat {repeat_idx + 1}: {mean_loss_score:.4f}")
        repeat_loss_scores.append(mean_precision_score)

        baseline_mean_accuracy = np.mean(baseline_fold_accuracy_scores)
        baseline_mean_f1 = np.mean(baseline_fold_f1_scores)
        baseline_mean_recall = np.mean(baseline_fold_recall_scores)
        baseline_mean_precision = np.mean(baseline_fold_precision_scores)

        baseline_repeat_accuracy_scores.append(baseline_mean_accuracy)
        baseline_repeat_f1_scores.append(baseline_mean_f1)
        baseline_repeat_recall_scores.append(baseline_mean_recall)
        baseline_repeat_precision_scores.append(baseline_mean_precision)

        print("=============================================")

    print("***************************************************")
    # Calculate the overall mean score across all repeats
    overall_mean_train_accuracy_score = np.mean(repeat_training_accuracies)
    print(f"Overall Mean Train Accuracy Score: {overall_mean_train_accuracy_score:.4f}")

    overall_mean_train_f1_score = np.mean(repeat_training_f1Scores)
    print(f"Overall Mean Train f1 Score: {overall_mean_train_f1_score:.4f}")

    overall_mean_train_recall_score = np.mean(repeat_training_losses)
    print(f"Overall Mean Train Loss Score: {overall_mean_train_recall_score:.4f}")
    print("--------------------")
    #--------#



    overall_mean_accuracy_score = np.mean(repeat_accuracy_scores)
    print(f"Overall Mean Accuracy Score: {overall_mean_accuracy_score:.4f}")

    overall_mean_f1_score = np.mean(repeat_f1_scores)
    print(f"Overall Mean f1 Score: {overall_mean_f1_score:.4f}")

    overall_mean_recall_score = np.mean(repeat_recall_scores)
    print(f"Overall Mean Recall Score: {overall_mean_recall_score:.4f}")

    overall_mean_precision_score = np.mean(repeat_precision_scores)
    print(f"Overall Mean Precision Score: {overall_mean_precision_score:.4f}")

    overall_mean_loss_score = np.mean(repeat_loss_scores)
    print(f"Overall Mean test Loss Score: {overall_mean_loss_score:.4f}")

    print("------------")
    print("Baseline:\n")
    baseline_overall_mean_accuracy_score = np.mean(baseline_repeat_accuracy_scores)
    print(f"Baseline Overall Mean Accuracy Score: {baseline_overall_mean_accuracy_score:.4f}")

    baseline_overall_mean_f1_score = np.mean(baseline_repeat_f1_scores)
    print(f"Baseline Overall f1 Score: {baseline_overall_mean_f1_score:.4f}")

    baseline_overall_mean_recall_score = np.mean(baseline_repeat_recall_scores)
    print(f"Baseline Overall Recall Score: {baseline_overall_mean_recall_score:.4f}")

    baseline_overall_mean_Precision_score = np.mean(baseline_repeat_precision_scores)
    print(f"Baseline Overall Precision Score: {baseline_overall_mean_Precision_score:.4f}")
    print("***************************************************")

def main():
    # Load and preprocess images
    # folder = "../../../preProcessImages/full_Images_Merged_Resized75"
    folder = '/content/drive/My Drive/ThesisData/DataForNewExperiments/full_Images_Merged_Resized75'
    features, labels = load_and_preprocess_images(folder)

    # Sampling Data
    sampled_features, sampled_labels = sampling(features, labels)

    del features
    del labels

    train_model(sampled_features, sampled_labels)



if __name__ == "__main__":
    main()
