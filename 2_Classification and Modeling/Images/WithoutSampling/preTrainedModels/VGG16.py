#----------------------------------------------------------#
# Full new dataset, Sampling with Layers , size 75                  #
#----------------------------------------------------------#


import numpy as np  # Importing numpy library for numerical operations
import os  # Importing os library for operating system functionalities
import cv2  # Importing OpenCV library for image processing
import pandas as pd  # Importing pandas library for data manipulation
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, KFold, \
    RepeatedStratifiedKFold  # Importing train_test_split and KFold for data splitting
import tensorflow as tf  # Importing TensorFlow library for deep learning
from sklearn.dummy import DummyClassifier  # Importing DummyClassifier for creating a baseline classifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report  # Importing metrics for evaluation
from tensorflow.keras.applications import VGG16  # Importing VGG16 pre-trained model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization
import gc


# List of categories
CAT = ['no_header', 'value_attr', 'col_0', 'row_0_col_0', 'row_01_col_0', 'row_012', 'row_01', 'row_0', 'others']


class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

class EarlyStoppingAtMinValLoss(tf.keras.callbacks.Callback):
    def __init__(self, patience=10):
        super(EarlyStoppingAtMinValLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epochs with no improvement after which training will be stopped.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best validation loss as infinity.
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if np.less(current_val_loss, self.best_val_loss):
            self.best_val_loss = current_val_loss
            self.wait = 0
            # Record the best weights if current results is better (lower loss).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("\nEarly stopping: val_loss did not improve for {} epochs.".format(self.patience))
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

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



# Function to create and train a dummy classifier
def create_and_train_dummy_classifier(x_train, y_train, x_test, y_test):
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

# Custom precision, recall, and F1-score functions
def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))  # Calculating true positives
    predicted_positives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))  # Calculating predicted positives
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())  # Calculating precision
    return precision  # Returning precision


def recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))  # Calculating true positives
    possible_positives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))  # Calculating possible positives
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())  # Calculating recall
    return recall  # Returning recall


def f1_scoree(y_true, y_pred):
    p = precision(y_true, y_pred)  # Calculating precision
    r = recall(y_true, y_pred)  # Calculating recall
    return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))  # Calculating F1-score


# Function to build the model architecture
def build_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(9, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='Adam',
                  metrics=['accuracy', precision, recall, f1_scoree])
    return model


# Function to train the model
def train_model(model, x_train, y_train):

    # Initialize lists to store the evaluation metrics
    validation_accuracies = []
    validation_precisions = []
    validation_f1Scores = []
    validation_recalls = []
    validation_losses = []
    training_accuracies = []
    training_precisions = []
    training_f1Scores = []
    training_recalls = []
    training_losses = []

    # Initialize KFold object
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        print(f"Fold {fold}")
        x_train_fold, x_validation_fold = x_train[train_index], x_train[test_index]
        y_train_fold, y_validation_fold = y_train[train_index], y_train[test_index]

        early_stopping = EarlyStoppingAtMinValLoss(patience=8)  # Adjust patience as needed

        # Train the model with early stopping callback
        history = model.fit(x_train_fold, y_train_fold, epochs=20, batch_size=32,
                            validation_data=(x_validation_fold, y_validation_fold),
                            callbacks=[GarbageCollectorCallback(), early_stopping])

        # --------------------------------------------------------#
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

        training_losses.append(fold_average_train_loss)
        training_accuracies.append(fold_average_train_accuracy)
        training_precisions.append(fold_average_train_precision)
        training_f1Scores.append(fold_average_train_f1Score)
        training_recalls.append(fold_average_train_recall)
        # --------------------------------------------------------#

        # 3. Evaluate the Model
        loss, accuracy, precision, recall, f1_scoree = model.evaluate(x_validation_fold, y_validation_fold)
        print(f'validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')
        print(f'Validation Precision: {precision}')
        print(f'Validation Recall: {recall}')
        print(f'Validation F1 Score: {f1_scoree}')
        print("---------------------------")

        validation_losses.append(loss)
        validation_accuracies.append(accuracy)
        validation_precisions.append(precision)
        validation_f1Scores.append(f1_scoree)
        validation_recalls.append(recall)

    print("========================================================")

    # Calculate and print the average accuracy across all folds
    print("Folds are Done\n")
    print("Training Overall Results\n")
    average_training_loss = sum(training_losses) / len(validation_losses)
    print("Average Training Loss for all folds:", average_training_loss)
    average_training_accuracy = sum(training_accuracies) / len(training_accuracies)
    print("Average Training Accuracy for all folds:", average_training_accuracy)
    average_training_precision = sum(training_precisions) / len(training_precisions)
    print("Average Training Precision for all folds:", average_training_precision)
    average_training_recall = sum(training_recalls) / len(training_recalls)
    print("Average Training Recall for all folds:", average_training_recall)
    average_training_f1Score = sum(training_f1Scores) / len(training_f1Scores)
    print("Average Training F1-Score for all folds:", average_training_f1Score)
    print("---------------------")
    print("Validation Overall Results\n")
    average_validation_loss = sum(validation_losses) / len(validation_losses)
    print("Average Validation Loss:", average_validation_loss)
    average_validation_accuracy = sum(validation_accuracies) / len(validation_accuracies)
    print("Average Validation Accuracy:", average_validation_accuracy)
    average_validation_precision = sum(validation_precisions) / len(validation_precisions)
    print("Average Validation Precision:", average_validation_precision)
    average_validation_recall = sum(validation_recalls) / len(validation_recalls)
    print("Average Validation Recall:", average_validation_recall)
    average_validation_f1Score = sum(validation_f1Scores) / len(validation_f1Scores)
    print("Average Validation F1-Score:", average_validation_f1Score)

def train_model_on_Full_training(model, features, labels):
    n_repeats = 5
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=42)


    training_losses = []
    training_accuracies = []
    training_f1Scores = []

    testing_losses = []
    testing_accuracies = []
    testing_f1Scores = []

    # Iterate through the splits
    i = 0
    for train_index, test_index in rskf.split(features, labels):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=9)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=9)

        print("Fold: ",i)
        # Fitting the model with the full train data
        history = model.fit(x_train, y_train, epochs=20, batch_size=32,
                  callbacks=[GarbageCollectorCallback()])

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
        y_pred_ohe = model.predict(x_test)  # Predict probabilities for each class.
        y_pred_labels = np.argmax(y_pred_ohe, axis=1)  # Convert probabilities to class labels.
        y_test_ohe = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels.

        # Calculate and print the confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_true=y_test_ohe, y_pred=y_pred_labels)
        print(confusion_matrix)

        # ---------------------#
        print("Testing Results:\n")
        print("=========================")
        print('\nClassification Report\n')
        print(classification_report(y_test_ohe, y_pred_labels, target_names=CAT))
        # ---------------------#

        print("---------------------------")
        loss, accuracy, precision, recall, f1_scoree = model.evaluate(x_test, y_test)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')
        print(f'Test Precision: {precision}')
        print(f'Test Recall: {recall}')
        print(f'Test F1 Score: {f1_scoree}')

        # foldsResults[i] = [fold_average_train_loss, fold_average_train_accuracy, fold_average_train_f1Score, loss, accuracy, f1_scoree]

        training_losses.append(fold_average_train_loss)
        training_accuracies.append(fold_average_train_accuracy)
        training_f1Scores.append(fold_average_train_f1Score)

        testing_losses.append(loss)
        testing_accuracies.append(accuracy)
        testing_f1Scores.append(f1_scoree)
        i += 1
        print("\n=======================================================")


    average_train_losses = sum(training_losses)/len(training_losses)
    average_train_accuracy = sum(training_accuracies)/len(training_accuracies)
    average_train_f1Score = sum(training_f1Scores)/len(training_f1Scores)

    average_test_losses = sum(testing_losses)/len(testing_losses)
    average_test_accuracy = sum(testing_accuracies)/len(testing_accuracies)
    average_test_f1Score = sum(testing_f1Scores)/len(testing_f1Scores)

    print("Average train losses: ", average_train_losses)
    print("Average train accuracy: ", average_train_accuracy)
    print("Average train f1score: ", average_train_f1Score)
    print("-------------------")
    print("Average test losses: ", average_test_losses)
    print("Average test accuracy: ", average_test_accuracy)
    print("Average test f1score: ", average_test_f1Score)
    print("===========================")



def main():
    # Load and preprocess images
    # folder = "../../../preProcessImages/full_Images_Merged_Resized75"
    folder = '/content/drive/My Drive/ThesisData/DataForNewExperiments/full_Images_Merged_Resized75'
    features, labels = load_and_preprocess_images(folder)

    # Sampling Data
    #sampled_features, sampled_labels = sampling(features, labels)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42,
                                                        stratify=labels)



    # Create and train a dummy classifier
    create_and_train_dummy_classifier(x_train, y_train, x_test, y_test)

    del x_train
    del x_test
    del y_train
    del y_test

    # Build the model
    model = build_model()

    # Train the model
    # train_model(model, x_train, y_train)

    # Training the model on the full dataset
    # train_model_on_Full_training(model, x_train, y_train, x_test, y_test)

    train_model_on_Full_training(model, features, labels)



if __name__ == "__main__":
    main()
