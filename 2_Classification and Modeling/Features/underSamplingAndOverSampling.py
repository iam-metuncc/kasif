# -----------------------------------------#
# Full new dataset, UnderSampling and oversampling, Only RF hyperparameter Tuning#
# -----------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
 
warnings.filterwarnings('ignore')
# List of categories
CAT = ['no_header', 'value_attr', 'col_0', 'row_0_col_0', 'row_01_col_0', 'row_012', 'row_01', 'row_0', 'others']


def load_data(file_path):
    """
    Load data from an Excel file.

    Args:
    file_path (str): Path to the Excel file.

    Returns:
    DataFrame: Loaded DataFrame containing the data.
    """
    df_processed = pd.read_excel(file_path)
    # df_processed = df_processed.drop(['border_style'], axis=1)
    return df_processed


def visualize_class_distribution(df):
    """
    Visualize the class distribution.

    Args:
    df (DataFrame): DataFrame containing the data.
    """
    x = np.arange(9)
    money = list(df['header'].value_counts())
    percentages = list(df.header.value_counts(normalize=True).mul(100).round(4).astype(str) + '%')

    fig, ax = plt.subplots(figsize=(18, 5))
    plt.barh(x, money)
    plt.yticks(x, (
    'Row_0', 'Others', 'Col_0', 'No Header', 'Row_01', 'Row_0_Col_0', 'Value/Attribute', 'Row_012', 'Row_01_col_0'))
    for i, v in enumerate(percentages):
        ax.text(money[i] + 3, i, str(v), color='grey', fontweight='bold')

    plt.show()


def preprocess_data(df):
    """
    Preprocess the data by separating features and labels.

    Args:
    df (DataFrame): DataFrame containing the data.

    Returns:
    tuple: Tuple containing features array and labels array.
    """
    features = np.array(df.drop(['headerDigit', 'header', 'page_name', 'table_no'], axis=1))
    labels = np.array(df['headerDigit'])
    return features, labels


def display_class_dis(features, labels):
    CAT = ['no_header', 'value_attr', 'col_0', 'row_0_col_0', 'row_01_col_0', 'row_012', 'row_01', 'row_0', 'others']
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


def baseline_classifier(X_train, y_train, X_test, y_test):
    """
    Create and evaluate a baseline dummy classifier.

    Args:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.
    X_test (array-like): Testing features.
    y_test (array-like): Testing labels.
    """
    dummy_clf = DummyClassifier(strategy='stratified', random_state=42)
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred, average='weighted')
    recallScore = recall_score(y_test, y_pred, average='weighted')
    precisionScore = precision_score(y_test, y_pred, average='weighted')
    return accuracy, f1Score, recallScore, precisionScore

def sampling(features, labels):
    print("------------------")
    print("Class Distribution before Sampling:")
    display_class_dis(features, labels)


    # Random undersampling
    rus = RandomUnderSampler(sampling_strategy={0: 283, 1: 382, 2: 684, 3: 242, 4: 48, 5: 34, 6: 263, 7: 800, 8: 633})
    features, labels = rus.fit_resample(features, labels)
    print("Class Distribution after UnderSampling:")
    features, labels = rus.fit_resample(features, labels)
    display_class_dis(features, labels)

    #SMOTE oversampling of the minority class
    smote = SMOTE(sampling_strategy={0: 800, 1: 800, 2: 800, 3: 800, 4: 800, 5: 800, 6: 800, 7: 800,
                                     8: 800})  # Specify ratios for each class
    print("Class Distribution after OverSampling:")
    features, labels = smote.fit_resample(features, labels)
    display_class_dis(features, labels)

    return features, labels

def grid_search_and_evaluate(features, labels):
    """
    Perform grid search and evaluation for each model.

    Args:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.
    X_test (array-like): Testing features.
    y_test (array-like): Testing labels.
    """
   
    estimators_list = [50, 100]
    depth_list = [5, 10]
    criterion_list = ["gini","entropy"]

    repeat_f1_scores = []
    repeat_accuracy_scores = []
    repeat_recall_scores = []
    repeat_precision_scores = []

    baseline_repeat_f1_scores = []
    baseline_repeat_accuracy_scores = []
    baseline_repeat_recall_scores = []
    baseline_repeat_precision_scores = []

    # Initialize StandardScaler
    scaler = StandardScaler()
    #oVr = OneVsRestClassifier(model)

    rskf = StratifiedKFold(n_splits=10, random_state=np.random.randint(1, 1000), shuffle=True)
    # Perform the repeated stratified k-fold cross-validation
    i = 0

    configurations_list = []
    for estimator in estimators_list:
        for depth in depth_list:
            for criterion in criterion_list:
                config_dic = {"n_estimators":estimator, "max_depth": depth, "criterion": criterion, "accuracy": 0.0, "f1": 0.0, "recall": 0.0, "precision": 0.0}
                print("Configuration: ", i)
                print("n_estimators: {}, max_depth: {}, criterion: {}".format(estimator, depth, criterion))
                print("---------------")
                for repeat_idx in range(5):
                    fold_accuracy_scores = []
                    fold_f1_scores = []
                    fold_recall_scores = []
                    fold_precision_scores = []

                    baseline_fold_accuracy_scores = []
                    baseline_fold_f1_scores = []
                    baseline_fold_recall_scores = []
                    baseline_fold_precision_scores = []
                    for fold_idx, (train_index, test_index) in enumerate(rskf.split(features, labels), start=1):
                        # Split the data
                        X_train, X_test = features[train_index], features[test_index]
                        y_train, y_test = labels[train_index], labels[test_index]

                        # Scale the data
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        model = RandomForestClassifier(n_estimators=estimator, criterion=criterion, max_depth=depth)
                        # Fit the model
                        ovo_classifier = OneVsOneClassifier(model)
                        ovo_classifier.fit(X_train_scaled, y_train)

                        # Predict and calculate accuracy
                        y_pred = ovo_classifier.predict(X_test_scaled)
                        accuracyScore = accuracy_score(y_test, y_pred)
                        f1Score = f1_score(y_test, y_pred, average='weighted')
                        recallScore = recall_score(y_test, y_pred, average='weighted')
                        precisionScore = precision_score(y_test, y_pred, average='weighted')

                        # Print the fold number and its score
                        print(
                            f"Repeat {repeat_idx + 1}, Fold {fold_idx}, Accuracy Score: {accuracyScore:.4f}, f1 Score: {f1Score:.4f},  Recall Score: {recallScore:.4f},  Precision Score: {precisionScore:.4f}")

                        fold_accuracy_scores.append(accuracyScore)
                        fold_f1_scores.append(f1Score)
                        fold_recall_scores.append(recallScore)
                        fold_precision_scores.append(precisionScore)

                        print('\nClassification Report\n')
                        print(classification_report(y_test, y_pred, target_names=CAT))

                        # ------------------------------------------------------------------#
                        # Dummy classifier #
                        baseline_accuracy, baseline_f1, baseline_recall, baseline_precision = baseline_classifier(X_train_scaled, y_train, X_test_scaled, y_test)
                        baseline_fold_accuracy_scores.append(baseline_accuracy)
                        baseline_fold_f1_scores.append(baseline_f1)
                        baseline_fold_recall_scores.append(baseline_recall)
                        baseline_fold_precision_scores.append(baseline_precision)

                        print("------------------")
                    print("---------------------------------------------------------------")
                    # Calculate and print the mean score for the repeat
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
                print("Result for Configuration: ", i)
                print("n_estimators: {}, max_depth: {}, criterion: {}".format(estimator, depth, criterion))
                print("------------------------------------------------")
                # Calculate the overall mean score across all repeats
                overall_mean_accuracy_score = np.mean(repeat_accuracy_scores)
                print(f"Overall Mean Accuracy Score: {overall_mean_accuracy_score:.4f}")

                overall_mean_f1_score = np.mean(repeat_f1_scores)
                print(f"Overall Mean f1 Score: {overall_mean_f1_score:.4f}")

                overall_mean_recall_score = np.mean(repeat_recall_scores)
                print(f"Overall Mean Recall Score: {overall_mean_recall_score:.4f}")

                overall_mean_precision_score = np.mean(repeat_precision_scores)
                print(f"Overall Mean Precision Score: {overall_mean_precision_score:.4f}")

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
                config_dic["accuracy"] = overall_mean_accuracy_score
                config_dic["f1"] = overall_mean_f1_score
                config_dic["recall"] = overall_mean_recall_score
                config_dic["precision"] = overall_mean_precision_score
                configurations_list.append(config_dic)
                i += 1

    #===============================================================================#


    # Sort by accuracy score (descending order)
    sorted_by_accuracy = sorted(configurations_list, key=lambda x: x['accuracy'], reverse=True)
    # Print the configuration with the highest accuracy
    highest_accuracy_config = sorted_by_accuracy[0]
    print(f"Configuration with highest accuracy: {highest_accuracy_config}")

    # Sort by recall score (descending order)
    sorted_by_recall = sorted(configurations_list, key=lambda x: x['recall'], reverse=True)
    # Print the configuration with the highest recall
    highest_recall_config = sorted_by_recall[0]
    print(f"Configuration with highest recall: {highest_recall_config}")

    # Sort by precision score (descending order)
    sorted_by_precision = sorted(configurations_list, key=lambda x: x['precision'], reverse=True)
    # Print the configuration with the highest precision
    highest_precision_config = sorted_by_precision[0]
    print(f"Configuration with highest precision: {highest_precision_config}")


    # Sort by f1 score (descending order)
    sorted_by_f1 = sorted(configurations_list, key=lambda x: x['f1'], reverse=True)
    # Print the configuration with the highest f1
    highest_f1_config = sorted_by_f1[0]
    print(f"Configuration with highest f1: {highest_f1_config}")
    print("----------------------")
    z = 1
    for config in sorted_by_f1:
        print("{}: {}".format(z, config))
        print("\n")
        z += 1
    print("====================================================================================")

def main():
    # Load data
    file_path = '/content/drive/My Drive/ThesisData/DataForNewExperiments/FullData_merged_preprocessed.xlsx'
    # file_path = '../../../FullData_merged_preprocessed.xlsx'
    df_processed = load_data(file_path)

    # Visualize class distribution
    visualize_class_distribution(df_processed)

    # Preprocess data
    features, labels = preprocess_data(df_processed)

    display_class_dis(features, labels)
    # Sampling Data
    sampled_features, sampled_labels = sampling(features, labels)

    # # Create and evaluate a baseline dummy classifier
    # baseline_classifier(X_train, y_train, X_test, y_test)

    # Perform grid search and evaluation for each model
    grid_search_and_evaluate(sampled_features, sampled_labels)


if __name__ == "__main__":
    main()
