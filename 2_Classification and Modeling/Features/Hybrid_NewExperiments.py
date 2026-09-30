#----------------------------------------------------------#
# Hybrid ,Featureas, new experiments#
#----------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# List of categories
CAT = ['no_header', 'value_attr', 'col_0', 'row_0_col_0', 'row_01_col_0','row_012', 'row_01', 'row_0', 'others']


def load_data(file_path):
    return pd.read_excel(file_path)


def visualize_class_distribution(df):
    x = np.arange(9)
    counts = list(df['header'].value_counts())
    percentages = list(df.header.value_counts(normalize=True).mul(100).round(4).astype(str) + '%')

    fig, ax = plt.subplots(figsize=(18, 5))
    plt.barh(x, counts)
    plt.yticks(x, ('Row_0', 'Others', 'Col_0', 'No Header', 'Row_01','Row_0_Col_0', 'Value/Attribute', 'Row_012', 'Row_01_col_0'))
    for i, v in enumerate(percentages):
        ax.text(counts[i] + 3, i, str(v), color='grey', fontweight='bold')
    plt.show()

#separating features and labels
def preprocess_data(df):
    features = np.array(df.drop(['headerDigit', 'header', 'page_name', 'table_no'], axis=1))
    labels = np.array(df['headerDigit'])
    return features, labels


def display_class_dis(features, labels):
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    max_len = max(len(item) for item in CAT)

    print("Class\t\tClass Digit\t\t\tcount")
    for index, count in counts.items():
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
   
    dummy_clf = DummyClassifier(strategy='stratified', random_state=42)
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    return (accuracy, f1_macro, recall_macro, precision_macro)
    

def sample_training_data(X_train, y_train):
    
    
    display_class_dis(X_train, y_train)

    rus = RandomUnderSampler(sampling_strategy={7: 800})
    X_train, y_train = rus.fit_resample(X_train, y_train)

    
    display_class_dis(X_train, y_train)

    
    smote = SMOTE(sampling_strategy={0: 800, 1: 800, 2: 800, 3: 800, 4: 800, 5: 800, 6: 800, 8: 800})
    X_train, y_train = smote.fit_resample(X_train, y_train)

    
    display_class_dis(X_train, y_train)

    return X_train, y_train

# Grid search function                   
def grid_search_and_evaluate(features, labels):
    estimators_list = [100]
    depth_list = [25]
    criterion_list = ["entropy"]
 
    i = 0
    configurations_list = []
 
    for estimator in estimators_list:
        for depth in depth_list:
            for criterion in criterion_list:
                config_dic = {"n_estimators": estimator, "max_depth": depth,"criterion": criterion,"accuracy": 0.0, "f1_macro": 0.0, "recall_macro": 0.0,"precision_macro": 0.0}
                print("Configuration: ", i)
                print("n_estimators: {}, max_depth: {}, criterion: {}".format(estimator, depth, criterion))
                print("---------------")
 
                repeat_f1_macro_scores = []
                repeat_accuracy_scores = []
                repeat_recall_macro_scores = []
                repeat_precision_macro_scores = []
 
                baseline_repeat_f1_macro_scores = []
                baseline_repeat_accuracy_scores = []
                baseline_repeat_recall_macro_scores = []
                baseline_repeat_precision_macro_scores = []
 
                for repeat_idx in range(5):
                    fold_accuracy_scores = []
                    fold_f1_macro_scores = []
                    fold_recall_macro_scores = []
                    fold_precision_macro_scores = []
 
                    baseline_fold_accuracy_scores = []
                    baseline_fold_f1_macro_scores = []
                    baseline_fold_recall_macro_scores = []
                    baseline_fold_precision_macro_scores = []
 
                    rskf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)

                    for fold_idx, (train_index, test_index) in enumerate(rskf.split(features, labels), start=1):
                        X_train, X_test = features[train_index], features[test_index]
                        y_train, y_test = labels[train_index], labels[test_index]
 
                        # Fit the scaler on this fold's training data only
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
 
                        # sampling happens here, inside the fold, on training data only. The test set stays untouched and imbalanced.

                        X_train_resampled, y_train_resampled = sample_training_data(X_train_scaled, y_train)
 
                        model = RandomForestClassifier(n_estimators=estimator, criterion=criterion, max_depth=depth,random_state=None)

                        
                        ovo_classifier = OneVsOneClassifier(model)
                        ovo_classifier.fit(X_train_resampled, y_train_resampled)
 
                        # Predictions
                        y_pred = ovo_classifier.predict(X_test_scaled)
 
                        
                        accuracyScore = accuracy_score(y_test, y_pred)
                        f1_macro = f1_score(y_test, y_pred, average='macro')
                        recall_macro = recall_score(y_test, y_pred, average='macro')
                        precision_macro = precision_score(y_test, y_pred, average='macro')
 
                        print(
                            f"Repeat {repeat_idx + 1}, Fold {fold_idx}, "
                            f"Accuracy: {accuracyScore:.4f}, "
                            f"F1 macro: {f1_macro:.4f}, "
                            f"Recall macro: {recall_macro:.4f}, "
                            f"Precision macro: {precision_macro:.4f}"
                        )
 
                        fold_accuracy_scores.append(accuracyScore)
                        fold_f1_macro_scores.append(f1_macro)
                        fold_recall_macro_scores.append(recall_macro)
                        fold_precision_macro_scores.append(precision_macro)
 
                        print('\nClassification Report\n')
                        print(classification_report(y_test, y_pred, target_names=CAT))
 
                        (baseline_accuracy, baseline_f1_m, baseline_recall_m, baseline_precision_m) = baseline_classifier(X_train_scaled, y_train, X_test_scaled, y_test)
                        baseline_fold_accuracy_scores.append(baseline_accuracy)
                        baseline_fold_f1_macro_scores.append(baseline_f1_m)
                        baseline_fold_recall_macro_scores.append(baseline_recall_m)
                        baseline_fold_precision_macro_scores.append(baseline_precision_m)
 
                        print("------------------")
                    print("---------------------------------------------------------------")
 
                    # Averages across the 10 folds of this repeat.
                    mean_accuracy_score = np.mean(fold_accuracy_scores)
                    print(f"Mean Accuracy for Repeat {repeat_idx + 1}: {mean_accuracy_score:.4f}")
                    repeat_accuracy_scores.append(mean_accuracy_score)
 
                    mean_f1_macro = np.mean(fold_f1_macro_scores)
                    print(f"Mean F1 (macro) for Repeat {repeat_idx + 1}: {mean_f1_macro:.4f}")
                    repeat_f1_macro_scores.append(mean_f1_macro)

                    mean_recall_macro = np.mean(fold_recall_macro_scores)
                    print(f"Mean Recall (macro) for Repeat {repeat_idx + 1}: {mean_recall_macro:.4f}")
                    repeat_recall_macro_scores.append(mean_recall_macro)
 
                    mean_precision_macro = np.mean(fold_precision_macro_scores)
                    print(f"Mean Precision (macro) for Repeat {repeat_idx + 1}: {mean_precision_macro:.4f}")
                    repeat_precision_macro_scores.append(mean_precision_macro)
 
                    baseline_repeat_accuracy_scores.append(np.mean(baseline_fold_accuracy_scores))
                    baseline_repeat_f1_macro_scores.append(np.mean(baseline_fold_f1_macro_scores))
                    baseline_repeat_recall_macro_scores.append(np.mean(baseline_fold_recall_macro_scores))
                    baseline_repeat_precision_macro_scores.append(np.mean(baseline_fold_precision_macro_scores))
                    
                    print("=============================================")
 
                # Averages across the 5 repeats for this configuration.
                print("***************************************************")
                print("Result for Configuration: ", i)
                print("n_estimators: {}, max_depth: {}, criterion: {}".format(estimator, depth, criterion))
                print("------------------------------------------------")
 
                overall_mean_accuracy_score = np.mean(repeat_accuracy_scores)
                print(f"Overall Mean Accuracy: {overall_mean_accuracy_score:.4f}")
 
                overall_mean_f1_macro = np.mean(repeat_f1_macro_scores)
                print(f"Overall Mean F1 (macro):    {overall_mean_f1_macro:.4f}")
 
                overall_mean_recall_macro = np.mean(repeat_recall_macro_scores)
                print(f"Overall Mean Recall (macro):    {overall_mean_recall_macro:.4f}")
 
                overall_mean_precision_macro = np.mean(repeat_precision_macro_scores)
                print(f"Overall Mean Precision (macro):    {overall_mean_precision_macro:.4f}")
 
                print("------------")
                print("Baseline (Dummy Classifier):")
                print(f"Baseline Overall Mean Accuracy:            {np.mean(baseline_repeat_accuracy_scores):.4f}")
                print(f"Baseline Overall Mean F1 (macro):          {np.mean(baseline_repeat_f1_macro_scores):.4f}")
                print(f"Baseline Overall Mean Recall (macro):      {np.mean(baseline_repeat_recall_macro_scores):.4f}")
                print(f"Baseline Overall Mean Precision (macro):   {np.mean(baseline_repeat_precision_macro_scores):.4f}")
                print("***************************************************")
 
                config_dic["accuracy"] = overall_mean_accuracy_score
                config_dic["f1_macro"] = overall_mean_f1_macro
                config_dic["recall_macro"] = overall_mean_recall_macro
                config_dic["precision_macro"] = overall_mean_precision_macro
                configurations_list.append(config_dic)
                i += 1
 
    # ------------------------------------------------------------------- #
    # Rankings                                                            #
    # ------------------------------------------------------------------- #
    
    sorted_by_accuracy = sorted(configurations_list, key=lambda x: x['accuracy'], reverse=True)
    print(f"Configuration with highest accuracy: {sorted_by_accuracy[0]}")
 
    sorted_by_recall_macro = sorted(configurations_list, key=lambda x: x['recall_macro'], reverse=True)
    print(f"Configuration with highest recall (macro): {sorted_by_recall_macro[0]}")
 
    sorted_by_precision_macro = sorted(configurations_list, key=lambda x: x['precision_macro'], reverse=True)
    print(f"Configuration with highest precision (macro): {sorted_by_precision_macro[0]}")
 
    sorted_by_f1_macro = sorted(configurations_list, key=lambda x: x['f1_macro'], reverse=True)
    print(f"Configuration with highest F1 (macro): {sorted_by_f1_macro[0]}")
 
    print("----------------------")
    print("All configurations sorted by F1 (macro), best first:")
    z = 1
    for config in sorted_by_f1_macro:
        print("{}: {}".format(z, config))
        print("\n")
        z += 1
    print("====================================================================================")


def main():
    file_path = 'FullData_merged_preprocessed.xlsx'

    print(f"Experiment: Hybrid_OnlyRF_New")

    df_processed = load_data(file_path)

    visualize_class_distribution(df_processed)

    features, labels = preprocess_data(df_processed)
    display_class_dis(features, labels)

    grid_search_and_evaluate(features, labels)


if __name__ == "__main__":
    main()
