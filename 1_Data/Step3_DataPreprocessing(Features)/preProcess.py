# ================================================================================#
#       Tables classifications using attributes taken from rendered HTML pages   #
#       Sameh Algharabli                                                         #
# ================================================================================#


"""~~~ Libraries ~~~"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier, Perceptron, \
    LogisticRegression
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing, model_selection, svm

# ==================================================================================#

# Read the first Excel file into a DataFrame
df1 = pd.read_excel('oldData/oldData_Labeled.xlsx')

# Read the second Excel file into another DataFrame
df2 = pd.read_excel('New_Data_Labeled_withClasses_withoutNotAtable_withoutDisagreements/newData_Labeled.xlsx')

# Merge the two DataFrames based on a common column
df = pd.concat([df1, df2], ignore_index=True)


# -----------------
def render(x):
    if x == '[{"row":"0"}]' or x == '[{"column":"0"}]' or x == 'No Header' or x == '[{"row":"0"},{"row":"1"},{"column":"0"}]' or x == '[{"row":"0"},{"column":"0"}]' or x == '[{"row":"0"},{"row":"1"}]' or x == '[{"row":"0"},{"row":"1"},{"row":"2"}]' or x == 'Value/attribute':
        return x
    else:
        return 'other'
# ---------------

df['header_new'] = df['header'].apply(render)

# ---------------#
# Creating a copy of the DataFrame
df_test = df.copy()
print("At the beginning, we have: " + str(len(df.columns)) + " Features")
for col_name in df_test.columns:
    print(col_name)
print("--------------------------")

# ---------------------------------------------------------------------------------------------------
def calc_isna(df):
    print("Statistics about nan cells:\n")
    empty_counts = df.isna().sum().sum()
    print("There are " + str(empty_counts) + " NaN cells in the full dataset\n")

    for col_name in df.columns:
        empty_counts = df[col_name].isna().sum().sum()
        if (empty_counts > 0):
            print("In column {} there are {} nan cells".format(col_name, empty_counts))


print("Originally:")
calc_isna(df_test)
# df_test.fillna("none", inplace=True)
# print("After Replacing nan cells with none:\n")
# calc_isna(df_test)
print("----------------------------------------")

# ---------------------------------------------------------------------------------------------------------#
# Reading the features and preproces them
df_new = pd.DataFrame()
df_new['page_name'] = df_test['page_name']
df_new['table_no'] = df_test['table_no']
df_new['rows'] = df_test['rows'].astype('int64')  # ok
df_new['cols'] = df_test['cols'].astype('int64')  # ok
df_new['th'] = df_test['th'].astype('int64')  # ok
df_new['height'] = df_test['height'].astype('float')  # ok
df_new['width'] = df_test['width'].astype('float')  # ok
df_new['y'] = df_test['y'].astype('float')  # ok
df_new['x'] = df_test['x'].astype('float')  # ok
df_new['total_numbers'] = df_test['total_numbers'].astype('int64')  # ok
df_new['total_textContent'] = df_test['total_textContent'].astype('int64')  # ok
df_new['has_col_span'] = df_test['has_col_span'].astype('int64')  # ok
df_new['has_row_span'] = df_test['has_row_span'].astype('int64')  # ok
df_new['img_tag_in_table'] = df_test['img_tag_in_table'].astype('int64')  # ok
df_new['cols_in_first_row'] = df_test['cols_in_first_row'].astype('int64')  # ok
df_new['cols_in_last_row'] = df_test['cols_in_last_row'].astype('int64')  # ok .replace('no elements found', '0').astype('int64')
df_new['header'] = df_test['header_new']

# ===================================================================================================#
# Data Preprocessing#
# ===================================================================================================#

df_new['is_leaf_table'] = df_test['is_leaf_table'].apply(lambda x: 1 if x > 0 else 0).astype('int64')  # ok
df_new['has_form_elements'] = df_test['has_form_elements'].apply(lambda x: 1 if x > 0 else 0).astype('int64')  # ok

# It has two values: collapse and separate, I made collapse as 0 and separate as 1
df_test.table_border_collapse = pd.Categorical(df_test.table_border_collapse)
# Converting the values from 0 to 1, since I have 9 classes
df_new['table_border_collapse_numeric'] = df_test.table_border_collapse.cat.codes # ok

# -------------------------------------------
# Preprocessing for the following features:
# last_row_background
# first_row_background
# table_background
# Task: We have rgb(255,255,255) and rgba(0,0,0,0), I extract only the numbers in this form
# (255, 255, 255) and (0, 0, 0, 0)

# print("--------------\nBefore\n-------------")
# print(df_test['first_row_background'])

def convert_color_to_numeric(s):
    if (pd.isna(s)):
        return -1
    else:
        numbers = re.findall(r'\d+', s)

        numeric_value = int(numbers[0]) * 65536 + int(numbers[1]) * 256 + int(numbers[2])
        return numeric_value


df_new['last_row_background_numeric'] = df_test['last_row_background'].apply(convert_color_to_numeric)  # ok
df_new['first_row_background_numeric'] = df_test['first_row_background'].apply(convert_color_to_numeric)  # ok
df_new['table_background_numeric'] = df_test['table_background'].apply(convert_color_to_numeric)  # ok
# print("--------------\nafter\n-------------")
# print(df_new['first_row_background'])
# ======================================================================================================#

# Preprocessing for the following feature:
# table_border
# Task: Replace empty with none,


print("--------------\nBefore\n-------------")
print(df_test['table_border'])

# df_new['table_border'] = df_test['table_border'].fillna('none none none')
#
# (\S+): This part captures one or more non-whitespace characters.
# The parentheses ( ) create a capturing group, which means that the matched content
# will be extracted as a separate group.
# \s+: This part matches one or more whitespace characters.
# (\S+): This is similar to the first part and captures one or more non-whitespace characters.
# \s+: Again, matches one or more whitespace characters.
# (.*): This part captures zero or more of any character (.) until the end of the string (*). The parentheses create another capturing group.
#
# In summary, this regular expression is designed to match and capture three groups:
#
# The first group ((\S+)) captures the first non-whitespace sequence.
# The second group ((\S+)) captures the second non-whitespace sequence.
# The third group ((.*)) captures the rest of the string, including any whitespace
# and characters after the second non-whitespace sequence.

# df_new[['border_width', 'border_style', 'border_color']] = df_new['table_border'].str.extract(r'(\S+)\s+(\S+)\s+(.*)')

# df_new['border_style'] = df_test['border_style']  # ok

'''border_style
none          3                       4620
solid         5                       1904
outset        4                        425
hidden        2                        189
dotted        0                         19
groove        1                          1
Name: count, dtype: int64

border_style
none      64.5432%
solid     26.5996%
outset     5.9374%
hidden     2.6404%
dotted     0.2654%
groove      0.014%
Name: proportion, dtype: object
'''
df_new['border_style'] = df_test['border_style'].fillna("none")
df_new.border_style = pd.Categorical(df_new.border_style)
df_new['border_style_numeric'] = df_new.border_style.cat.codes # ok


print((df_new[['border_style', 'border_style_numeric']].value_counts()))
print("------------------------------------")
print("The percentage of each class: ")
print("------------------------------------")
print(df_new.border_style.value_counts(normalize=True).mul(100).round(4).astype(str) + '%')
print("------------------------------------")
# ======================================================================================================#
df_new['border_color_numeric'] = df_test['border_color'].apply(convert_color_to_numeric)

# Removing the px from the border width
def remove_px(s):
    if (pd.isna(s)):
        return 0
    elif s == '100%' or s == 'auto':
        return 100
    elif (s == 0):
        return float(0)
    elif type(s) != str:
        return float(s)
    else:
        return float(re.sub(r'[^\d.]', '', s))


# There is no 'auto' here
df_new['border_width'] = df_test['border_width'].apply(remove_px).astype(float)  # apply(lambda x: x if x == 'none' else float(re.sub(r'[^\d.]', '', x)))

#=================================================================================================#

def calc_noProperty(df):
    print("Statistics about no property found cells:\n")
    total_no_property_count = 0  # = df_test.value_counts().get('no property found', 0)

    for col_name in df.columns:
        # change df_test to df_neww
        no_property_count = df[col_name].value_counts().get('no property found', 0)
        if (no_property_count > 0):
            total_no_property_count += no_property_count
            print("In column {} there are {} (No property found) cells".format(col_name, no_property_count))

    print("There are {} cells in that has `No property found` in the full dataset\n".format(total_no_property_count))


def calc_auto(df):
    print("Statistics about auto cells:\n")
    total_auto_count = 0  # = df_test.value_counts().get('no property found', 0)

    for col_name in df.columns:
        # change df_test to df_neww
        auto_count = df[col_name].value_counts().get('auto', 0)
        if (auto_count > 0):
            total_auto_count += auto_count
            print("In column {} there are {} (auto) cells".format(col_name, auto_count))

    print("There are {} cells that have 'auto' in the full dataset\n".format(total_auto_count))



#
print("Originally:")
calc_noProperty(df_test)

# # last_col_width, last_col_height, first_col_height, first_col_width
print("----------------------------------------")
# After removing the no property found from the column that have it:
# Replacing "No property found with 0




df_new['first_col_width'] = df_test['first_col_width'].replace('no property found', 0)
df_new['first_col_height'] = df_test['first_col_height'].replace('no property found', 0)
df_new['last_col_height'] = df_test['last_col_height'].replace('no property found', 0)
df_new['last_col_width'] = df_test['last_col_width'].replace('no property found', 0)
#----------------------------#
df_new['first_col_width'] = df_new['first_col_width'].replace('', 0)
df_new['first_col_height'] = df_new['first_col_height'].replace('', 0)
df_new['last_col_height'] = df_new['last_col_height'].replace('', 0)
df_new['last_col_width'] = df_new['last_col_width'].replace('', 0)

df_new['first_row_width'] = df_test['first_row_width'].replace('', 0)
df_new['first_row_height'] = df_test['first_row_height'].replace('', 0)
df_new['last_row_width'] = df_test['last_row_width'].replace('', 0)
df_new['last_row_height'] = df_test['last_row_height'].replace('', 0)
#---------------------------#
print("after:")
calc_noProperty(df_new)
print("---------------------------")


'''Statistics about auto cells:

In column last_col_width there are 9 (auto) cells
In column last_col_height there are 14 (auto) cells
In column last_row_height there are 10 (auto) cells
In column last_row_width there are 12 (auto) cells
In column first_col_height there are 3 (auto) cells
In column first_row_height there are 1 (auto) cells
In column first_col_width there are 4 (auto) cells
In column first_row_width there are 1 (auto) cells
There are 54 cells that have 'auto' in the full dataset
'''
calc_auto(df_test)


# Removing the "px" characters form the 6 features below
df_new['first_col_width'] = df_new['first_col_width'].apply(remove_px).astype(float)
df_new['first_col_height'] = df_new['first_col_height'].apply(remove_px).astype(float)
df_new['last_col_height'] = df_new['last_col_height'].apply(remove_px).astype(float)
df_new['last_col_width'] = df_new['last_col_width'].apply(remove_px).astype(float)

df_new['first_row_width'] = df_new['first_row_width'].apply(remove_px).astype(float)
df_new['first_row_height'] = df_new['first_row_height'].apply(remove_px).astype(float)
df_new['last_row_width'] = df_new['last_row_width'].apply(remove_px).astype(float)
df_new['last_row_height'] = df_new['last_row_height'].apply(remove_px).astype(float)
calc_auto(df_new)


# print("first_col_width:\n",df_new['first_col_width'].to_string())
# print("first_col_height:\n",df_new['first_col_height'].to_string())
# print("last_col_height:\n",df_new['last_col_height'].to_string())
# print("last_col_width:\n",df_new['last_col_width'].to_string())
#
# print("first row width:\n",df_new['first_row_width'].to_string())
# print("first_row_height:\n",df_new['first_row_height'].to_string())
# print("last_row_width:\n",df_new['last_row_width'].to_string())
# print("first col last_row_height:\n",df_new['last_row_height'].to_string())
# ======================================================================================================#

print("---------------------------")
print("after choosing some features, we have: " + str(len(df_new.columns)) + " Features")
for col_name in df_new.columns:
    print(col_name)
print("---------------------------")


# End of preprocessing the new dataset
# ----------------

# -----------------
# Finding the categories of the headers (the classes) #
df_new.header = pd.Categorical(df_new.header)
classes = df_new["header"].unique()
print("classes: ", classes)

# Converting the classes to digits from 0 to 8, since I have 9 classes
df_new['headerDigit'] = df_new.header.cat.codes

# print(df_new[['header','headerDigit']].to_string())
# print(df_new['header'].unique())

print("------------------------------------")
print("The number of rows for all classes: ", df_new.shape[0])
print("------------------------------------")
print("The classes, their corresponding digits, and their counts: ")
print("------------------------------------")
print((df_new[['header', 'headerDigit']].value_counts()))
print("------------------------------------")
print("The percentage of each class: ")
print("------------------------------------")
print(df_new.header.value_counts(normalize=True).mul(100).round(4).astype(str) + '%')
print("------------------------------------")

# ---------------------------#
# Plotting the 9 classes
import matplotlib.pyplot as plt

x = np.arange(9)
money = list(df_new['header'].value_counts())
percentage = list(df_new.header.value_counts(normalize=True).mul(100).round(4).astype(str) + '%')

fig, ax = plt.subplots(figsize=(18, 5))
plt.barh(x, money)
plt.yticks(x, ('Row_0', 'Others', 'Col_0', 'No Header', 'Row_01', 'Row_0_Col_0', 'V/A', 'Row_012', 'Row_01_col_0'))
for i, v in enumerate(percentage):
    ax.text(money[i] + 3, i, str(v), color='grey', fontweight='bold')

plt.show()

# -------------
# # Spliting data to features and target values y
features = np.array(df_new.drop(['headerDigit', 'header', 'page_name', 'table_no'], axis=1))  # check this
labels = np.array(df_new['headerDigit'])
for col_name in df_new.drop(['headerDigit', 'header', 'page_name', 'table_no'], axis=1):
    print(col_name)
# # checking shapes of features and target data
# # -----------------------------------------#
print("Data After Preprocessing\n")

# We have 31 Features, and 7158 entries or samples
print("Features and targets datasets")
print("Features dataset: ", features.shape)
print("Targets dataset: ", labels.shape)
print("Number of features (columns): ", features.shape[1])
print("Number of samples (rows): ", features.shape[0])
print("-----------------------------\n")

print(df_new.drop(['header', 'page_name', 'table_no'], axis=1).dtypes)
df_new = df_new.drop(['border_style'], axis=1)
# df_new.to_excel('FullData_merged_preprocessed.xlsx', index=False)


# Testing
print("=============")
print("Testing")
print(df_new[df_new['page_name'] == 'link_111_table_1'].to_string())
print("-------------")
print(df_new[df_new['page_name'] == '0730_wikipedia'].to_string())
print("-------------")
print(df_new[df_new['page_name'] == '0001_sindar'].to_string())
print("-------------")

print("Nan cells after preprocessing:")
calc_isna(df_new)


######################################################
######################################################
######################################################
