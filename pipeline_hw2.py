'''
Homework 1 Diego Escobar
April 9th 2019
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Auxiliary Functions:
def open_csv_data(filename):
    '''
    Checks if the file exists and turns it into a pandas dataframe.
    Input: filename (string)
    Output: Pandas df (if file exist)
    '''
    try:
        df = pd.read_csv(filename)
        return df
    except IOError as e:
        print("File not found.")

# Correlation Table:
def correlation_plot(df, varlist):
    '''
    Generates a plot displaying the correlation matrix.
    Input: dataframe (pandas)
           varlist (list of string with var names)
    Output: figure
    '''
    corr = df[varlist].corr()
    figure = sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    return figure

# Cross Tabulation
def crosstab_plt(df, var1, var2, outcome_var, fig_size=(10,5)):
	'''
	Creates a graph of the cross-tabulation with frequencies for each bin.
    Input: df (pandas df)
           var1, var2 (string - var names)
           _norm (boolean - whether to normalize frequency or not).
           fig_size (tuple - figure dimension).
	'''
	crossing = pd.crosstab(df[var1], df[var2], values=df[outcome_var], \
                        aggfunc=[np.mean])
	figure = plt.figure(figsize=fig_size)
	sns.heatmap(crossing)
	return figure

# Fixing outliers:
def clean_outliers(dframe, variable, max_val, replace_val):
    '''
    Replaces values above certain threshold with the replacement value.
    input: dataframe (pandas), variable name (string), max_val (float)
           replace_val (any type that is supported by the pandas column).
    '''
    dframe[variable] = np.where(dframe[variable] <= max_val, dframe[variable], 
        replace_val)

# Clearn Nulls:
def missing_for_nulls(df, variable_list):
    '''
    Replaces null wiht 0 and creates dummy indicating that.
    Input: dframe (pandas df)
           variable to clean (list of strings)
           fill_with (string -> mean or median)
    Output: Changes are made in the dataset
    '''
    new_vars = []
    for variable in variable_list:
        new_vars.append("missing_" + variable)
        df["missing_" + variable] = np.where(df[variable].isnull(), 1, 0)
        df[variable] = np.where(df[variable].isnull(), 0, df[variable])
    return new_vars

# Replaces nulls with mean or median:
def missings_to_mean(df, variable_list):
    '''
    Replaces null values with mean
    Input: dframe (pandas df)
           variable to clean (list of strings)
    Output: Changes are made in the dataset
    '''
    for variable in variable_list:
        df[variable] = df[variable].fillna(df[variable].mean())

def gen_dummies(dframe, var_list):
    '''
    Create dummies from specific variables.
    Input: df (pandas dataframe)
           var_list (list of strings)
    Output: name of newly created variables (list of strings)
    '''
    new_vars = []
    for variable in var_list:
        for value in dframe[variable].unique():
            new_vars.append(variable + str(value))
            dframe[variable + str(value)] = dframe[variable] == value
    return new_vars

def gen_quantiles(df, var_list, n_bins=3, labels=False):
    '''
    Divides the data into n_bins-quantiles.
    Input: n_bins (int -> in how many quantiles to split the data)
           labels (list of string -> names for quantiles, optional)
    '''
    new_vars = []
    for variable in var_list:
        new_vars.append(variable + "_bins")
        print(variable)
        df[variable + "_bins"] = pd.cut(df[variable], n_bins, labels)
    return new_vars

def gen_bins(df, variable, bins, labels=False):
    '''
    Divides the data into bins; with limits specified in bins list.
    Input: variable name (string)
           bins (list of ints -> cut-offs for the bins)
           labels (list of string -> names for bins, optional)
    Output: name of the newly created variable
    '''
    df[variable + "_bins"] = pd.cut(df[variable], bins, labels)
    return variable + "_bins"

def data_format(dframe, outcome_var):
    '''
    Transform pandas df to np array and data type float32 required by sklearn
    '''
    x_data = np.array(dframe.drop([outcome_var], axis=1).values, 
                      dtype=np.float32)
    y_data = np.array(dframe.filter([outcome_var], axis=1).values, 
        dtype=np.float32).ravel()
    return y_data, x_data

def transform_data(x_data, y_data, test_size=0.33, random_state=1):
    '''
    Splits the data into test and train and then normalize it.
    Input: np arrays
    Output: standardized split np arrays
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, \
        test_size=0.33, random_state=1)
    # Some of the methods require normalization (LASSO, NN) so I will normalize
    # First with the training data and then apply the same scale as test data.
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_test = scaler.transform(x_test)
    x_train = scaler.transform(x_train)
    return x_train, x_test, y_train, y_test
