# ML-for-PubPol
Assignment Machine Learning for Public Policy

Diego Escobar
May 30th 2019

This code loops over several machine learning classifiers implemented using sklearn package. The model also features data pre-processing semi-automatic (i.e. after choosing certain parameters and classifying a few specific variables types, such as outcomes). It also test several hyperparameters for each model and selects the best one based on the preferred metric.

There are three grid sizes:

test: to test if things are working
small: if you've got less than an hour
large: if you've got time or cores

The package is modular so more models or hyperparameters can be easily added. 

There are three files in the folder:

1) pipeline.py - This file includes several pre-processing functions to clean the data, split into test-test, create quantiles, create dummy variables, etc.

2) pipeline_ml.py - This includes the code for fitting the models and going through different sets of hyperparameters. It also relies on pipeline.py to do some data pre-processing if called directly, although the functions can also be used one by one.

Models implemented: all the models from sklearn are supported.
Metrics implemented: precision at k%, recall at k%, f1 at k%, area under ROC curve

3) hw5_diego_escobar.py - Example of the usage of both codes to an application to predict what projects will have problems getting funded within 60 days.

Models currently implemented: ['RF', 'DT', 'LR', 'KNN', 'AB', 'GB', 'SVM', 'ET', 'SGD']
