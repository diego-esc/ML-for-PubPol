'''
Homework 1 Diego Escobar
April 9th 2019
'''

import pipeline_hw2 as pl

# Problem 1: Read Data
filename = "credit-data.csv"
df = pl.open_csv_data(filename)
outcome_var = "SeriousDlqin2yrs"

# Problem 2: Summary Stats
# For the first cases I consider a redundancy to define functions and I know
# I would not use them in my pipeline so I prefer to use them directly.
varlist = [outcome_var, 'DebtRatio', 'NumberOfDependents']
df[varlist].describe()
df['age'].hist(bins=20)
df['NumberOfDependents'].value_counts()
fig1 = pl.correlation_plot(df, list(df)[1:])
fig2 = df.plot.scatter(x='NumberOfOpenCreditLinesAndLoans', y='DebtRatio')
fig3 = pl.crosstab_plt(df, 'NumberOfDependents', 'NumberRealEstateLoansOrLines',
                    outcome_var)

# Problem 3: Pre-Processing
pl.clean_outliers(df, 'age', 100, None)
all_but_ID = list(df)[1:]
for var in all_but_ID:
    pl.clean_outliers(df, var, df[var].std()*4, df[var].std()*4)
# For the missing values I will either replace with zero and create a dummy
# indicating that (statistically more meaninful in my opinion) or replace with
# mean or median:
var_null = ['age']
pl.missing_for_nulls(df, var_null)
var_missing_to_mean = ['NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse', 'MonthlyIncome',
            'RevolvingUtilizationOfUnsecuredLines', 'NumberOfDependents',
            'NumberOfOpenCreditLinesAndLoans', 'DebtRatio']
pl.missings_to_mean(df, var_missing_to_mean)

# Problem 4:
pl.gen_bins(df, 'age', [0, 25, 50, 75, 100])
var_quantiles = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio',
                 'NumberOfTime30-59DaysPastDueNotWorse', 'MonthlyIncome', 
                 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 
                 'NumberOfTime60-89DaysPastDueNotWorse', 
                 'NumberOfOpenCreditLinesAndLoans', 'NumberOfDependents']
var_quantiles_mod = pl.gen_quantiles(df, var_quantiles, 3, 
                                  ["high", "medium", "low"])
# Now I will generate dummies for each group of the categorical data:
var_categorical = ['zipcode', 'age_bins']
var_dummies_mod = pl.gen_dummies(df, var_categorical + var_quantiles_mod)
df.drop(var_categorical + var_quantiles_mod, axis=1, inplace=True)

# Data in the format required by sklearn:
y_data, x_data = pl.data_format(df, outcome_var)

# Problem 5:    
x_train, x_test, y_train, y_test = pl.transform_data(x_data, y_data, \
                                                  test_size=0.33)

# For sklearn command I consider it more flexible to use the command directly
# because they have better documentation and this will allow me to add
# more of their functionalities depending on how I use it.
# Fit tree:
tree = pl.DecisionTreeClassifier(max_depth=10)
tree.fit(x_train, y_train)
y_tree = tree.predict(x_test)
tree.score(x_test, y_test)

# Fit forest:
rforst = pl.RandomForestClassifier(n_estimators=100, verbose=True, \
                                random_state=123)
rforst.fit(x_train, y_train)
rforst.score(x_test, y_test)
