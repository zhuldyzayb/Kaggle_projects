import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler # for scaling data
import seaborn as sns # visualisation
import matplotlib.pyplot as plt # visualisation
from sklearn.linear_model import LinearRegression # Linear regression model
from sklearn.metrics import mean_squared_error # MSE calculation
from sklearn.feature_selection import RFE # backward selection
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(train_data.head())

'''
Once all the data and necessary packages are loaded, we perform exploratory data analysis.
1. Take a look at the data
2. Check data typs, formats and look for missing values
3. Change data types whenever necessary. Here we decide to change only the type of cityCode column
4. Define a complete list of features
Note that the project specifically says it has to be a linear regression model, 
thus the other possible regression techniques (including random forest) are not implemented.
'''

print(train_data.info()) # no null values or missing values
train_data['cityCode'] = train_data['cityCode'].astype('category')
test_data['cityCode'] = test_data['cityCode'].astype('category')
features = train_data.columns[1:-1]
print(train_data.describe())

'''
Now once we have an idea about the data, lets take a look at some visualisations to:
1. Identify outliers
2. See possible existing patterns
'''
# Boxplots to identify outliers
fig, axes = plt.subplots(2,5, figsize = (12,5))
plt1 = sns.boxplot(train_data['squareMeters'], ax = axes[0,0])
plt2 = sns.boxplot(train_data['numberOfRooms'], ax = axes[0,1])
plt3 = sns.boxplot(train_data['floors'], ax = axes[0,2])
plt4 = sns.boxplot(train_data['numPrevOwners'], ax = axes[0,3])
plt5 = sns.boxplot(train_data['made'], ax = axes[0,4])
plt6 = sns.boxplot(train_data['basement'], ax = axes[1,0])
plt7 = sns.boxplot(train_data['attic'], ax = axes[1,1])
plt8 = sns.boxplot(train_data['garage'], ax = axes[1,2])
plt9 = sns.boxplot(train_data['hasGuestRoom'], ax = axes[1,3])
plt10 = sns.boxplot(train_data['price'], ax = axes[1,4])
plt.tight_layout()
plt.show()

# Histograms
hist_vars = ['squareMeters', 'numberOfRooms', 'floors', 'cityPartRange', 'numPrevOwners', 'made',
             'basement', 'attic', 'garage', 'hasGuestRoom', 'price']
fig, axes = plt.subplots(3,4, figsize = (12,5))
for i in range(len(hist_vars)):
    sns.histplot(train_data[hist_vars[i]], ax = axes[i//4, i%4])
plt.tight_layout()
plt.show()

'''The easiest method to remove outliers is to use 1.5*IQR rule:'''
# Remove outliers
outlier_vars = ['squareMeters', 'floors', 'made', 'basement', 'attic', 'garage']

for val in outlier_vars:
    q3 = train_data[val].quantile(0.75)
    q1 = train_data[val].quantile(0.25)
    iqr = q3 - q1
    train_data = train_data[(train_data[val]<= q3+ 1.5*iqr) & (train_data[val]>= q1 - 1.5*iqr) ]
print("The new shape of the data: ", train_data.shape)

## Updated histograms
fig, axes = plt.subplots(2,3, figsize = (12,5))
for i in range(len(outlier_vars)):
    sns.histplot(train_data[outlier_vars[i]], ax = axes[i//3, i%3])
plt.tight_layout()
plt.show()

'''
Note that the boolean type features (hasYard, hasPool, etc) as well as categorical type 
features (is, cityCode) are not included intentionally. We see above that the total number 
of outliers across the selected features are 20. Since we have enough data, we can simply 
eliminate the outliers.
'''
# Plot features agains the price
plot_vars = ['squareMeters', 'numberOfRooms', 'floors', 'cityPartRange', 'numPrevOwners', 'made', 'basement',
             'attic', 'garage', 'hasGuestRoom']
fig, axes = plt.subplots(2,5, figsize = (12,5))
for i in range(len(plot_vars)):
    sns.scatterplot(train_data, x = plot_vars[i], y='price', ax = axes[i//5, i%5])
plt.tight_layout()
plt.show()

'''
From the scatterplots above we only wee the clear pattern in the linear relationship 
between price and square meters. Let's see if there are measurable correlations between 
the variable of interest and its covariates.
'''
corrMat = train_data.corr()
sns.heatmap(corrMat)
plt.show()

'''
There are some correlation between the floors and squareMeters and numberOfRooms and 
between the latter two, which is quite logical. Some correlation between made and 
squareMeters, and garage and numberOfRooms, which are not as straightforward. Let's 
explore further with the pairplots:
'''
# Preclean for pairplot
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
train_data.dropna(inplace=True)

num_vars = ['squareMeters', 'numberOfRooms', 'floors', 'cityPartRange', 'numPrevOwners', 'made',
            'basement', 'attic', 'garage','hasGuestRoom', 'price']
sns.pairplot(train_data[num_vars])
plt.show()

'''
We see that some of the columns has much wider range than others. To avoid these features 
dominating the model, we perform rescaling. Additionally, since we have to use the parameters 
trained on the rescaled model to perform predistion on the test data, we perform rescaling on the test data too.
'''
scaler_vars = ['squareMeters', 'numberOfRooms', 'floors',
               'basement', 'attic', 'garage'] # covariates
scaler = MinMaxScaler()
train_data[scaler_vars] = scaler.fit_transform(train_data[scaler_vars])
test_data[scaler_vars] = scaler.transform(test_data[scaler_vars])
# Response variable
pmax = max(train_data['price'])
pmin = min(train_data['price'])
train_data['price'] = (train_data['price'] - pmin)/(pmax - pmin)

'''
Now we perform covariate selection using backward selection method. To assess the model 
we split the data (75-25) into training and testing sets and assess the model resibuals on a testing set.
'''
selected_features = ['squareMeters', 'numberOfRooms', 'hasYard', 'hasPool', 'floors', 'cityPartRange',
                     'numPrevOwners', 'made', 'isNewBuilt','hasStormProtector', 'basement', 'attic',
                     'garage', 'hasStorageRoom', 'hasGuestRoom']
X = train_data[selected_features]
y = train_data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.75, random_state = 123)

model = LinearRegression()
selector = RFE(model)
selector.fit(X_train, y_train)
rfe_features = selector.get_feature_names_out()
print("Seleted features are: ", rfe_features)
y_pred = selector.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("The model RMSE is: ", rmse)

plt.scatter(y_pred,y_test)
plt.xlabel("Predicted price")
plt.ylabel("Observed price")
plt.show()

'''We have achieved a pretty decent fit using backward selection. Let's check other model properties'''
X_rfe = X_train[selector.get_feature_names_out()]

import statsmodels.api as sm
X_rfe_fin = sm.add_constant(X_rfe)
finmod = sm.OLS(y_train, X_rfe).fit()
print(finmod.summary())

'''
Both R^2 and adjusted R^2 are extremely close to one indicating that almost all the variance in 
the training data is explained by the covariates. We also see a very high F-statistics suggesting 
significant relation between the independent and dependent variables.
'''
X_pred_rfe = test_data[selected_features]
y_out = selector.predict(X_pred_rfe)
submission_data = pd.DataFrame({'id':test_data['id'], 'price': (y_out*(pmax-pmin) + pmin)})
submission_data.to_csv('submission.csv', index=False)