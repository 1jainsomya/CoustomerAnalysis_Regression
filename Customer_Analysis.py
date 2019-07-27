#!/usr/bin/env python
# coding: utf-8

# WEBSITE VS MOBILE APP

# Aanalysis of activity of coustomer in store,mobile app and website.
# Using this analysis we will try to figure out that whether company need to focus more on Mobile app or website.
# This will help them in marketing and give direction to the company.


# # Imports Libraries




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


## Data

# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 




customers = pd.read_csv("Ecommerce Customers")


## Understanding the data




customers.head()




customers.describe()



customers.info()


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


### Jointplot to compare the Time on Website and Yearly Amount Spent

# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


### Jointplot to compare the Time on Website and Yearly Amount Spent


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


### Jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership




sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


### Pairplot to undeerstand all relationship betweeen features


sns.pairplot(customers)


### Linear model plot of Yearly Amount Spent vs Length of Membership.

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


## Training and Testing Data

y = customers['Yearly Amount Spent']


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]





from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


## Training the Model using Linear Regression

from sklearn.linear_model import LinearRegression


lm = LinearRegression()


lm.fit(X_train,y_train)


### Coefficients of the model




# The coefficients
print('Coefficients: \n', lm.coef_)


## Predicting our Data

predictions = lm.predict( X_test)


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


## Evaluating the Model


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R^2 :', metrics.r2_score(y_test, predictions))


## Residuals

sns.distplot((y_test-predictions),bins=50);


## Results

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


## Company should focus more on their app or on website?

# The app is doing far better work.
# According to my analysis focus on the website to catch up with the performance of the mobile app. 
# Also, the customer tends to get bound with the app they installed rather than a website where they can be easily distracted.
 
