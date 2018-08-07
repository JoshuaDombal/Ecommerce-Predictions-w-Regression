import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 1000)

customers = pd.read_csv('Ecommerce_Customers.csv')
#print(df.head())

#sns.distplot(df['Yearly Amount Spent'])
#plt.show()

#sns.pairplot(df)
#plt.show()

### AVERAGES ---------------------------------------
#avg_money = np.average(df['Yearly Amount Spent'])
#avg_session = np.average(df['Avg. Session Length'])
#avg_app_time = np.average(df['Time on App'])
#avg_site_time = np.average(df['Time on Website'])
avg_membership = np.average(customers['Length of Membership'])
#print(avg_membership)
###       ------------------------------------------

print(customers.describe())

sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)
plt.show()

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.intercept_)
print(X_train.columns)

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
print(cdf)

# PREDICTIONS ______________________________________________________________________________
predictions = lm.predict(X_test)
print("Predictions: ")
print(predictions)

plt.scatter(y_test, predictions)
sns.distplot((y_test-predictions))
plt.show()

from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))