import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv('Ecommerce Customers')

customers.head()
# customers.describe()
# customers.info()
# sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')
# sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent')

# sns.jointplot(data=customers, x='Time on App', y='Length of Membership', kind='hex')
# sns.pairplot(customers)

# sns.lmplot(data=customers, x='Length of Membership', y='Yearly Amount Spent')

# Customers column
customers.column

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App',
               'Time on Website', 'Length of Membership']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
# Dealing with training data
lm.fit(X_train, y_train)

lm.coef_
prediction = lm.predict(X_test)

plt.scatter(y_test, prediction)
plt.xlabel('Y test(true value)')
plt.ylabel('Predicted values')

print('MAE', metrics.mean_absolute_error(y_test, prediction))
print('MSE', metrics.mean_squared_error(y_test, prediction))
print('RMSE', np.sqr(metrics.mean_squared_error(y_test, prediction)))

metrics.explained_variance_score(y_test, prediction)

sns.distplot((y_test - prediction), bins=50)

# So which data feature have got the highest effect on the ecommerce
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])

