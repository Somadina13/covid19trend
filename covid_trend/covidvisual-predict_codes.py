import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
country=pd.read_csv('country_wise.csv')
country.head()
country.tail()
country.isnull().sum()
duplicates = country[country.duplicated()]
print(duplicates)
WHO_map = {'Europe':0, 'Africa':1, 'Americas':2, 'Eastern Mediterranean':3, 
           'Western Pacific':4, 'South-East Asia':5}
country['WHO Region'] = country['WHO Region'].map(WHO_map)
country.head()
country['Daily growth rate'] = ((country['New cases'] / country['Confirmed']) * 100).round(2)
country['Mortality rate'] = ((country['Deaths'] / country['Confirmed']) * 100).round(2)
country['Recovery rate'] = ((country['Recovered'] / country['Confirmed']) * 100).round(2)
country.isnull().sum()
country.head()
C=country.groupby('WHO Region')['Confirmed'].sum()
C.plot(kind='line')
plt.title('COVID-19 Trends Across Region')
plt.xlabel('WHO Region')
plt.ylabel('Count')
R=country.groupby('WHO Region')['Recovered'].sum()
R.plot(kind='line')
plt.title('COVID-19 Trends Across Region')
plt.xlabel('WHO Region')
plt.ylabel('Count')
M=country.groupby('WHO Region')['Mortality rate'].sum()
M.plot(kind='bar')
plt.title('Mortality Rate Across Region')
plt.xlabel('WHO Region')
plt.ylabel('Count')
RR=country.groupby('WHO Region')['Recovery rate'].sum()
RR.plot(kind='bar')
plt.title('Recovery Rate Across Region')
plt.xlabel('WHO Region')
plt.ylabel('Count')
DGR=country.groupby('WHO Region')['Daily growth rate'].sum()
RR.plot(kind='line')
plt.title('Growth Rate Across Region')
plt.xlabel('WHO Region')
plt.ylabel('Count')
plt.figure(figsize=(8,6))
sns.heatmap(country.corr(), annot=True, cmap='viridis')
plt.figure(figsize=(10, 6))
corr = country[['Confirmed', 'Deaths', 'Recovered', 'Active', 'Daily growth rate',
           'Mortality rate']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
sns.boxplot(x='WHO Region', y='Confirmed', data=country)
plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(8,6))
sns.boxplot(x="WHO Region", y="Confirmed", data=country);
plt.title('Outliers in Confirmed Cases by WHO Region')
country.isnull().sum()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x = country[['Recovered','New cases','New deaths','New recovered','1 week change']]
y = country['Confirmed']
from sklearn.linear_model import LinearRegression
print(x_train.shape)
print(x_test.shape)
model = LinearRegression()
model.fit(x_train, y_train)
predictions=model.predict(x_test)
print(predictions[:10])
print('Training set score: {:.2f}'.format(model.score(x_train,y_train)))
print('Testing set score: {:.2f}'.format(model.score(x_test,y_test)))

