#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.simplefilter("ignore")


# In[58]:


#loading the data
df_hp = pd.read_csv('HousingData.csv')
df_hp


# In[59]:


#splitting the data
x = df_hp[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
y = df_hp['MEDV']


# In[60]:


x


# In[61]:


y


# In[62]:


df_hp.isnull().sum()


# In[63]:


df_hp.info()


# In[64]:


# Training the model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[65]:


x_train.shape


# In[66]:


x_test.shape


# In[67]:


y_train.shape


# In[68]:


y_test.shape


# In[69]:


# Fit the model to the training data
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)
y_pred


# In[70]:


# Calculate the model's performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[71]:


#plotting the regression line
lr_plot = sns.regplot(y_test,y_pred)
lr_plot.set_xlabel('Actual prices')
lr_plot.set_ylabel('Predicted prices')


# In[ ]:




