#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.simplefilter("ignore")


# In[2]:


#load the data 
df_wqp = pd.read_csv('Wine_data.csv')
df_wqp


# In[3]:


df_wqp.info()


# In[4]:


df_wqp.isnull().sum()


# In[6]:


df_wqp["good_quality"]=[1 if x>=6 else 0 for x in df_wqp["quality"]]
df_wqp


# In[10]:


# splitting the data
x = df_wqp[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y = df_wqp['good_quality']
x


# In[11]:


y


# In[21]:


#training the model
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)


# In[14]:


x_train.shape


# In[15]:


x_test.shape


# In[16]:


y_train.shape


# In[17]:


y_test.shape


# In[18]:


# fitting the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)
y_pred


# In[19]:


# Calculate the model's performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[20]:


#plotting the regression line
wqp_plot = sns.regplot(y_test,y_pred)
wqp_plot.set_xlabel('Actual quality')
wqp_plot.set_ylabel('Predicted quality')


# In[ ]:




