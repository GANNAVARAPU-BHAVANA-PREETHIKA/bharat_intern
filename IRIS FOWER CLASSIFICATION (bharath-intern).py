#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


# In[6]:


# Load the iris dataset
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
df_iris = pd.read_csv(path, names = headernames)
df_iris


# In[7]:


df_iris.info()


# In[10]:


df_iris.isnull().sum()


# In[11]:


df_iris['species'].value_counts()


# In[13]:


# splitting the data
x = df_iris.iloc[:,:4]
y = df_iris.iloc[:,4]
x


# In[14]:


y


# In[15]:


# training the model
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# In[19]:


y_train.shape


# In[20]:


y_test.shape


# In[21]:


#fitting the model
model = LogisticRegression()
model.fit(x_train , y_train)

#predicting the species of the flower
y_pred = model.predict(x_test)
y_pred


# In[22]:


#confusioon matrix
con_mat = confusion_matrix(y_test , y_pred)
con_mat


# In[23]:


#accuracy scoore
acc = accuracy_score(y_test , y_pred)*100
acc


# In[ ]:




