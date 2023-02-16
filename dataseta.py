#!/usr/bin/env python
# coding: utf-8

# In[305]:


import pandas as pd
import numpy as np


# In[306]:


from sklearn.ensemble import RandomForestRegressor


# In[307]:


data = pd.read_csv("D:\\New folder (2)\\Customers.csv")


# In[308]:


df= pd.DataFrame(data)


# In[309]:


df.head(5)


# In[310]:


df.isnull().sum()


# In[311]:


df["Profession"] = df["Profession"].fillna(value = 0)


# In[312]:


df.isnull().sum()


# In[313]:


dummy = pd.get_dummies(df['Gender'])


# In[314]:


df2 = pd.concat((df,dummy), axis = 1)


# In[ ]:





# In[315]:


x = df2.drop(['Spending_Score', 'Profession', 'Gender', 'Female'], axis="columns")

print(x.shape)
x


# In[358]:


y = df.Spending_Score
print(y.shape)


# In[359]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)


# In[360]:


len(x_train)


# In[371]:


model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(x_train, y_train)


# In[372]:


y_pred=model.predict(x_test)


# In[375]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[ ]:




