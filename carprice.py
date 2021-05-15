#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.read_csv('carprices.csv')
df


# In[7]:


dummies=pd.get_dummies(df.CarModel)
dummies


# In[26]:


dummies=dummies.drop(columns=['Audi A5'])


# In[27]:


merged = pd.concat([df,dummies],axis='columns')
merged


# In[28]:


final=merged.drop(columns=['CarModel'])


# In[29]:


final


# In[30]:


x=final.drop(columns=['SellPrice'])


# In[34]:


y=final['SellPrice']
y


# In[35]:


reg=linear_model.LinearRegression()
reg.fit(x,y)


# In[36]:


reg.score(x,y)


# In[37]:


reg.predict([[45000,4,0,0]])


# In[ ]:




