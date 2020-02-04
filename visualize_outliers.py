#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
from zipfile import ZipFile 


# In[2]:


df2 = pd.read_json("./train.json")
df2.head(2)


# In[3]:


df2.info()


# In[4]:


df2.describe()


# In[5]:


# show histrograms for numerical attirbutes/columns
df2.hist()
plt.show()


# In[44]:


indexes = list(df2.index.values.tolist())
values = list(df2['bathrooms'][indexes])

# for index in indexes:
#     print(index, ",", df2['bathrooms'][index])
 
plt.scatter(indexes, df2['bathrooms'][indexes], edgecolor = 'black')
plt.xlabel('indexes')
plt.ylabel('Number of Bathrooms')
plt.show()
df2['bathrooms'].describe()


# In[38]:


# For testing
# df2['bathrooms'].head(10)


# In[43]:


indexes = list(df2.index.values.tolist())
values = list(df2['bedrooms'][indexes])
    
plt.scatter(indexes, df2['bedrooms'][indexes], edgecolor = 'black')
plt.xlabel('indexes')
plt.ylabel('Number of Bedrooms')
plt.show()
df2['bedrooms'].describe()


# In[45]:


indexes = list(df2.index.values.tolist())
values = list(df2['latitude'][indexes])
    
plt.scatter(indexes, df2['latitude'][indexes], edgecolor = 'black')
plt.xlabel('indexes')
plt.ylabel('Maginitude of Latitude')
plt.show()
df2['latitude'].describe()


# In[47]:


indexes = list(df2.index.values.tolist())
values = list(df2['longitude'][indexes])
    
plt.scatter(indexes, df2['longitude'][indexes], edgecolor = 'black')
plt.xlabel('indexes')
plt.ylabel('Maginitude of longitude')
plt.show()
df2['longitude'].describe()


# In[46]:


indexes = list(df2.index.values.tolist())
values = list(df2['price'][indexes])
    
plt.scatter(indexes, df2['price'][indexes], edgecolor = 'black')
plt.xlabel('indexes')
plt.ylabel('price')
plt.show()
df2['price'].describe()

