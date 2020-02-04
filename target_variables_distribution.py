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


# In[25]:


df2['interest_level'].value_counts().plot(kind="bar")
plt.xlabel('Interest Level')
plt.xticks(rotation=0)
plt.ylabel('Counts')
plt.title('Interest Level Distribution')
plt.show()


# In[ ]:




