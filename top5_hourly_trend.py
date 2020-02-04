#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
from zipfile import ZipFile 


# In[7]:


df2 = pd.read_json("./train.json")
df2.head(2)


# In[8]:


# Extract year, month, day, hour from created column
print(df2['created'].dtype)
df2['created'] = pd.to_datetime(df2['created'])
print(df2['created'].dtype)
df2['year'] = df2['created'].dt.year
df2['month'] = df2['created'].dt.month
df2['day'] = df2['created'].dt.day
df2['hour'] = df2['created'].dt.hour
df2.head(2)


# In[9]:


hour_counts = df2['hour'].value_counts().sort_index()
hour_counts.plot.bar(x="Hour", y="Count")
plt.xlabel('Hours')
plt.ylabel('Listing Count')
plt.title('Hour-wise Listing Trend by Count')
plt.show()

formatted_hour_counts = hour_counts.reset_index()
formatted_hour_counts.columns = ['Hour', 'Count']

print("Top 5 busiest hours of postings")
formatted_hour_counts.sort_values(by=['Count'], ascending=False).head(5)

