#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
from zipfile import ZipFile 


# In[2]:


# os.chdir("./")


# In[3]:


# for item in os.listdir("./"):
#    if item.endswith(".zip"): # check for ".zip" extension
#         file_name = os.path.abspath(item) # get full path of files
#         zip_ref = zipfile.ZipFile(file_name) # create zipfile object
#         zip_ref.extractall("./") # extract file to dir
#         zip_ref.close() # close file
#         os.remove(file_name) # delete zipped file


# In[4]:


# df = pd.read_csv("./sample_submission.csv")
# df.head(2)


# In[5]:


# df.info()


# In[6]:


# # Plot histograms for Price, Latitude & Longitude
# %matplotlib inline
# df.hist(edgecolor='black', linewidth= 1.2)
# fig = plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()


# In[7]:


df2 = pd.read_json("./train.json")
df2.head(2)


# In[31]:


maxPrice = int(max(df2['price']))
minPrice = int(min(df2['price']))
maxLongitude = int(max(df2['longitude']))
minLongitude = int(min(df2['longitude']))
maxLatitude = int(max(df2['latitude']))
minLatitude = int(min(df2['latitude']))
print("Max price = ", maxPrice, ", Min price = ", minPrice)
print("Max longitude = ", maxLongitude, ", Min longitude = ", minLongitude)
print("Max latitude = ", maxLatitude, ", Min latitude = ", minLatitude)


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
df2.hist(column='price', range=(minPrice, maxPrice), edgecolor='black', linewidth= 1.2)
df2.hist(column='longitude', range=(minLongitude, maxLongitude), edgecolor='black', linewidth= 1.2)
df2.hist(column='latitude', range=(minLatitude, maxLatitude), edgecolor='black', linewidth= 1.2)
fig = plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[ ]:





# In[ ]:




