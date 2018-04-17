
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[2]:


train_data = pd.read_table("data/train.dat", sep=" ", header=None)
train_data.head()


# In[3]:


pca1 =PCA()
pca1.fit(train_data)
pca_data1 = pca1.transform(train_data)


# In[4]:


pca_df = pd.DataFrame(pca_data1)


# In[5]:


pca_df.head()


# In[6]:


train = pca_df.iloc[:,0:50]


# In[7]:


train.head()


# In[8]:


target = pd.read_csv("data/train.labels", sep='\n', header=None, skiprows=0)


# In[9]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=0).fit(train, target)


# In[10]:


test_data = pd.read_table("data/test.dat", sep=" ", header=None)
test_data.head()


# In[11]:


pca2 = PCA()
pca2.fit(test_data)
pca_data2 = pca2.transform(test_data)


# In[12]:


pca_df_test = pd.DataFrame(pca_data2)
pca_df_test.head()


# In[13]:


test = pca_df_test.iloc[:,0:50]
test.head()


# In[14]:


predicted_test_data = lr_model.predict(test)
print predicted_test_data


# In[15]:


outdf = pd.DataFrame(predicted_test_data)
outdf.to_csv('data/results.dat',index= False, header=None)

