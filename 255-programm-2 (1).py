
# coding: utf-8

# In[152]:


from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, TruncatedSVD
kclf = KNeighborsClassifier(n_neighbors=10)


# In[ ]:


from numpy import loadtxt
train = loadtxt('/Users/ishaan/Documents/255-Prog-2/data/train.dat') 
test = loadtxt('/Users/ishaan/Documents/255-Prog-2/data/test.dat')
labels = loadtxt('/Users/ishaan/Documents/255-Prog-2/data/train.labels')
sample_format = loadtxt('/Users/ishaan/Documents/255-Prog-2/data/format.dat')


# In[ ]:


#Dimensionality Reduction

svd = TruncatedSVD(n_components = 80)
x_rd = svd.fit(train).transform(train)


# In[ ]:



#Using K Nearest Neighbours to classify data

kclf = kclf.fit(train, labels)
pred = kclf.predict(test)


np.savetxt('/Users/ishaan/Desktop/predictions17.dat', pred, delimiter=',',fmt='%i')


# In[ ]:




