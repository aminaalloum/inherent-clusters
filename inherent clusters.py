#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
df = [['Skirt', 'Sneakers', 'Scarf', 'Pants', 'Hat'],

        ['Sunglasses', 'Skirt', 'Sneakers', 'Pants', 'Hat'],

        ['Dress', 'Sandals', 'Scarf', 'Pants', 'Heels'],

        ['Dress', 'Necklace', 'Earrings', 'Scarf', 'Hat', 'Heels', 'Hat'],

      ['Earrings', 'Skirt', 'Skirt', 'Scarf', 'Shirt', 'Pants']]


# In[3]:


tr=TransactionEncoder()
tr_ary=tr.fit(df).transform(df)
df=pd.DataFrame(tr_ary,columns=tr.columns_)
df


# In[4]:


from mlxtend.frequent_patterns import apriori
apriori(df,min_support=0.6)


# In[6]:


frequent_itm=apriori(df,min_support=0.6,use_colnames=True)
frequent_itm


# In[8]:


from mlxtend.frequent_patterns import association_rules
association_rules(frequent_itm,metric="confidence",min_threshold=0.7)


# In[12]:


association_rules(frequent_itm,metric="lift",min_threshold=1.20)


# In[13]:


df=pd.read_csv("./Downloads/Market_Basket_Optimisation.csv")
df


# In[14]:


df.isnull().sum()


# In[15]:


df.info()


# In[22]:


tr_ary=tr.fit(df).transform(df)
df=pd.DataFrame(tr_ary,columns=tr.columns_)
df


# In[54]:


from mlxtend.frequent_patterns import apriori
apriori(df,min_support=0.001)
#apriori(df,min_support=0.6)


# In[67]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

df = df.applymap(encode_units)
frqt_itm=apriori(df,min_support=0.0000006,use_colnames=True)
frqt_itm


# In[71]:


from mlxtend.frequent_patterns import association_rules
association_rules(frqt_itm,metric="lift",min_threshold=1.20)


# In[73]:


association_rules(frqt_itm,metric="confidence",min_threshold=0.7)


# In[ ]:




