#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Right Click on Data Folder - Properties - Security - Object Name -
# Copy Folder Path. 
import io
get_ipython().run_line_magic('cd', "'/Users/rajeshprabhakarkaila/Desktop/Hackathon/HousePrice Prediction'")


# In[3]:


housetrain=pd.read_csv("train.csv")


# In[4]:


housetest=pd.read_csv("test.csv")


# In[5]:


print(housetrain.shape)
print(housetest.shape)

# Train data is complete data with depenendent variable that will be used
# for model building
# Test data will not have dependent variable and it needs to be predicted
# using train model.
# Sample Submission is for submitting predictions into competition site.


# In[6]:


housetrain.info()


# In[7]:


housetest.info()


# In[8]:


# Data Preprocessing - Train & Test Data must have same preprocessing
# Combine both train data and test data into 1 dataframe
# But for combining or concatenation of dataframes the number of variables
# and sequence of variables must be same.


# In[9]:


# Add Dependent Variable temporariliy to test data
housetest['SalePrice']='test'


# In[10]:


# Conacatenate both Data Frames
combinedf=pd.concat([housetrain,housetest],axis=0)
# axis=0 refers to Row wise Concatenation


# In[11]:


combinedf.shape


# In[12]:


pd.set_option("display.max_rows",82)
combinedf.isnull().sum().sort_values(ascending=False)


# In[13]:


# Split Data into object and numeric
objectcols=combinedf.select_dtypes(include=['object'])
numericcols=combinedf.select_dtypes(include=np.number)
# np.number selects both int64 and float64


# In[14]:


print(objectcols.shape)
print(numericcols.shape)


# In[15]:


objectcols.isnull().sum().sort_values(ascending=False)


# In[16]:


# IF Missing values are more than 70% of the column, impute with word
# missing or Not Avaialable
# IF Missing values are less than 70% and variable type
# numerical - mean or median
# object - mode or most_frequent


# In[17]:


(objectcols.isnull().sum().sort_values(
    ascending=False)/objectcols.shape[0])*100


# In[18]:


objectcols.columns


# In[19]:


notavailable=['Alley','FireplaceQu','PoolQC', 'Fence', 'MiscFeature']


# In[20]:


for col in notavailable:
    objectcols[col]=objectcols[col].fillna("NotAvailable")


# In[21]:


garage_cols=[col for col in objectcols if col.startswith("Gar")]
garage_cols


# In[22]:


for col in garage_cols:
    freq=objectcols[col].value_counts(dropna=False)
    print(freq)


# In[23]:


Bsmt_cols=[col for col in objectcols if col.startswith("Bsmt")]
Bsmt_cols


# In[24]:


for col in Bsmt_cols:
    freq=objectcols[col].value_counts(dropna=False)
    print(freq)


# In[25]:


# Based on Frequency Analysis - Mode or most_frequent imputation can
# be done for all missing values in object cols.
for col in objectcols.columns:
    objectcols[col]=objectcols[col].fillna(
    objectcols[col].value_counts().idxmax())
# idxmax() - identifies the maximum frequency categoryname & uses it for
# imputation


# In[26]:


numericcols.columns
# Within Numericcols, there might be categorical cols like dates, ratings,
# dummyencoded (0/1),etc.
# Seperate categorical cols within numeric cols into seperate dataframe


# In[27]:


categorycols=numericcols[['OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd','GarageYrBlt',
                         'MoSold', 'YrSold']]


# In[28]:


numericcols=numericcols.drop(['OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd','GarageYrBlt',
                         'MoSold', 'YrSold'],axis=1)


# In[29]:


print(objectcols.shape)
print(numericcols.shape)
print(categorycols.shape)


# In[30]:


categorycols.isnull().sum().sort_values(ascending=False)


# In[31]:


categorycols.GarageYrBlt=categorycols.GarageYrBlt.fillna(
categorycols.GarageYrBlt.value_counts().idxmax())


# In[32]:


for col in numericcols.columns:
    numericcols[col]=numericcols[col].fillna(numericcols[col].median())


# In[33]:


# Dummy Variable Encoding or Label Encoding of object & category cols
from sklearn.preprocessing import LabelEncoder


# In[34]:


le=LabelEncoder()


# In[35]:


numericcols['SalePrice']=objectcols.SalePrice


# In[36]:


objectcols=objectcols.drop('SalePrice',axis=1)


# In[37]:


objectcols_encode=objectcols.apply(le.fit_transform)


# In[38]:


categorycols_encode=categorycols.apply(le.fit_transform)


# In[39]:


from sklearn.preprocessing import StandardScaler


# In[40]:


scaler=StandardScaler()


# In[41]:


numeric_scaled=scaler.fit_transform(numericcols.drop(['Id','SalePrice'],
                                                    axis=1))


# In[42]:


numeric_scaled=pd.DataFrame(numeric_scaled,
                            columns=numericcols.columns[1:30])


# In[43]:


numeric_scaled.head()


# In[44]:


objectcols_encode=objectcols_encode.reset_index(drop=True)


# In[45]:


numeric_scaled=numeric_scaled.reset_index(drop=True)


# In[46]:


categorycols_encode=categorycols_encode.reset_index(drop=True)


# In[47]:


combinedf_clean=pd.concat([numeric_scaled,objectcols_encode,
                          categorycols_encode],axis=1)


# In[48]:


numericcols=numericcols.reset_index(drop=True)


# In[49]:


combinedf_clean['SalePrice']=numericcols.SalePrice


# In[50]:


# Split dataframe back to train and test
housetrain_df=combinedf_clean[combinedf_clean.SalePrice!='test']
housetest_df=combinedf_clean[combinedf_clean.SalePrice=='test']


# In[51]:


housetest_df=housetest_df.drop('SalePrice',axis=1)


# In[52]:


# Split Data into dependentvariable(y) & Independent Variables(X)
y=housetrain_df.SalePrice
X=housetrain_df.drop('SalePrice',axis=1)


# In[53]:


y=y.astype("int64")


# In[54]:


from sklearn.linear_model import LinearRegression


# In[55]:


reg=LinearRegression()


# In[56]:


regmodel=reg.fit(X,y)


# In[57]:


regmodel.score(X,y)  # R Square


# In[58]:


regtestpredict=regmodel.predict(housetest_df)


# In[59]:


pd.DataFrame(regtestpredict).to_csv("mlr.csv")


# In[60]:


from sklearn.tree import DecisionTreeRegressor


# In[61]:


tree=DecisionTreeRegressor()


# In[62]:


treemodel=tree.fit(X,y)


# In[63]:


treemodel.score(X,y)


# In[64]:


treepredict=treemodel.predict(housetest_df)


# In[65]:


pd.DataFrame(treepredict).to_csv("dectree.csv")


# In[66]:


from sklearn.ensemble import RandomForestRegressor


# In[67]:


RF=RandomForestRegressor(n_estimators=3000)


# In[68]:


RFmodel=RF.fit(X,y)


# In[69]:


RFmodel.score(X,y)


# In[70]:


RFpredict=RFmodel.predict(housetest_df)


# In[71]:


pd.DataFrame(RFpredict).to_csv("RF.csv")


# In[72]:


from sklearn.ensemble import GradientBoostingRegressor


# In[73]:


gbm=GradientBoostingRegressor(n_estimators=3000)


# In[74]:


gbmmodel=gbm.fit(X,y)


# In[75]:


gbmmodel=gbm.fit(X,y)


# In[76]:


gbmpredict=gbmmodel.predict(housetest_df)


# In[77]:


pd.DataFrame(gbmpredict).to_csv("gbm.csv")


# In[ ]:




