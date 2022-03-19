#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore Data Science Internship (VIP)
# ## Name: Kajal Kashyap
# 
# ## Title: Iris Flowers Classification ML Project
# 
# ## Batch: March

# # Step 1:Data Collection

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading Dataset

# In[2]:


df=pd.read_csv("C:/Users/Mehvish/Downloads/iris.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df=df.replace({'species':{'Iris-setosa':1,'Iris-virginica':2,'Iris-versicolor':3}})
df.head()


# In[6]:


df.info()


# ## Step2: Checking nulls

# In[7]:


df.isnull()


# In[8]:


df.columns


# ## Step 3: Understanding Data

# In[9]:


df.describe()


# In[10]:


df.nunique


# In[11]:


df.species


# In[106]:


x=np.array(df.iloc[:,3]).reshape(-1,1)
y=np.array(df.iloc[:,-1]).reshape(-1,1)


# In[107]:


x


# In[108]:


y


# In[109]:


cor=df.corr()
cor


# ## Step 4: Data Visualization
# 

# ### Heatmap

# In[110]:


sns.heatmap(cor,annot=True)


# In[111]:


df.sample


# ### Box  Plot

# In[112]:


plt.figure(figsize=(8,5))
sns.boxplot(data=df)
plt.show()


# ### Pairplot Graph

# In[113]:


sns.pairplot(df,hue='species')


# ## Step 6:Model Building
In the iris flower classification we have measurements for which we know the correct species of iris, this is a supervised learning problem. We want to predict one of several options (the species of iris), making it an example of a classification problem.

To test how the model’s performance,this is usually done by splitting the labelled data we have collected into two parts with 80%-20%. One part of the data is used to build the machine learning model, and is called the training data (i.e. X_train and y_train). The rest of the 20% data will be used to test how well the model works; this is called the test data(i.e. X_test, y_test).

X is having all the dependent variables.

Y is having an independent variable (here in this case ‘species’ is an independent variable).
# In[115]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[116]:


x_train.shape


# In[117]:


#from sklearn.linear_model import LinearRegression


# In[118]:


#lr=LinearRegression()


# In[119]:


#lr.fit(x_train,y_train)


# In[120]:


y_predict=lr.predict(x_test)


# In[121]:


y_predict


# In[122]:


from sklearn.svm import SVC


# In[123]:


classifier=SVC(kernel='linear',random_state=0)


# In[124]:


classifier.fit(x_train,y_train)


# In[125]:


#from sklearn.metrics import confusio



# In[126]:


#confusion_matrix(y_test,y_predict)


# In[127]:


#from sklearn.metrics import r2_score


# In[128]:


#r2_score=lr.score(y_test,y_predict)
#print(r2_score*100,'%')


# In[129]:


y_predict=classifier.predict(x_test)


# In[130]:


from sklearn.metrics import confusion_matrix


# In[131]:


confusion_matrix(y_test,y_predict)


# In[132]:


from sklearn.metrics import r2_score


# In[133]:


r2_score=lr.score(y_test,y_predict)
print(r2_score*100,'%')


# In[ ]:




