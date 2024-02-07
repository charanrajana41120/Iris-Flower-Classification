#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


iris_data = pd.read_csv('C:\\Users\\Admin\\Desktop\\Iris.csv')


# In[27]:


column_names = ["id","sepal_length", "sepal_width", "petal_length", "petal_width", "Species"]


# In[28]:


iris_data.head()


# In[29]:


iris_data.describe()


# In[30]:


sns.pairplot(iris_data, hue="Species")
plt.show()


# In[31]:


X = iris_data.drop("Species", axis=1)
X


# In[32]:


y = iris_data["Species"]
y


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[34]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)


# In[35]:


y_pred = knn.predict(X_test)


# In[36]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[37]:


print(classification_report(y_test, y_pred))


# In[38]:


X_test.head(2)  


# In[62]:


new_data = pd.DataFrame({"Id": 149,"SepalLengthCm":[5.9],"SepalWidthCm":3.0,"PetalLengthCm":5.1,"PetalWidthCm":1.8})


# In[63]:


prediction = knn.predict(new_data)


# In[64]:


prediction[0]


# In[53]:


iris_data.iloc[0:]


# In[ ]:




