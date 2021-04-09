#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import sklearn
import scipy

 
sns.set()


# In[2]:


#from google.colab import files 
#uploaded=files.upload() 
df=pd.read_csv('indian_liver_patient.csv') 
#dataset=df.values


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df['Gender']=df.Gender[df.Gender == 'male'] = 1
df['Gender']=df.Gender[df.Gender == 'female'] = 0


# In[7]:


df.describe().T


# In[8]:


data_feature = df.columns

for feature in data_feature:
    p = sns.distplot(a = df[feature])
    plt.show()


# In[9]:


#df=df.drop(columns=['Skin darkening (Y/N)'])   
df['Age'] = df['Age'].fillna(df['Age'].mean()) 


# In[10]:


#df=df.drop(columns=['  I   beta-HCG(mIU/mL)']) 
df['Gender'] = df['Gender'].fillna(df['Gender'].mean())  


# In[11]:


#df=df.drop(['hair growth(Y/N)'],axis=1) 
df['Total_Bilirubin'] = df['Total_Bilirubin'].fillna(df['Total_Bilirubin'].mean()) 


# In[12]:


#df=df.drop(['BMI'],axis=1) 
df['Direct_Bilirubin'] = df['Direct_Bilirubin'].fillna(df['Direct_Bilirubin'].mean()) 


# In[13]:


#df=df.drop(['Weight gain(Y/N)'],axis=1) 
df['Alkaline_Phosphotase'] = df['Alkaline_Phosphotase'].fillna(df['Alkaline_Phosphotase'].mean())  


# In[14]:


#df= df.dropna(axis='rows')
df['Alamine_Aminotransferase'] = df['Alamine_Aminotransferase'].fillna(df['Alamine_Aminotransferase'].mean()) 


# In[15]:


df['Total_Protiens'] = df['Total_Protiens'].fillna(df['Total_Protiens'].mean()) 


# In[16]:


df['Aspartate_Aminotransferase'] = df['Aspartate_Aminotransferase'].fillna(df['Aspartate_Aminotransferase'].mean()) 


# In[17]:


df['Albumin '] = df['Albumin'].fillna(df['Albumin'].mean()) 


# In[18]:


df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean()) 


# In[ ]:





# In[19]:


p = df.hist(figsize = (12,12))


# In[20]:


df.describe().T


# In[21]:


#df['Cycle length(days)'] = df['Cycle length(days)'].fillna(df['Cycle length(days)'].mean())


# In[22]:


for i in range(10):
    print(df.columns[i])


# In[23]:


p = df.hist(figsize = (20,20))


# In[24]:


#sns.pairplot(data =df)
plt.show()


# In[ ]:





# In[25]:


from scipy import stats
for feature in df.columns:
    stats.probplot(df[feature], plot = plt)
    plt.title(feature)
    plt.show()


# In[26]:


df.head()


# In[27]:


X = df.iloc[:,1 :-1]
y = df.iloc[:, 0]
print(y) 


# In[ ]:





# In[28]:


X.head()


# In[29]:


y.head()


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[31]:


#from sklearn.preprocessing import MinMaxScaler 
#from sklearn.preprocessing import StandardScaler 
#scaler=StandardScaler()
#scaler.fit(df) 
#scaled_data=scaler.transform(df) 
#scaled_data 


# In[32]:


#from sklearn.decomposition import PCA 

#pca=PCA(n_components=2) 

#pca.fit(scaled_data) 
#x_pca=pca.transform(scaled_data) 


# In[33]:


#scaled_data.shape 


# In[34]:


#x_pca.shape 


# In[35]:


from sklearn.model_selection import train_test_split 
print (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# In[ ]:





# In[36]:


def svm_classifier(X_train, X_test, y_train, y_test):
    
    classifier_svm = SVC(kernel = 'rbf', random_state = 0)
    classifier_svm.fit(X_train, y_train)

    y_pred = classifier_svm.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_svm.score(X_train, y_train)}\nTest score : {classifier_svm.score(X_test, y_test)}")


# In[37]:


def knn_classifier(X_train, X_test, y_train, y_test):
    
    classifier_knn = KNeighborsClassifier(metric = 'minkowski', p = 2)
    classifier_knn.fit(X_train, y_train)

    y_pred = classifier_knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_knn.score(X_train, y_train)}\nTest score : {classifier_knn.score(X_test, y_test)}")


# In[38]:


def naive_classifier(X_train, X_test, y_train, y_test):
    
    classifier_naive = GaussianNB()
    classifier_naive.fit(X_train, y_train)

    y_pred = classifier_naive.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_naive.score(X_train, y_train)}\nTest score : {classifier_naive.score(X_test, y_test)}")


# In[39]:


def tree_classifier(X_train, X_test, y_train, y_test):
    
    classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_tree.fit(X_train, y_train)

    y_pred = classifier_tree.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_tree.score(X_train, y_train)}\nTest score : {classifier_tree.score(X_test, y_test)}")


# In[40]:


def forest_classifier(X_train, X_test, y_train, y_test):
    classifier_forest = RandomForestClassifier(criterion = 'entropy', random_state = 0)
    classifier_forest.fit(X_train, y_train)

    y_pred = classifier_forest.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_forest.score(X_train, y_train)}\nTest score : {classifier_forest.score(X_test, y_test)}")


# In[41]:


def logistic_regression (X_train,X_test,y_train,y_test):

  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression(random_state = 0)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  
  return print(f"Train score : {model.score(X_train, y_train)}\nTest score : {model.score(X_test, y_test)}")


# In[42]:



def print_score(X_train, X_test, y_train, y_test):
    print("SVM:\n")
    svm_classifier(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()

    print("KNN:\n")
    knn_classifier(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()

    print("Naive:\n")
    naive_classifier(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()

    print("Decision Tree:\n")
    tree_classifier(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()

    print("Random Forest:\n")
    forest_classifier(X_train, X_test, y_train, y_test)
    
    print("-"*100)
    print()

    print("logistic Regression:\n")
    logistic_regression(X_train, X_test, y_train, y_test)

    print("-"*100)
    print()


# In[43]:


print_score(X_train, X_test, y_train, y_test)


# In[44]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot = True, cmap = "Blues")
plt.show() 


# In[45]:


classifier_forest = RandomForestClassifier(criterion = 'entropy')
classifier_forest.fit(X_train, y_train)
y_pred = classifier_forest.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm
print(cm)


# In[46]:


from sklearn.metrics import confusion_matrix
classifier_forest = GaussianNB()
classifier_forest.fit(X_train, y_train)
y_pred = classifier_forest.predict(X_test)
print(confusion_matrix(y_test, y_pred))


# In[47]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[48]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[50]:


#from xgboost import XGBClassifier 
#from sklearn.model_selection import train_test_split 
#from sklearn.metrics import accuracy_score 

#model=XGBClassifier() 
#model.fit(X_train,y_train) 
#y_pred=model.predict(X_test) 
#accu=accuracy_score(y_test,y_pred) 
#accu=accuracy_score(y_test,y_pred) 
#print(accu) 


# In[51]:


from sklearn.ensemble import BaggingClassifier 
from sklearn.metrics import accuracy_score 
model = BaggingClassifier() 
model.fit(X_train,y_train) 
y_pred=model.predict(X_test) 
accu=accuracy_score(y_test,y_pred) 
accu=accuracy_score(y_test,y_pred) 
print(accu)  


# In[52]:


import pickle
filename = 'liver.pkl' 
pickle.dump(classifier_forest, open(filename, 'wb')) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




