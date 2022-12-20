# TugasProjectygKemarin
Weather Prediction Dataset
A dataset for teaching machine learning and deep leaerning
Hands-on teaching of modern machine learning and deep learning techniques heavily relies on the use of well-suited datasets.
The "weather prediction dataset" is a novel tabular dataset that was specifically created for teaching machine learning and deep learning to an academic audience.
The dataset contains intuitively accessible weather observations from 18 locations in Europe. It was designed to be suitable for a large variety of different training goals, many of which are not easily giving way to unrealistically high prediction accuracy. Teachers or instructors thus can chose the difficulty of the training goals and thereby match it with the respective learner audience or lesson objective.
The compact size and complexity of the dataset make it possible to quickly train common machine learning and deep learning models on a standard laptop so that they can be used in live hands-on sessions.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
```
Read Data
```python
df=pd.read_csv('/content/drive/MyDrive/Data Mining/seattle-weather.csv')
df.sample(10)
df.describe()
df.info()
df.isnull().sum() #Hence no missing data found in dataset
```
Now converting data, data type to date time format
```python
df['date'] = pd.to_datetime(df['date'])
```
Count number of distinc elements in specified axis
```python
df.nunique()
plt.figure(figsize=(10,5))
sns.countplot(x = 'weather',data = df)
plt.xlabel("Weather Condition",fontweight='bold',size=14)
plt.ylabel("Count",fontweight='bold',size=14)
plt.show()
plt.figure(figsize=(18,8))
sns.set_theme()
sns.lineplot(x = 'date',y='temp_max',data=df)
plt.xlabel("Date",fontweight='bold',size=13)
plt.ylabel("Temp_Max",fontweight='bold',size=13)
plt.show()
plt.figure(figsize=(14,8))
sns.pairplot(df.drop('date',axis=1),hue='weather')
plt.show()
classes={
        'drizzle':0,
        'fog':1,
        'rain':2,
        'snow':3,
        'sun':4
}
dataset=df.drop('date', axis=1)
dataset['weather']=dataset['weather'].astype('category')
dataset['weather']=dataset['weather'].cat.codes
```
Preparing X data and Y data for model training
```python
x = dataset.drop('weather',axis=1)
y = dataset['weather']
```
Split the dataset into train and test
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```
Models Included
```python
#LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier_log=LogisticRegression(random_state=0)
classifier_log.fit(x_train,y_train)
#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier_knn=KNeighborsClassifier(n_neighbors = 5)
classifier_knn.fit(x_train,y_train)
#SVC
from sklearn.svm import SVC
classifier_svc=SVC(kernel = 'linear', random_state=42)
classifier_svc.fit(x_train, y_train)
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier_gnb=GaussianNB()
classifier_gnb.fit(x_train, y_train,sample_weight=None)
#RandomForest
from sklearn.ensemble import RandomForestClassifier
classifier_rfc=RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier_rfc.fit(x_train, y_train,sample_weight=None)
#DecisionTree
from sklearn.tree import DecisionTreeClassifier
classifier_dtc = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier_dtc.fit(x_train, y_train,sample_weight=None)
#XGBoost
from xgboost.sklearn import XGBClassifier
classifier_xgb=XGBClassifier()
classifier_xgb.fit(x_train,y_train)
classifier_list=[classifier_log,classifier_knn,classifier_svc,classifier_gnb,classifier_dtc,classifier_rfc,classifier_sgdc,classifier_gbc,classifier_lgbm,classifier_xgb]

# accuracy =correct answer(marks)/total marks

def accuracy(X_test, Y_test, classifier):
    print(f"\nclassifier\n{classifier}")
    Y_pred=classifier.predict(X_test)
    correct=0
    total=len(Y_pred)
    for i,j in zip(Y_pred, Y_test):
        if i==j:
            correct+=1
    acc=(correct/total)*100
    print("\n"+str(acc)+"\n"+"-"*30+"\n")
    print("\n"+"*"*50+"\n"+classification_report(Y_test,Y_pred)+"\n")
    cm=confusion_matrix(Y_test,Y_pred)
    print(f"Confusion Matrix\n{cm}\n")
    return acc
    
accuracy_list=[]
classifier_list_str=[]

for clf in classifier_list:
    acc=accuracy(x_test, y_test, clf)
    accuracy_list.append(acc)
    classifier_list_str.append(str(clf)[:6])    
    
accuracy_list
classifier_list_str
plt.figure(figsize=(22,8))
ax = sns.barplot(x=classifier_list_str, y=accuracy_list, saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20 )
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Accuracy of different Classification Models", fontsize = 20)
plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height/100:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()
```
