import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import (recall_score,
                             accuracy_score,
                             precision_score,
                             f1_score)                             
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

features = pd.read_csv('features.csv')
stores= pd.read_csv('stores.csv')
train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
print("features.csv")
print(features.columns)
print("stores.csv")
print(stores.columns)
print("train.csv")
print(train.columns)
print("test.csv")
print(test.columns)

dataset = pd.read_csv('train.csv', names= ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday'], sep= ',', header=0)
features = pd.read_csv('features.csv', names= ['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
       'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment',
       'IsHoliday'], sep= ',', header=0).drop(columns=['IsHoliday'])
stores = pd.read_csv('stores.csv', names= ['Store', 'Type', 'Size'], sep= ',', header=0)
dataset = dataset.merge(stores, how='left').merge(features, how='left')
dataset


colunas = features.drop('Store', axis=1)
X = colunas.values
y = dataset['Weekly_Sales'].values
y = y.reshape(-1, 1)

y_train, y_test = train_test_split( y, test_size=0.8)
sc = StandardScaler()
sc.fit(y_train)
sc.fit(y_test)
y_treino = sc.transform(y_train)
y_test = sc.transform(y_test)

# print('lojas: ', stores['Store'].unique())
# print('Tipos: ', stores['Type'].unique())

# grouped = stores.groupby( 'Type')
# print( grouped.describe()['Size'].round(2))

# data = pd.concat([dataset['Type'], train], axis=1),
# df = data[0]
# figg = px.scatter_matrix(df, dimensions=['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday'],
#     color='Type')
# # figg.show()


# fig = px.box( df, x='Type', y='Weekly_Sales', points='all', color='Type',)
# # fig.show()

clf_DT = DecisionTreeClassifier()
lab = preprocessing.LabelEncoder()
new_y1 = lab.fit_transform(y_train)
new_y1

clf_DT.fit(y_treino, new_y1)
DT_pred = clf_DT.predict(y_test)

sns.heatmap(confusion_matrix(y_test, DT_pred), cmap='OrRd', annot=True, fmt='2.0f')
plt.title('Decision Tree')
plt.ylabel('P R E V I S T O')
plt.xlabel('R E A L')
plt.show()

# Acuracidade
print("ACC (DT) :%.2f" %(accuracy_score(y_test,DT_pred))) 

#Revocação
print("Recall (DT) :%.2f" %(recall_score(y_test,DT_pred))) 

#Precisão
print("Precision (DT) :%.2f" %(precision_score(y_test,DT_pred))) 

#F1-score
print("F1-score (DT) :%.2f" %(f1_score(y_test,DT_pred))) 