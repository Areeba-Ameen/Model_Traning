from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
import pandas as pd


#svc 
vt = pd.read_csv("iris.csv")

x = vt.drop(['variety'],axis=1)
y = vt['variety']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)

svc = SVC(C=1,kernel = 'rbf', tol = 0.001)
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

print(f"accuracy score {accuracy_score(y_pred,y_test)}")

#neighbor 
vt = pd.read_csv("iris.csv")

x = vt.drop(['variety'],axis=1)
y = vt['variety']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)

ngb = KNeighborsClassifier()
ngb.fit(x_train,y_train)

y_pred = ngb.predict(x_test)

print(f"accuracy score {accuracy_score(y_pred,y_test)}")



#linear model 
vt = pd.read_csv("iris.csv")

x = vt[['sepal.width','petal.length','petal.width']]
y = vt['sepal.length']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)

llm = LinearRegression()
llm.fit(x_train,y_train)

y_pred = llm.predict(x_test)

#print(f"accuracy score {accuracy_score(y_pred,y_test)}")
print(f"mean squared error, {mean_squared_error(y_test,y_pred)}")
print(f"r2 score {r2_score(y_pred,y_test)}")