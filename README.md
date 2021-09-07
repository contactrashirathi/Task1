# Task1
Prediction using supervised ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url ="http://bit.ly/w-data"
dataset = pd.read_csv(url)
print("Data imported successfully")
 
dataset.shape
dataset.head(5)
 
dataset.describe( )
 
dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()
 
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]
 
Splitting Dataset to train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


Training
from sklearn.linear_model import LinearRegression  
model=LinearRegression()
model.fit(X,Y)

Plotting regression line
line = model.coef_*X+model.intercept_

plt.scatter(X, Y)
plt.plot(X, line);
plt.show()

Predicting the scores
y_pred=model.predict(x_test)

from sklearn.metrics import r2_score
r2score = r2_score(y_test,y_pred)
print("R2Score",r2score*100)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df

x=9.25
score=[[x]]
PredictedmodelResult = model.predict(score)
print(PredictedmodelResult)

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
