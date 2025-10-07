# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
## Program:

## Developed by: SHARON CLARA A
## RegisterNumber:  212224040310
```

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```

## Output:

## Data Head:

<img width="390" height="265" alt="318778934-b2f6f2eb-1e0c-4fbb-8784-4a8bd706c979" src="https://github.com/user-attachments/assets/87c05eb7-5707-41f0-a39a-62c6f937fa09" />

## Data Info:

<img width="603" height="237" alt="318779092-7c13b486-2ad5-4e1f-82f6-48d7f77e2649" src="https://github.com/user-attachments/assets/642810f0-ddc7-4275-96cb-8a359168d06d" />

## isnull() sum():

<img width="201" height="88" alt="318779178-3a21fac0-df89-4aaf-827f-bc00aa3f0286" src="https://github.com/user-attachments/assets/53730c38-8ed8-47d8-bd8c-59b65449d273" />

## Data Head for salary:

<img width="323" height="234" alt="318779270-0a79abfa-f32d-4394-a73d-47161eaeec30" src="https://github.com/user-attachments/assets/9e9190f0-6773-428d-ad77-58b4b767e2ba" />

## Mean Squared Error :

<img width="239" height="38" alt="318779364-3c7acf12-adb7-4a3f-807e-cb49ad260032" src="https://github.com/user-attachments/assets/f9e3461c-f179-4a36-907b-71f4c0eb0d8b" />

## r2 Value:

<img width="1065" height="41" alt="318779545-e6f5cab9-dab9-4c69-bb0e-6fa0abee1da0" src="https://github.com/user-attachments/assets/3072b798-ed00-48b4-b199-de96d0ee7cbf" />

## Data prediction :

<img width="311" height="38" alt="318779650-92b5c1d6-e495-4eaa-9a9a-8eb3a37ae0bc" src="https://github.com/user-attachments/assets/58e48283-f02b-429a-9014-8759e7fcbd37" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
