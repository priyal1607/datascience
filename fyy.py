from numpy import blackman
import pandas as pd
from matplotlib import pyplot as pt
import seaborn as sns
from seaborn.palettes import color_palette
data=pd.read_csv ('data.csv')
print(data.head())
y=data[['Duration']]
x=data[['Maxpulse']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train.head())
print(x_train.describe())
print(x_test.head())
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
print(lr.fit(x_train,y_train))
print(lr.predict(x_test))
y_pred=lr.predict(x_test)
print(y_test.head())
print(y_pred[0:5])
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))
from matplotlib import pyplot as plt
import seaborn as sns
sns.scatterplot(x='Duration',y='Maxpulse',data=data)
plt.colorbar
plt.show()