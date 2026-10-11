import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df=pd.read_csv('ml_practice_2col.csv')

X = df[['HoursStudied']]
y = df['Score']

model = LinearRegression()
model.fit(X, y)

predicted = model.predict(X)

#evaluate
mae = mean_absolute_error(y, predicted)
mse = mean_squared_error(y, predicted)
rmse=np.sqrt(mse)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

predict=float(input("Enter hour:"))
new_pred=model.predict([[predict]])
print(f"Predicted Score:{new_pred}")