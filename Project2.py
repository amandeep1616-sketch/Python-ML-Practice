import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
df=pd.read_csv('ml_practice_data.csv')
df=df.dropna(subset=['SleepHours'])
print(df.isnull().sum())
X=df[['HoursStudied','PreviousScore','AttendancePct','SleepHours','ExtraClasses']]
y=df['FinalScore']


#train model
model=LinearRegression()
model.fit(X,y)
predicted=model.predict(X)

#Valid regression metrice
mae=mean_absolute_error(y,predicted)
mse=mean_squared_error(y,predicted)
rmse=np.sqrt(mse)
r2score=r2_score(y,predicted)

#show
print("MAE:",round(mae,2))
print("MSE:",round(mse,2))
print("RMSE:",round(rmse,2))
print("R2 Score:",round(r2score,4)) #closer to 1= better

#histogram
plt.figure(figsize=(10,6))
plt.hist(df['FinalScore'], bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Final Score')
plt.xlabel('Final Score')
plt.ylabel('No. of Students')
plt.grid(True)
plt.show()

#scatter + regression line
plot_df = pd.DataFrame({
    'HoursStudied':X['HoursStudied'],
    'Predicted':predicted,
}).sort_values('HoursStudied')
plt.figure(figsize=(10,6))
plt.scatter(X['HoursStudied'],y, color='blue', label='Actual scores')
plt.plot(plot_df['HoursStudied'], plot_df['Predicted'], color='red', label='Predicted scores')
plt.title('Model prediction Vs Actual Scores')
plt.xlabel('Study Hours')
plt.ylabel('Final Output')
plt.grid(True)
plt.show()


