"""from sklearn.preprocessing import LabelEncoder
import pandas as pd
df=pd.read_csv("pandas_practice_data.csv")
df_label=df.copy()
le=LabelEncoder()
df_label['Region_Encoded']=le.fit_transform(df_label['Region'])
df_label['Product_Encoded']=le.fit_transform(df_label['Product'])

print('\n Label encoded data')
print(df_label[['OrderID','Region','Region_Encoded','Product','Product_Encoded']])
df_encoded=pd.get_dummies(df, columns=['Region', 'Product'])
print(df_encoded)"""



from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
df=pd.read_csv("pandas_practice_data.csv")
X= df[['Quantity']]
y=df[['Quantity']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data")
print(X_train)
print('Test data')
print(X_test)
scalar = StandardScaler()
scalar = scalar.fit_transform(df[['Quantity', 'Revenue']])

print(pd.DataFrame(scalar, columns=['Quantity', 'Revenue']))