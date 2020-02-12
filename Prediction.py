# Import dependencies
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
include = ['Age', 'Sex', 'Embarked', 'Survived']  # Only four features
df1 = df[include]

categorical = []
for col, col_type in df1.dtypes.iteritems():
    if col_type == '0':
        categorical.append(col)  # appending non-numeric values to a list categorical
    else:
        df1[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df1, columns=categorical, dummy_na=True)  # One-Hot Encoding

