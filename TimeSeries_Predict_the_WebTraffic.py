# AUTHOR : HAMORA HADI

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

#read data to df
n_rows=int(input())
df=pd.DataFrame(index=pd.date_range(start='October 1st 2012', periods=n_rows+30, freq='D'), columns=['TS_INPUT'])
for i in range(n_rows):
    df.iloc[i] = input()

df['DOW']=df.index.weekday
df['month']=df.index.month
df['dayofmonth']=df.index.day

for i in range(20):
    df['INPUT_PREV'+str(i)]=df.TS_INPUT.shift(i+1).fillna(0)

X = df.iloc[:-30,1:]
y = df.iloc[:-30,0]

reg=Ridge()
for i in range(30):
    for i in range(20):
        df['INPUT_PREV'+str(i)]=df.TS_INPUT.shift(i+1).fillna(0)
    test = df.iloc[-30+i,1:]
    reg.fit(X,y)
    preds=reg.predict(test.values.reshape(1, -1))

preds=df.iloc[-30:,0] 

for element in preds:
    print(element)