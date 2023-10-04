import pandas as pd
df1=pd.read_csv('dataset/traditional_spambots1.csv', encoding='latin-1')

df1=df1.head(20000)
df1['class']=0
print(df1.head(10))
print(df1.shape)
print(df1.columns)
df1=df1[['text','class']]
print(df1.head(10))
df1.to_csv("spam1.csv",index=False)

df1=pd.read_csv('traditional_spambots1.csv', encoding='latin-1')

