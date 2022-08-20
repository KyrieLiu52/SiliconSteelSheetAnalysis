import pandas as pd
import csv

data2 = pd.read_csv("./90C-1D.csv")
# print(type(data[1][2]) == type("abc"))
data = pd.read_csv("./90C-1D_no_head.csv", header=None)
# print(data2[0][2] > 2650)
# columns = data2.columns
# print(data2)
# data2 = data2.loc[:, [columns[1], columns[0]]]
# print(data2)

columns = data.columns
data = data[(data[columns[0]] >= 2450) & (data[columns[0]] < 2500)]
data.columns = ['Wavenumber','Absorbance']
data.to_csv("./test.csv", index=0)

