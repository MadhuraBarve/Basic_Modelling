## Importing the libraries
from statistics import mean
import numpy as np
import pandas as pd
import quandl as qd
import math, datetime, random
from sklearn import preprocessing, svm
# from sklearn.model_selection import 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import pickle



def best_fit_slope_and_intercept(xs,ys):
	m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs*xs)))
	b = mean(ys) - m * mean(xs)
	return m,b

def squared_error(ys_orig,ys_line):
	return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_reqr = squared_error(ys_orig,ys_line)
	squared_error_y_mean = squared_error(ys_orig,y_mean_line)
	return 1 - (squared_error_reqr / squared_error_y_mean)

df = qd.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['Hl_PCT']= (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change']= (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','Hl_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X= np.array(df.drop(['label','Adj. Close'],1))
X = preprocessing.scale(X)
X= X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y= np.array(df['label'])
y = np.array(df['label'])


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size= 0.2)

m,b = best_fit_slope_and_intercept(X_train,y_train)
#print(m,b)
regression_line = [(m*x)+b for x in xs]

predict_x = X_lately
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,s=100,color = 'g')
plt.plot(xs,regression_line)
plt.show()


