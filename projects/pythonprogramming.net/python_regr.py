import quandl
import sklearn
import pandas
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')
#style.use('fivethirtyeight')
#style.use('bmh')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
df = df[['Adj. Close','HL_PCT', 'PCT_change','Adj. Volume']]
df.head()
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.head()
df.tail()

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

#clf = LinearRegression()
#clf.fit(X_train, y_train)

#with open('linearregression.pickle','wb') as f:
#    pickle.dump(clf, f)
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)


clf.score(X_test, y_test)
accuracy = clf.score(X_test, y_test)

print(accuracy)

clf = svm.SVR()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

forecast_set = clf.predict(X_lately)
forecast_set
print(forecast_set)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


#y = mx + b is linear equation
# we know x, need to estimate value of m and b
# m = [x'*y' - (x*y)']/[(x'^2 - x^2']

from statistics import mean
import numpy as np
xs = [1,2,3,4,5,6]
ys = [5,4,6,5,6,7]

plt.plot(xs, ys)
plt.scatter(xs, ys)

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)
def best_fit_slope(xs, ys):
    m = ((mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b
m,b = best_fit_slope(xs, ys)
regression_line = [(m*x)+b for x in xs]

#style.use('bmh')

plt.scatter(xs,ys)
plt.plot(xs, regression_line)

predict_x = 8
predict_y = (m*predict_x)+b
plt.scatter(predict_x, predict_y, color='g')
plt.show()

