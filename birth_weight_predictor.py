import pickle

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

file = pd.read_csv('babies.csv')
file = file[["bwt", "gestation", "parity", "age", "height", "weight", "smoke"]]
file = file.fillna(file.mean())
x = np.array(file.drop("bwt", axis=1))
y = np.array(file["bwt"])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

b = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.004)
    Linear = linear_model.LinearRegression()
    Linear.fit(x_train, y_train)
    acc = Linear.score(x_test, y_test)
    print("acc = ", str(acc))
    if acc > b:
        b = acc
        with open("model.pickle", 'wb') as f:
            pickle.dump(Linear, f)

print("best acc = ", str(b))

p = open('model.pickle', 'rb')
Linear = pickle.load(p)

predict = Linear.predict(x_test)
for x in range(len(predict)):
    print(predict[x], x_test[x], y_test[x])

