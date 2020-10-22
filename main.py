import tensorflow
import keras
import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
best = 0
data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

# The use of the number 1 removes the column
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

for _ in range(3500):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    #print(acc)

    if acc > best:
        best = acc
        with open("StudentModel.pickle", "wb") as f:
            pickle.dump(linear, f)

print("The best was: ", best)

pickle_in = open("StudentModel.pickle", "rb")

linear = pickle.load(pickle_in)

print("CO : ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

print(data.head())

for i in range(len(predictions)):
    print(round(predictions[i], 1), x_test[i], y_test[i])

p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()

