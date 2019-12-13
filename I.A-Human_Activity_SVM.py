#!/usr/bin/env python
# coding: utf-8
from sklearn import svm
from Data import Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix as plot


import numpy as np

data = Data()

clf = svm.SVC(gamma = 0.001, C = 100).fit(data.X_train, data.Y_train)

#predict
y_pred = clf.predict(data.X_test)
cm = confusion_matrix(data.Y_test, y_pred)
ac = accuracy_score(data.Y_test, y_pred)

print("Acurácia SVM : {:.2f}".format(ac * 100))

g = plot(
    clf, 
    data.X_test, 
    data.Y_test, 
    display_labels = [x.split(" ")[0] for x in data.classes], 
    cmap = plt.cm.Blues,
    normalize = 'true'
)

g.ax_.set_title("Matriz de Confusão Normalizada - SVM")
plt.show()
