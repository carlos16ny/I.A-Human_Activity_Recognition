#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix as plot
import matplotlib.pyplot as plt
import seaborn as sns

from Data import Data

data = Data()

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), 
                    max_iter=500, alpha=0.0001,
                    solver='sgd', verbose=10, 
                    random_state=21, tol=0.000000001)

clf.fit(data.X_train, data.Y_train)
y_pred = clf.predict(data.X_test)

d = plot(
    clf, 
    data.X_test, 
    data.Y_test, 
    display_labels = [x.split(" ")[0] for x in data.classes], 
    cmap = plt.cm.Blues,
    normalize = 'true'
).ax_.set_title("Matriz de Confusão Normalizada - MLP")

ac = accuracy_score(data.Y_test, y_pred)
cm = confusion_matrix(data.Y_test, y_pred)

print("Acurácia MLP : {:.2f}".format(ac * 100))

# sns.heatmap(cm, center = True)
plt.show()



