#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando algumas funções para este código
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from Data import Data

data = Data()

sc = StandardScaler()
X_train = sc.fit_transform(data.X_train)
X_test = sc.fit_transform(data.X_test)

classifier = KNeighborsClassifier(
    n_neighbors = 17,
    weights = 'distance',
    p = 1
)
classifier.fit(X_train, data.Y_train)
y_predict = classifier.predict(X_test)

cm = confusion_matrix(data.Y_test, y_predict)
ac = accuracy_score(data.Y_test, y_predict)
print("Acurácia SVM : {:.2f}".format(ac * 100))

df = pd.DataFrame(cm, columns = [x.split(" ")[0] for x in data.classes], index = data.classes)
print(df)