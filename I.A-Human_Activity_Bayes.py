#!/usr/bin/env python
# coding: utf-8
from Data import Data

from sklearn import datasets  
import matplotlib.pyplot as plt  
import pandas as pd  

# Bibliotecas para amostragem e Modelo Bayesiano
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix as plot

data = Data()

nb = GaussianNB()  
nb.fit(data.X_train, data.Y_train)  
y_predict = nb.predict(data.X_test)  

ac = accuracy_score(data.Y_test, y_predict)

d = plot(
    nb, 
    data.X_test, 
    data.Y_test, 
    display_labels = [x.split(" ")[0] for x in data.classes], 
    cmap = plt.cm.Blues,
    normalize = 'true'
).ax_.set_title("Matriz de Confusão Normalizada - Bayes")

print("Precisão NB: {:.2f}".format(ac * 100))
plt.show()


