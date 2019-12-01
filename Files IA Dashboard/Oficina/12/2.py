# -*- coding: utf-8 -*-
"""
Created on Sar Nov  30 14:07 2019

Modelo para predecir el tiempo de espera por oficina y número de Usuarios

Para Tiempo se observa comportamienti peridico se usa Modelo Aurogresivo
Autoregressive integrated moving average--
Modelo autorregresivo integrado de media móvil

------------------------------

Para la predicción del nunuero de clienes 
USAR
Rgresión lineal o Regresor de Soporte Vecotirla


Modelo, Regresor de soporte Vectorial...

@author: DVJJ
"""

import pandas as pd 
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#Cargo Datos
#datos = pd.read_excel("Datos_luz.xlsx") 
datos = pd.read_csv("datos.csv", sep = ";", dtype=None)



import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv("datos.csv", sep = ";", dtype=None, encoding = 'latin') #todos los Datos


data = data.dropna(axis=1, how="all")  #borra columnas vacias
data = data.dropna(axis=0, how="all")  #borra filas vacias

# CodOficina;NClientes;Fecha;TiempoESP;ConsecutivoP;Minutos

data_tiempo = data.drop(['CodOficina','NClientes','Fecha','TiempoESP'],axis=1) #Quitar lo que no se usa

data_clientes = data.drop(['CodOficina','Fecha','TiempoESP','ConsecutivoP'],axis=1) #Quitar lo que no se usa



#hago Vector Tiempo y ejecución presupuestal.
a = datos["AñoMes"]
b = datos["real"]

X=np.array(a)
X = np.resize(X,[10,1])
y=np.array(b)
#X = np.sort(np.random.rand(10, 1), axis=0)
#X= np.array(a)
#X= np.arange(1,84)


#t=t.astype(float)
#real=real.astype(float)




# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e9, gamma=2.5)
#svr_lin = SVR(kernel='linear', C=1e5)
#svr_poly = SVR(kernel='poly', C=1e5, degree=3)



y_rbf = svr_rbf.fit(X, y).predict(X)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

# #############################################################################
# Look at the results-
lw = 2
plt.scatter(X, y, color='darkorange', label='Datos')
plt.plot(X, y_rbf, color='navy', lw=lw, label='Con Kernel RBF')
#plt.plot(X, y_lin, color='c', lw=lw, label='Con Kernel Lineal')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Con Kernel Polinomial')
plt.xlabel('datos')
plt.ylabel('Objetivos')
plt.title('Predicción de gastos de Electricidad según Support Vector Regression')
plt.legend()
plt.show()