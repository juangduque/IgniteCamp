# -*- coding: utf-8 -*-
"""
Created on Sar Nov  30 09:11 2019

Modelo para predecir el  y número de Usuarios por oficina


------------------------------

Para la predicción del nunuero de clienes 
Se Usa rgresión  de Soporte Vecotrial

@author: Xamay
"""



import pandas as pd 
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#Cargo Datos

datos = pd.read_csv("datos.csv", sep = ";", dtype=None)

#hago Vector Tiempo y # de Clienyes
a = datos['ConsecutivoP']
b = datos['NClientes']


# Vectorespara Entrenamiento
X=np.array(a)
X = np.resize(X,[255,1])
y=np.array(b)
#X = np.sort(np.random.rand(10, 1), axis=0)
#X= np.array(a)
#X= np.arange(1,84)


#t=t.astype(float)
#real=real.astype(float)
# Graficos de tendencia...
#rcParams['figure.figsize'] = 15, 6
#decomposition = sm.tsa.seasonal_decompose(y, model='additive')
#fig = decomposition.plot()
#plt.show()

###  ARIMA, which stands for Autoregressive Integrated Moving Average.



# ###############
# Enttrenar el regresor de soporte vecotial
# Se hacen pruebas con Kernel Lienal, Polinomica y RBF Radial basis function
svr_rbf = SVR(kernel='rbf', C=1e4, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e2)
#svr_poly = SVR(kernel='poly', C=1e2, degree=2)
#Polinómica no converge con datos del banco, es lo esperado

# Luego de entrenado se hace la predicción
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

# #########################
# Grafica Los Resultados
lw = 2
plt.scatter(X, y, color='darkorange', label='Datos')
plt.plot(X, y_rbf, color='navy', lw=lw, label='Con Kernel RBF')
plt.plot(X, y_lin, color='c', lw=lw, label='Con Kernel Lineal')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Con Kernel Polinomial')
plt.xlabel('Tiempo (Fecha)')
plt.ylabel('Número de clientes por Oficina')
plt.title('Predicción de Número de Usuarios por Oficina Support Vector Regression')
plt.legend()
plt.show()

#SSD = sum((test['Numero_Horas']-pr)**2)
#print (SSD)

#RSE = np.sqrt(SSD/(len(test)-2-1))
#print (RSE)

#Media= np.mean(test['Numero_Horas'])
#error = RSE/Media


#print("porcentaje de Error", error*100, "%")


#pred= pd.series
#predictions().values

# plot
#pyplot.plot(test)
#pyplot.plot(predictions, color='red')
#pyplot.show()
