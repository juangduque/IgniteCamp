# -*- coding: utf-8 -*-
"""
Created on Sar Nov  30 08:07 2019

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
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv("datos.csv", sep = ";", dtype=None, encoding = 'latin') #todos los Datos


data = data.dropna(axis=1, how="all")  #borra columnas vacias
data = data.dropna(axis=0, how="all")  #borra filas vacias

# CodOficina;NClientes;Fecha;TiempoESP;ConsecutivoP;Minutos

data_tiempo = data.drop(['CodOficina','NClientes','Fecha','TiempoESP'],axis=1) #Quitar lo que no se usa

data_clientes = data.drop(['CodOficina','Fecha','TiempoESP','ConsecutivoP'],axis=1) #Quitar lo que no se usa


data['Fecha']=pd.to_datetime(data['Fecha']) #se transforma Date a formato Tiempo
#data['Date']=data.set_index(data['Date'])

data['Fecha'].min(),data['Fecha'].max() # Ser ordenan los datos por su Fecha

data= data.set_index('Fecha') # La fecha se hace el inice del dataFrame


# Calcularemos el # de Horas extras mensual...
y = data['Costo_Valor'].resample('MS').sum()
# Calcularemos el # de Horas extras Diario...
y1 = data['Costo_Valor'].resample('MS').mean() # esta es la media

#Hacer Graficas estadísitcas.

#y.plot(figsize=(15,6))
#y.rolling(12).mean().plot(label = 'Media Anual')
#y.rolling(12).std().plot(label = 'desviación STD Anual')
#plt.legend()
#plt.show()

from pylab import rcParams
import statsmodels.api as sm

# Graficos de tendencia...
#rcParams['figure.figsize'] = 15, 6
#decomposition = sm.tsa.seasonal_decompose(y, model='additive')
#fig = decomposition.plot()
#plt.show()

###  ARIMA, which stands for Autoregressive Integrated Moving Average.



from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf

autocorrelation_plot(y) # Graficar autocorrelación de horas extras...
plot_acf(y)
pyplot.show()


from statsmodels.tsa.arima_model import ARIMA #Importar Modelo Arima



size = int(len(y) * 0.8)
train, test = y[0:size], y[size:len(y)] # Separar Test y Train
#X1= test["Numero_Horas"].values

#model = ARIMA(train, order=(3,0,1)) # Este da error de 16%
model = ARIMA(train, order=(3,0,1))  # Definir parámetros del modelo

model_fit = model.fit()  # Entrenar Modelo

#y_predict= model_fit.predict(start=28,end=35)
y_predict= model_fit.predict(start=28,end=35)   # Predecir para probar


# Graficar Predicción contra datos Reales...
plt.plot(test, label = 'Costo Horas Reales Cliente/mes')
plt.plot(y_predict, label = 'Costo Horas Predicción/mes')
plt.legend()

## Calculo del porcentaje de Error del modelo

SSD = sum((test-y_predict)**2)
print (SSD)

RSE = np.sqrt(SSD/(len(test)-1-1))
print (RSE)

Media= np.mean(test)
error = RSE/Media


print("porcentaje de Error", error*100, "%") # Imprimir Error


