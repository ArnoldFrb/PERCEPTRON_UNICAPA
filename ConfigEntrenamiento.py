import random as rn
import numpy as np
from Neurona import *

class ConfigEntrenamiento:

    #CONSTRUCTOR
    def __init__(self, ENTRENAMIENTO):

        self.MATRIZ_ENTRADA = np.loadtxt('DATA/' + ENTRENAMIENTO + '/' + 'ENTRADAS.TXT')
        self.MATRIZ_SALIDA = np.loadtxt('DATA/' + ENTRENAMIENTO + '/' + 'SALIDAS.TXT')
        self.ENTRENAMIENTO = ENTRENAMIENTO

    #METODO PARA GENERAR PESOS
    def GENERAR_PESOS(self):
        MATRIZ = []
        for N in range(self.MATRIZ_SALIDA.ndim):
            FILA = []
            for M in range(len(self.MATRIZ_ENTRADA[0])):
                FILA.append(round(rn.uniform(-1, 1), 2))
            MATRIZ.append(FILA)
        return MATRIZ

    #METODO PARA GENERAR UMBRALES
    def GENERAR_UMBRALES(self):
        FILA = []
        for N in range(self.MATRIZ_SALIDA.ndim):
            FILA.append(round(rn.uniform(-1, 1), 2))
        return FILA

    #EJECUTAR NEURONA
    def ENTRENAR_NEURONA(self, RATA_APRENDIZAJE, ERROR_MAXIMO, NUMERO_ITERACIONES, FUNCION_SALIDA):
        neuro = Neurona(
            self.MATRIZ_ENTRADA, self.MATRIZ_SALIDA, self.GENERAR_PESOS(), self.GENERAR_UMBRALES(),
            RATA_APRENDIZAJE, ERROR_MAXIMO, NUMERO_ITERACIONES, FUNCION_SALIDA
            )
        neuro.ENTRENAR(self.ENTRENAMIENTO)