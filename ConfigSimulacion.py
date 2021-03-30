import random as rn
import numpy as np
from TestNeurona import *

class ConfigSimulacion:

    #CONSTRUCTOR
    def __init__(self, ENTRENAMIENTO):

        self.MATRIZ_ENTRADA = np.loadtxt('DATA/' + ENTRENAMIENTO + '/' + 'ENTRADAS.TXT')
        self.MATRIZ_SALIDA = np.loadtxt('DATA/' + ENTRENAMIENTO + '/' + 'SALIDAS.TXT')
        self.FUNCION_SALIDA = int(np.loadtxt('DATA/' + ENTRENAMIENTO + '/' + 'CONFIG.TXT'))
        self.ENTRENAMIENTO = ENTRENAMIENTO
    
    #EJECUTAR SIMULACION
    def SIMULACION(self):
        PESOS_TEMPORAL = np.loadtxt("DATA/" + self.ENTRENAMIENTO + "/" + self.NOMBRE_SALIDAS() + "/PESOS.TXT")
        PESOS_OPTIMOS = np.array([PESOS_TEMPORAL]) if PESOS_TEMPORAL.ndim==1 else PESOS_TEMPORAL

        UMBRALES_TEMPORALES = np.loadtxt("DATA/" + self.ENTRENAMIENTO + "/" + self.NOMBRE_SALIDAS() + "/UMBRALES.TXT")
        UMBRALES_OPTIMOS = np.array([UMBRALES_TEMPORALES]) if UMBRALES_TEMPORALES.ndim==0 else UMBRALES_TEMPORALES

        simul = TestNeurona(self.MATRIZ_ENTRADA, self.MATRIZ_SALIDA, PESOS_OPTIMOS, UMBRALES_OPTIMOS, self.FUNCION_SALIDA)
        simul.SIMULAR()
    
    #NOMBRE DE LA FUNCION SALIDA
    def NOMBRE_SALIDAS(self):
        switcher = {
            1: 'ESCALON',
            2: 'LINEAL',
            3: 'SIGMOIDE'
        }
        return switcher.get(self.FUNCION_SALIDA, "ERROR")