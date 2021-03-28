import numpy as np
import pylab as plt
import pandas as pd
from IPython.display import display

class TestNeurona:

    #CONSTRUCTOR
    def __init__(self, MATRIZ_ENTRADA = [[1,0,1], [0,1,1], [1,1,0]], MATRIZ_SALIDA = [[1,0], [0,1], [1,1]], MATRIZ_PESOS = [[0.1,-0.5,-0.9], [0.6,0.2,-0.3]], MATRIZ_UMBRALES = [0.5, -0.8], FUNCION_SALIDA = 1):
        #MATRICES
        self.MATRIZ_ENTRADA = MATRIZ_ENTRADA
        self.MATRIZ_SALIDA = MATRIZ_SALIDA
        self.MATRIZ_PESOS = MATRIZ_PESOS
        self.MATRIZ_UMBRALES = MATRIZ_UMBRALES
        self.FUNCION_SALIDA = FUNCION_SALIDA

    #METODO PARA SIMULAR LA NEURONA
    def SIMULAR(self):

        print()
        print("---CONFIGURACION---")
        print()

        print('MATRIZ DE DATOS')
        MATRIZ = []
        for I in range(len(self.MATRIZ_ENTRADA)):
            FILA = []
            for J in range(self.MATRIZ_ENTRADA.ndim):
                FILA.append(self.MATRIZ_ENTRADA[I,J])

            for J in range(self.MATRIZ_SALIDA.ndim):
                FILA.append(self.MATRIZ_SALIDA[I] if self.MATRIZ_SALIDA.ndim==1 else (self.MATRIZ_SALIDA[I,J]))

            MATRIZ.append(FILA)

        COL = []
        for J in range(self.MATRIZ_ENTRADA.ndim):
            COL.append("X"+str(J))

        for J in range(self.MATRIZ_SALIDA.ndim):
            COL.append("Y"+str(J))

        df2 = pd.DataFrame(MATRIZ, columns=COL)
        display(df2)

        print()
        print('PESOS OPTIMOS')
        display(pd.DataFrame(self.MATRIZ_PESOS))

        print()
        print('UMBRALES OPTIMOS')
        display(pd.DataFrame(self.MATRIZ_UMBRALES))

        plt.xlabel('PATRONES')
        plt.ylabel('SALIDAS YD Y YR')
        plt.title('YD vs YR')
        plt.grid()
        
        print("---------------------------")
        print("---------------------------")

        print()
        print("---SIMULACION---")
        print()

        #CICLO ENCARGADO DE PRESENTAR LOS PATRONES
        for I in range(len(self.MATRIZ_ENTRADA)):
            PATRON_PRESENTADO = (self.MATRIZ_ENTRADA[I,:])
            SALIDA = self.FUNCION_SALIDAS(self.FUNCION_SOMA(PATRON_PRESENTADO))

            plt.plot(I+1, self.MATRIZ_SALIDA[I] if (self.MATRIZ_SALIDA.ndim == 1) else sum(self.MATRIZ_SALIDA[I]), 'rd',
             I+1, sum(SALIDA) if (len(SALIDA) !=1) else SALIDA, 'b.')
        
        plt.legend(('YD', 'YR'), prop = {'size': 10},)

    #METODO PARA OBTENER LA FUNCION SOMA
    def FUNCION_SOMA(self, PATRON):
        SL = []         #SALIDA DE LA FUNCION SOMA
        for N in range(len(self.MATRIZ_PESOS)):
            SLD = 0     #SUMATORIA DE LA FUNCION SOMA
            for M in range(self.MATRIZ_PESOS.ndim):
                SLD += (PATRON[M] * self.MATRIZ_PESOS[N][M])
            SL.append(SLD - self.MATRIZ_UMBRALES[N])
        return SL

    def FUNCION_SALIDAS(self, SALIDA_SOMA):
        switcher = {
            1: self.FUNCION_ESCALON(SALIDA_SOMA),
            2: self.FUNCION_LINEAL(SALIDA_SOMA),
            3: self.FUNCION_SIGMOIDE(SALIDA_SOMA)
        }
        return switcher.get(self.FUNCION_SALIDA, "ERROR")

    #METODO PARA OBTENER LA FUNCION ESCALON
    def FUNCION_ESCALON(self, SALIDA_SOMA):
        YR = []
        for N in range(len(SALIDA_SOMA)):
            YR.append(1 if SALIDA_SOMA[N]>=0 else 0)
        return YR

    #METODO PARA OBTENER LA FUNCION SIGMOIDE
    def FUNCION_SIGMOIDE(self, SALIDA_SOMA):
        YR = []
        for N in range(len(SALIDA_SOMA)):
            YR.append(1 / (1 + np.exp(-SALIDA_SOMA[N])))
        return YR

    #METODO PARA OBTENER LA FUNCION LINEAL
    def FUNCION_LINEAL(self, SALIDA_SOMA):
        YR = SALIDA_SOMA
        return YR