import numpy as np
import os
import errno
import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from IPython.display import display

class Neurona:

    #CONSTRUCTOR
    def __init__(self, MATRIZ_ENTRADA = [[1,0,1], [0,1,1], [1,1,0]], MATRIZ_SALIDA = [[1,0], [0,1], [1,1]], MATRIZ_PESOS = [[0.1,-0.5,-0.9], [0.6,0.2,-0.3]], MATRIZ_UMBRALES = [0.5, -0.8], RATA_APRENDIZAJE = 1, ERROR_MAXIMO = 0.1, NUMERO_ITERACIONES = 0, FUNCION_SALIDA = 1):
        #MATRICES
        self.MATRIZ_ENTRADA = MATRIZ_ENTRADA
        self.MATRIZ_SALIDA = MATRIZ_SALIDA
        self.MATRIZ_PESOS = MATRIZ_PESOS
        self.MATRIZ_UMBRALES = MATRIZ_UMBRALES
        #VARIABLES DE CONFIGURACION
        self.RATA_APRENDIZAJE = RATA_APRENDIZAJE
        self.ERROR_MAXIMO = ERROR_MAXIMO
        self.NUMERO_ITERACIONES = NUMERO_ITERACIONES
        self.FUNCION_SALIDA = FUNCION_SALIDA
        self.Error = []
        self.Itera = []


    #METODO PARA ENTRENAR LA NEURONA
    def ENTRENAR(self, ENTRENAMIENTO):

        MATRIZ = []
        for I in range(len(self.MATRIZ_ENTRADA)):
            FILA = []
            for J in range(len(self.MATRIZ_ENTRADA[0])):
                FILA.append(self.MATRIZ_ENTRADA[I,J])

            for J in range(self.MATRIZ_SALIDA.ndim):
                FILA.append(self.MATRIZ_SALIDA[I] if self.MATRIZ_SALIDA.ndim==1 else (self.MATRIZ_SALIDA[I,J]))

            MATRIZ.append(FILA)

        COL = []
        for I in range(len(self.MATRIZ_ENTRADA[0])):
            COL.append("X"+str(I))

        for J in range(self.MATRIZ_SALIDA.ndim):
            COL.append("Y"+str(J))

        df2 = pd.DataFrame(MATRIZ, columns=COL)
        display(df2)
        print()
        
        plt.style.use('ggplot')
        plt.title('ENTRENAMIENTO - ERROR LA ITERACION')
        plt.xlabel('ITERACION')
        plt.ylabel('ERROR RMS')

        #CICLO PARA ITERACIONES
        ITERACION_INICIAL = 0
        while True:

            ERROR_PATRON = []

            #CICLO ENCARGADO DE PRESENTAR LOS PATRONES
            for I in range(len(self.MATRIZ_ENTRADA)):

                PATRON_PRESENTADO = (self.MATRIZ_ENTRADA[I,:])
                SALIDA_PATRON = np.array([self.MATRIZ_SALIDA[I]]) if self.MATRIZ_SALIDA.ndim==1 else (self.MATRIZ_SALIDA[I,:])
                
                ERROR_PATRON.append((np.abs(self.FUNCION_ERROR_LINEAL(SALIDA_PATRON, self.FUNCION_SALIDAS(self.FUNCION_SOMA(PATRON_PRESENTADO)))).sum()) / self.MATRIZ_SALIDA.ndim)

                self.ACTUALIZAR_PESOS(PATRON_PRESENTADO, self.FUNCION_ERROR_LINEAL(SALIDA_PATRON, self.FUNCION_SALIDAS(self.FUNCION_SOMA(PATRON_PRESENTADO))))
                self.ACTUALIZAR_UMBRALES(self.FUNCION_ERROR_LINEAL(SALIDA_PATRON, self.FUNCION_SALIDAS(self.FUNCION_SOMA(PATRON_PRESENTADO))))

            #METODO PARA OBTENER EL ERROR DE LA ITERACION
            ERROR_RMS = (np.sum(ERROR_PATRON)) / len(self.MATRIZ_ENTRADA)

            #plt.scatter(ITERACION_INICIAL, ERROR_RMS)
            #plt.pause(0.0005)
            
            self.Error.append(ERROR_RMS)
            self.Itera.append(ITERACION_INICIAL+1)

            ITERACION_INICIAL+=1

            #CONDICIONES DE PARADA
            if((ITERACION_INICIAL > self.NUMERO_ITERACIONES-1) or (ERROR_RMS <= self.ERROR_MAXIMO)):
                break
        
        if(ERROR_RMS <= self.ERROR_MAXIMO):

            self.GUADAR_PESOS_UMBRALES(ENTRENAMIENTO)

            print('SE HAN ENCONTRADO LOS PESOS Y UMBRALES OPTIMOS')
            print()
        
        plt.plot(self.Itera, self.Error, linewidth=1)
        plt.show()

        print("ERROR RMS: ", ERROR_RMS)
        print("NUMERO DE ITERACIONES REALIZADAS: ", ITERACION_INICIAL)
        print()

    #METODO PARA OBTENER LA FUNCION SOMA
    def FUNCION_SOMA(self, PATRON):
        SL = []         #SALIDA DE LA FUNCION SOMA
        for N in range(len(self.MATRIZ_PESOS)):
            SLD = 0     #SUMATORIA DE LA FUNCION SOMA
            for M in range(len(self.MATRIZ_PESOS[0])):
                SLD += (PATRON[M] * self.MATRIZ_PESOS[N][M])
            SL.append(SLD - self.MATRIZ_UMBRALES[N])
        return SL

    #METODO PARA OBTENER EL ERROR LINEAL
    def FUNCION_ERROR_LINEAL(self, SALIDA_PATRON, ESCALON):
        EL = []          #ERROR LINEAL
        for N in range(len(ESCALON)):
            EL.append(SALIDA_PATRON[N] - ESCALON[N])
        return EL
    
    #METODO PARA ACTUALIZAR PESOS
    def ACTUALIZAR_PESOS(self, PATRON_PRESENTADO, ERROR_LINEAL):
        for N in range(len(self.MATRIZ_PESOS)):
            for M in range(len(self.MATRIZ_PESOS[0])):
                self.MATRIZ_PESOS[N][M] += (self.RATA_APRENDIZAJE * ERROR_LINEAL[N] * PATRON_PRESENTADO[M])

    #METODO PARA ACTUALIZAR UMBRALES
    def ACTUALIZAR_UMBRALES(self, ERROR_LINEAL):
        for N in range(len(self.MATRIZ_UMBRALES)):
            self.MATRIZ_UMBRALES[N] += (self.RATA_APRENDIZAJE * ERROR_LINEAL[N] * 1)

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

    #NOMBRE DE LA FUNCION SALIDA
    def NOMBRE_SALIDAS(self):
        switcher = {
            1: 'ESCALON',
            2: 'LINEAL',
            3: 'SIGMOIDE'
        }
        return switcher.get(self.FUNCION_SALIDA, "ERROR")

    #GUADAR PESOS Y UMBRALES
    def GUADAR_PESOS_UMBRALES(self, ENTRENAMIENTO):
        
        try:
            os.mkdir('DATA/'+ ENTRENAMIENTO + "/" + self.NOMBRE_SALIDAS())
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        np.savetxt("DATA/" + ENTRENAMIENTO + "/CONFIG.TXT", [self.FUNCION_SALIDA])
        np.savetxt("DATA/" + ENTRENAMIENTO + "/" + self.NOMBRE_SALIDAS() + "/PESOS.TXT", self.MATRIZ_PESOS)
        np.savetxt("DATA/" + ENTRENAMIENTO + "/" + self.NOMBRE_SALIDAS() + "/UMBRALES.TXT", self.MATRIZ_UMBRALES)
