'''import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from random import randrange
 
plt.style.use('ggplot')
x_data = []
y_data = []
 
 
figure = pyplot.figure()
line, = pyplot.plot_date(x_data, y_data, '-')
 
def grafica3(frame):
    #asfasdfasfaf
    #asfasfasf
    #temperatura 
    x_data.append(datetime.now())
    y_data.append(randrange(0, 100))
    line.set_data(x_data, y_data)
    figure.gca().relim()
    figure.gca().autoscale_view()
    return line,
 
#animacion3 = FuncAnimation(figure, grafico, interval=5000)
animacion3 = FuncAnimation(figure, grafica3, interval=100)
pyplot.show()'''
'''
import numpy as np
import time
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.cos(x)

plt.ion()

figure, ax = plt.subplots(figsize=(8,6))
line1, = ax.plot(x, y)

plt.title("Dynamic Plot of sinx",fontsize=25)

plt.xlabel("X",fontsize=18)
plt.ylabel("sinX",fontsize=18)

for p in range(100):
    updated_y = np.cos(x-0.05*p)
    
    line1.set_xdata(x)
    line1.set_ydata(updated_y)
    
    figure.canvas.draw()
    
    figure.canvas.flush_events()
    time.sleep(0.1)'''

import numpy as np
import matplotlib.pyplot as plt
x=0
for i in range(100):
    x=x+0.04
    y = np.sin(x)
    plt.scatter(x, y)
    plt.title("Real Time plot")
    plt.xlabel("x")
    plt.ylabel("sinx")
    plt.pause(0.05)

plt.show()