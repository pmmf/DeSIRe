import numpy as np
import matplotlib.pyplot as plt

K = 0.00025
X0 = 10*35*32
EPOCHS = 10

def main():
    linear = []
    exponential = []
    exponential2 = []
    exponential3 = []
    x_step = range(X0*2)
    for step in x_step:
        linear.append(min(1, step/X0))
        exponential.append(1 / (1 + np.exp(-K*(step-X0))))
        exponential2.append(1 / (1 + np.exp(-K*2*(step-X0))))
        exponential3.append(1 / (1 + np.exp(-K*0.5*(step-X0))))


    plt.figure()
    plt.plot(x_step, linear)
    plt.plot(x_step, exponential, 'red')
    plt.plot(x_step, exponential2, 'blue')
    plt.plot(x_step, exponential3, 'green')
    plt.vlines(x=X0, ymin=0, ymax=1, color='red')
    plt.xlabel('step')
    plt.ylabel('KL weight')
    plt.title('KL annealing')
    plt.show()

if __name__ == '__main__':
    main()
