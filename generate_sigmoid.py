from generate_graphs import plot_activation_function
import numpy as np

if __name__=="__main__":
    random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
    plot_activation_function(random_values, "Sigmoid")