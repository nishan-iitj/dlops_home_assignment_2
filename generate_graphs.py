import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions as dictionaries for easy access
activation_functions = {
    "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "ReLU": lambda x: np.maximum(0, x),
    "Leaky ReLU": lambda x, alpha=0.01: np.where(x > 0, x, x * alpha),
    "Tanh": lambda x: np.tanh(x)
}

def plot_activation_function(x, name, subplot=None):
    """
    Plots an activation function given an array of x values and the name of the function.
    :param x: Array of x values to be used for plotting.
    :param name: Name of the activation function (for title and label).
    :param subplot: matplotlib subplot object to plot the function on. If None, creates a new plot.
    """
    y = activation_functions[name](x)

    if subplot is None:
        plt.figure(figsize=(5, 4))
        plt.plot(x, y, label=name)
        plt.title(f"{name} Activation Function")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        subplot.plot(x, y, label=name)
        subplot.set_title(f"{name} Activation Function")
        subplot.set_xlabel("Input")
        subplot.set_ylabel("Output")
        subplot.grid(True)
        subplot.legend()


if __name__=="__main__":
    # Generate a range of x values
    random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    plot_activation_function(random_values, "ReLU", subplot=axs[0, 1])
    plot_activation_function(random_values, "Leaky ReLU", subplot=axs[1, 0])
    plot_activation_function(random_values, "Tanh", subplot=axs[1, 1])

    plt.tight_layout()
    plt.show()
