import lib.neuralNetwork as neuralNetwork
import numpy as np

if __name__ == "__main__":
    nw1 = neuralNetwork.NeuralNetwork(2, 4, [5, 10], [False, True], 4)

    p_this = np.random.randn(4)
    print(nw1.predict(p_this))