import os
import numpy as np
from datetime import datetime
import hashlib
import colorama
import platform
import threading
import time

class NeuralNetwork:
    class Neuron:
        def __init__(self, bias: float=-1, weights: list[float]=-1, number_of_weights: int=-1, has_memory: bool=False) -> None:
            # Create a unique identifier for the neuron
            self.identifier = str(hashlib.sha256(datetime.now().strftime("%H:%M NEURON").encode()).hexdigest())[0:32]

            # Initialize memory
            self.has_memory = has_memory
            if has_memory:
                self.memory = 0.000000000001

            # Warn if a number of weights is specified while loading a preset of weights
            if weights != -1 and number_of_weights != -1:
                print(f"WARN: Number of weights was specified to neuron #{self.identifier}, this is ignored due to the neuron being given weights")
            
            # Error if types do not match
            if type(weights) != list and weights != -1:
                print(f"ERR: Weights should be type list not {type(weights)}")
                exit(1)
            if type(bias) != float and bias != -1:
                print(f"ERR: Bias should be type float not {type(bias)}")
                exit(1)
            if type(number_of_weights) != int and number_of_weights != -1:
                print(f"ERR: Bias should be type int not {type(bias)}")
                exit(1)

            # Check for if a bias was specified but not a list of weights and vise versa.
            passes_given = [(bias != -1), (weights != -1)]
            # Either load or create weights and bias
            if (bias != -1 or weights != -1) and passes_given != [1, 1]:
                # Error if one was specified but not the other
                print(f"ERR: {["Bias", "Weights"][passes_given.index(1)]} was specified but {["bias", "weights"][passes_given.index(0)]} was not.")
                exit(1)
            elif (bias != -1 or weights != -1) and passes_given == [1, 1]:
                # Load weights
                self.weights = weights
                self.bias = bias
            else:
                # Create weights
                if number_of_weights == -1:
                    # Error if a number of weights was not specified
                    print(f"ERR: Number of weights not specified to neuron #{self.identifier}")
                    exit(1)
                self.weights = np.random.randn(number_of_weights)
                self.bias = np.random.uniform(0, 1)

            self.number_of_weights = len(self.weights)

            return
        
        def predict(self, y: list[float]) -> float:
            # Check if number of weights matches number of inputs
            if len(y) != self.number_of_weights:
                print(f"ERR: Number of weights does not match number of inputs.")
                exit(1)
            # Dot multiply the arrays
            pred = np.dot(y, self.weights)

            # Set memory if neuron has one
            if self.has_memory:
                pred = pred * self.memory
                self.memory = float(pred)

            pred += self.bias

            # Return the prediction
            return pred

    class Layer:
        def __init__(self, number_of_inputs: int, number_of_neurons: int, loadf_name: str=None, memory_layer: bool=False) -> None:
            self.layer = []
            
            if loadf_name != None:
                pass
            else:
                for _ in range(number_of_neurons):
                    self.layer.append(NeuralNetwork.Neuron(number_of_weights=number_of_inputs, has_memory=memory_layer))

        def predict(self, y: list[float]) -> list[float]:
            pred = []
            
            for neuron in self.layer:
                pred.append(neuron.predict(y))

            return pred
        
        def tweak(self) -> None:
            for neuron in self.layer:
                for i,weight in enumerate(neuron.weights):
                    if np.random.uniform(-1, 1) > weight:
                        neuron.weights[i] += np.random.uniform(-1, 1)
                        if neuron.weights[i] < -1:
                            neuron.weights[i] = -1
                        elif neuron.weights[i] > 1:
                            neuron.weights[i] = 1

            return
    
    def interactive_loading_animation(self) -> None:
        self.interactive_loading_frames = ["/", "-", "\\", "|"]
        self.running = True
        self.animate = False
        self.posistion = (1, 1)

        current_frame = 0
        while self.running:
            while self.animate:
                print(f"{colorama.Cursor.POS(self.posistion[0], self.posistion[1])} {self.interactive_loading_frames[current_frame]}")
                current_frame += 1
                if current_frame > len(self.interactive_loading_frames)-1:
                    current_frame = 0
                time.sleep(0.1)

    def clear_screen(self) -> None:
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")

    def __init__(self, number_of_layers: int, number_of_inputs: int, layer_neurons: list[int], layer_has_memory: list[bool], number_of_classes: int, is_interactive=True):
        self.is_interactive = is_interactive
        self.layers = []
        
        if self.is_interactive:
            self.clear_screen()
            print("Creating layers...")
            ila_thread = threading.Thread(target=self.interactive_loading_animation)
            ila_thread.start()
        for i in range(number_of_layers+1):
            if self.is_interactive:
                create_start = time.time()
                print(f"  Layer {i}")
                self.posistion = (1, 2+i)
                self.animate = True
            if i == 0:
                self.layers.append(NeuralNetwork.Layer(number_of_inputs, layer_neurons[i], memory_layer=layer_has_memory[i]))
            elif i == number_of_layers:
                self.layers.append(NeuralNetwork.Layer(layer_neurons[i-1], number_of_classes))
            else:
                self.layers.append(NeuralNetwork.Layer(layer_neurons[i-1], layer_neurons[i], memory_layer=layer_has_memory[i]))
            if self.is_interactive:
                self.animate = False
                print(f"{colorama.Cursor.POS(1, 2+i)}Layer {i} created. ({time.time()-create_start} seconds)")

        self.running = False

    def predict(self, y: list[float]) -> list[float]:
        pred = y
        for layer in self.layers:
            pred = layer.predict(pred)

        return pred
    
    def tweak(self):
        for layer in self.layers:
            layer.tweak()

if __name__ == "__main__":
    nw1 = NeuralNetwork(2, 4, [5, 10], [False, True], 4)

    p_this = np.random.randn(4)
    print(nw1.predict(p_this))