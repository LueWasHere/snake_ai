import lib.neuralNetwork as neuralNetwork
import numpy as np
import lib.snake as snake

if __name__ == "__main__":
    nw1 = neuralNetwork.NeuralNetwork(2, 5, [5, 10], [False, False], 3)
    best_copy = None
    sizes = []
    passes = []
    map_size = (30, 15)

    #print(nw1.layers[0].layer[0].weights)
    #exit()

    for i in range(0, 100):
        size, passesx = snake.main_ai(nw1, i, map_size)
        if size == -1:
            break
        if best_copy is None:
            best_copy = nw1
        else:
            if sizes[-1] > size: # or passes[-1] > passesx: # Decided passes is not helping xdddd
                # Need to retweak
                nw1 = best_copy
                nw1.tweak()
                nw1.tweak()
                nw1.tweak()
            elif sizes[-1] < size: # and passes[-1] < passesx:
                # New best
                best_copy = nw1
                nw1.tweak()
            else:
                # Same performance
                nw1.tweak()
                nw1.tweak()
            
        sizes.append(size)
        passes.append(passesx)

    snake.clear_screen()
    for i,tup in enumerate(zip(sizes, passes)):
        print(f"Gen {i}: Reached {tup[0]} size and survived {tup[1]} passes")
    