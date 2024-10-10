from typing import List
from methods import *



class Neuron:
    valor: float

    def __init__(self, valor = 0):
        self.valor = valor

    def somaValores(self, layerAnterior: list, pesos: list):
        for i in range(0, len(layerAnterior)):
            self.valor += layerAnterior[i] * pesos[i]

    def phi(self, method = step_function):
        self.y = method(self.valor)


if __name__ == "__main__":
    tamEntradas = 5
    array_entradas = [1] + [random_gen(-.5, .5) for _ in range(1, tamEntradas)]
    array_pesos = [random_gen(-0.5, 0.5) for _ in range(0, tamEntradas)]

    print(array_entradas)
    print(array_pesos)

    neuronium = Neuron()
    neuronium.somaValores(array_entradas, array_pesos)

    print(neuronium.valor, neuronium.phi())
    
    # tamNeurons = 5
    # array_neurons_inicial = [Neuron() for _ in range(tamNeurons)]
    
