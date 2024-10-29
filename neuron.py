from methods import *
from datasetsSimples import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn import datasets
import numpy as np



class Neuron:
    valor: float
    y: float

    def __init__(self, valor = 0):
        self.valor = valor

    def somaValores(self, layerAnterior: list, pesos: list):
        for i in range(0, len(layerAnterior)):
            self.valor += layerAnterior[i] * pesos[i]

    def phi(self, method = step_function):
        self.y = method(self.valor)
        return self.y




def updatePesos(array_pesos: list, n, classe_esperada, classe_recebida, vetor_resolucao):
    array_pesos = np.array(array_pesos)
    vetor_resolucao = np.array(vetor_resolucao)

    print("\nupdating pesos:")
    print(f"\tpesos novos = {array_pesos} + {n} * ({classe_esperada} - {classe_recebida}) * {vetor_resolucao}")
    print(f"\tpesos novos = {array_pesos} + {n * (classe_esperada - classe_recebida) * vetor_resolucao}")
    array_pesos = array_pesos + (n * (classe_esperada - classe_recebida) * vetor_resolucao) 

    print("\tpesos novos =", array_pesos)
    return array_pesos











def printaDataset(selecao):
    #diferencia as 2 classes do dataset
    class_0 = np.array([data['array'] for data in selecao if data['classe_esperada'] == 0])
    class_1 = np.array([data['array'] for data in selecao if data['classe_esperada'] == 1])

    #cria a janela
    plt.figure(figsize=(8, 6))

    #classe 0 = red,
    #classe 1 = blue
    plt.scatter(class_0[:, 0], class_0[:, 1], color='red',  label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')

    #titulos e etc
    plt.title("Dataset - Classe 0 vs Classe 1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()









if __name__ == "__main__":
    bias = random_gen(0, 1)
    n = 0.1





    iris = datasets.load_iris()

    #input de 2D
    dadosiris = iris.data[:, :2]

    #classe 0 -> Setosa and Versicolor,
    #classe 1 -> Virginica
    classesiris = np.where(iris.target == 2, 1, 0)

    #limpa o dataset pro algoritmo
    datasetIris = [
        {'array': list(dadosiris[i]), 'classe_esperada': int(classesiris[i])}
        for i in range(len(dadosiris))
    ]






    #mudar numero da seleção para qual precisar
    ##############################################
    selecao = datasetIris
    ##############################################

    printaDataset(selecao)
    print("==========[pressione enter]==========")
    input()

    array_entradas = selecao

    array_pesos = [0 for _ in range(len(selecao[0]['array']) + 1)] 
    for i in range(len(selecao[0]['array']) + 1):
        array_pesos[i] = random_gen()
    copia_comeco_pesos = np.copy(array_pesos)

    neuronium = Neuron()

    maxIter = 10000
    epoca = 0
    repetir = True
    while repetir and epoca < maxIter:
        repetir = False
        epoca += 1

        print(f"\n\n\nepoca: {epoca}")
        for entrada in array_entradas:
            neuronium.somaValores(entrada['array'], array_pesos)
            classe_esperada = entrada['classe_esperada']
            classe_recebida = neuronium.phi()

            print("\nsoma entradas calculada: ", str(neuronium.valor))

            if classe_recebida != classe_esperada:
                print("classe_recebida != classe_esperada:", classe_recebida, classe_esperada)
                array_pesos = updatePesos(array_pesos, n, classe_esperada, classe_recebida, [bias] + entrada['array'])

                repetir = True
            else:
                print("classe_recebida == classe_esperada:", classe_recebida, classe_esperada)

            print("==========[pressione enter]==========")
            # input()


    print("""
        ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░              
        ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░""")
    print(f"pesos iniciais = {copia_comeco_pesos}\npesos finais = {array_pesos}\n\nepoca = {epoca}\nn = {n}, bias = {bias}\ndataset = {selecao}")


    # reta = 
    #   (y, x) = (0, -w0/w2)
    #   m = -w1/w2