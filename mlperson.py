from methods import *
from datasetsSimples import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn import datasets
import numpy as np







class Neuron:
    def __init__(self, valor=0):
        self.valor = valor
        self.y = 0

    def somaValores(self, layerAnterior, pesos):
        self.valor = sum(layerAnterior[i] * pesos[i] for i in range(len(layerAnterior)))

    def phi(self, method=step_function):
        self.y = method(self.valor)
        return self.y








def initialization(inputs, hidden, outputs):
    matrizPesosOculta = [[random_gen() for _ in range(inputs + 1)] for _ in range(hidden)]
    matrizPesosSaida = [[random_gen() for _ in range(hidden + 1)] for _ in range(outputs)]
    return {"Wh": matrizPesosOculta, "Wo": matrizPesosSaida, "structure": (inputs, hidden, outputs)}







def forward(exemplo, classe, modelo):
    Wh = modelo["Wh"]
    Wo = modelo["Wo"]


    #hidden layer 
    NetHidden = [sum(exemplo[j] * Wh[i][j] for j in range(len(exemplo))) for i in range(len(Wh))]
    I_Hidden = [sigmoideActivation(net) for net in NetHidden]


    #bias aqui
    exemplo = [1] + exemplo
    I_Hidden = [1] + I_Hidden

    #layer de output
    NetOutputs = [sum(I_Hidden[j] * Wo[i][j] for j in range(len(I_Hidden))) for i in range(len(Wo))]
    O_outputs = [sigmoideActivation(net) for net in NetOutputs]



    return {"O_outputs": O_outputs, "NetHidden": NetHidden, "NetOutputs": NetOutputs, "I_Hidden": I_Hidden}







def MLPTrain(dataset, modelo, epocas, taxaAprendizagem, tolerancia):

    errosEpocas = []
    copia_comeco_pesos = (modelo["Wh"].copy(), modelo["Wo"].copy())
    bias = 1


    for epoca in range(epocas):
        epochError = 0
        for exemplo in dataset:
            X = exemplo["array"]
            Y = exemplo["classe_esperada"]

            valores = forward(X, Y, modelo)
            erroExemplo = [Y - valores["O_outputs"][0]]

            # Gradients
            OutputGradient = [erroExemplo[0] * derivativeSigmoideActivation(valores["NetOutputs"][0])]
            HiddenGradient = [derivativeSigmoideActivation(valores["NetHidden"][i]) *
                              sum(OutputGradient[k] * modelo["Wo"][k][i + 1] for k in range(len(OutputGradient)))
                              for i in range(len(valores["NetHidden"]))]

            # Update weights
            for i in range(len(modelo["Wo"])):
                for j in range(len(modelo["Wo"][i])):
                    modelo["Wo"][i][j] += taxaAprendizagem * OutputGradient[i] * valores["I_Hidden"][j]

            for i in range(len(modelo["Wh"])):
                for j in range(len(modelo["Wh"][i])):
                    modelo["Wh"][i][j] += taxaAprendizagem * HiddenGradient[i] * ([bias] + X)[j]

            epochError += sum(e ** 2 for e in erroExemplo)

        errosEpocas.append(epochError)
        if epochError < tolerancia:
            break

    return errosEpocas, copia_comeco_pesos, modelo





if __name__ == "__main__":
    bias = 1

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
    dataset = datasetOr
    ##############################################

    # printaDataset(dataset)
    print("==========[pressione enter]==========")
    input()









    model = initialization(2, 3, 1)
    maxEpocas = 10000
    taxaAprendizagem = 0.1
    tolerancia = 0.01

    errosEpocas, copia_comeco_pesos, modelo_final = MLPTrain(dataset, model, maxEpocas, taxaAprendizagem, tolerancia)

    print("""
        ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░              
        ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░""")
    print(f"Initial weights = {copia_comeco_pesos}\nFinal weights = {modelo_final}\nEpochs = {maxEpocas}, Learning rate = {taxaAprendizagem}, Bias = {bias}\nDataset = {dataset}")
