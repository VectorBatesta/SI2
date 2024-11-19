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

    count_epocas = 0
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
            count_epocas = epoca
            break

    return errosEpocas, copia_comeco_pesos, modelo, count_epocas





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
    dataset = datasetXor
    ##############################################









    model = initialization(2, 3, 1)
    maxEpocas = 10000
    taxaAprendizagem = 0.1
    tolerancia = 0.01

    errosEpocas, copia_comeco_pesos, modelo_final, count_epocas = MLPTrain(dataset, model, maxEpocas, taxaAprendizagem, tolerancia)

    print("""
        ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░              
        ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░""")









    #peguei do chat gebitoca
    def format_weights(weights):
        """
        Formats the weight matrices for better readability.
        """
        if isinstance(weights, list):
            return "[" + ", ".join(format_weights(w) for w in weights) + "]"
        elif isinstance(weights, dict):
            return "{" + ", ".join(f"{k}: {format_weights(v)}" for k, v in weights.items()) + "}"
        elif isinstance(weights, float):
            return f"{weights:.2f}"  # Format floats to two decimal places
        else:
            return str(weights)

        
    #peguei do chat gebitoca tbm
    def format_dataset(dataset):
        formatted = []
        for item in dataset:
            array_str = ", ".join(map(str, item['array']))
            formatted.append(f"Array: [{array_str}] | Expected Class: {item['classe_esperada']}")
        return "\n".join(formatted)
    
    #mais um do chat gebitoca... when will it ever end???
    def convert_to_float(data):
        """
        Recursively converts all numpy float64 elements in the data structure
        to standard Python float.
        """
        if isinstance(data, list):
            return [convert_to_float(item) for item in data]
        elif isinstance(data, dict):
            return {key: convert_to_float(value) for key, value in data.items()}
        elif isinstance(data, (np.float64, np.float32, float)):  # Cover all float cases
            return float(data)
        else:
            return data  # Return as-is for unsupported types






    initial_weights = convert_to_float(copia_comeco_pesos)
    final_weights = convert_to_float(modelo_final)

    initial_weights_formatted = format_weights(initial_weights)
    final_weights_formatted = format_weights(final_weights)
    dataset_formatted = format_dataset(dataset)

    print(f"\nInitial weights = {initial_weights_formatted}\n")
    print(f"Final weights = {final_weights_formatted}\n")
    print(f"Max Epochs = {maxEpocas}, Epochs used = {count_epocas}, Learning rate = {taxaAprendizagem}, Bias = {bias}\n")
    print("Dataset:\n" + dataset_formatted)

    # print(f"Initial weights = {copia_comeco_pesos}\nFinal weights = {modelo_final}\nEpochs = {maxEpocas}, Learning rate = {taxaAprendizagem}, Bias = {bias}\nDataset = {dataset}")
