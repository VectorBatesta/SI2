import numpy as np
from methods import *

#inicializando o mlperson
def initialization(inputs, hidden, outputs):
    #add bias (+1 input)
    matrizPesosOcultos = np.random.uniform(-0.5, 0.5, (hidden + 1, inputs + 1))
    matrizPesosSaida = np.random.uniform(-0.5, 0.5, (outputs, hidden + 1))
    modelo = {
        "Wh": matrizPesosOcultos,
        "Wo": matrizPesosSaida,
        "E": inputs,
        "H": hidden,
        "O": outputs
    }

    return modelo



#forward propagation
def forwardPropagation(Exemplo, Classe, modelo):
    #bias
    Exemplo = np.append(Exemplo, 1) #+1 = bias
    Wh = modelo["Wh"]
    Wo = modelo["Wo"]

    #hidden layer
    NetHidden = np.dot(Wh, Exemplo)
    I_Hidden = sigmoideActivation(NetHidden)
    I_Hidden = np.append(I_Hidden, 1) #add bias no hiddenlayer 

    #output layer
    NetOutputs = np.dot(Wo, I_Hidden)
    O_outputs = sigmoideActivation(NetOutputs)

    return O_outputs, NetHidden, NetOutputs, I_Hidden




#treinando mlp
def MLPTrain(dataset, modelo, epocas, taxaAprendizagem, tolerancia):
    ErrosEpocas = []

    for epoca in range(epocas):
        epochError = 0

        for exemplo in dataset:
            X, Y = exemplo["input"], exemplo["class"]
            O_outputs, NetHidden, NetOutputs, I_Hidden = forwardPropagation(X, Y, modelo)

            #acha o erro
            erroExemplo = Y - O_outputs

            #gradiente na camada oculta?
            OutputGradient = erroExemplo * derivativeSigmoideActivation(O_outputs)
            HiddenGradient = np.dot(modelo["Wo"].T, OutputGradient)[:-1] * derivativeSigmoideActivation(
                sigmoideActivation(NetHidden)
            )

            #update nos pesos
            modelo["Wo"] += taxaAprendizagem * np.outer(OutputGradient, I_Hidden)
            modelo["Wh"] += taxaAprendizagem * np.outer(HiddenGradient, np.append(X, 1))

            #update erro
            epochError += np.mean(np.abs(erroExemplo))

        ErrosEpocas.append(epochError)
        print(f"Epoch {epoca + 1}, Error: {epochError}")
        if epochError < tolerancia:
            break
    #
    return modelo, ErrosEpocas





if __name__ == "__main__":
    dataset = [
        {"input": [0, 0], "class": [0]},
        {"input": [0, 1], "class": [1]},
        {"input": [1, 0], "class": [1]},
        {"input": [1, 1], "class": [0]},
    ]

    #inicializa modelo
    inputs, hidden, outputs = 2, 3, 1
    modelo = initialization(inputs, hidden, outputs)

    # Train model
    modelo, erros = MLPTrain(dataset, modelo, epocas=10000, taxaAprendizagem=0.1, tolerancia=0.01)
