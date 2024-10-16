from methods import *
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






if __name__ == "__main__":
    bias = -1
    array_entradas = [
        {
            'array': [2, 2],
            'classe_esperada': 1
        },
        {
            'array': [4, 4],
            'classe_esperada': 0
        }
    ]
    array_pesos = [-0.5441, 0.5562, 0.4074]
    n = 0.1





    neuronium = Neuron()

    epoca = 0
    repetir = True
    while repetir:
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
            input()


    print("""
        ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░ 
        ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░              
        ░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░""")
    print(f"pesos = {array_pesos}, epoca = {epoca}")