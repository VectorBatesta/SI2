# modelo (E, O, S)

# -----------------------------------
# -----------------------------------

def initilization(inputs, hidden, outputs):
	# bias - pesos indice 0 (+1)
	# matrizPesosOculta [hidden + 1  x inputs] 
	# matrizPesosSaida  [outputs + 1 x hidden + 1]

	# inicializar aleatoriamente os pesos
	# Sugestão: (-0.5 0.5)
	# matrizPesosOcultos = valores aleatorios
	# matrizPesosSaida   = valores aleatorios

	# modelo(Wh, Wo, (E,H,O))
	# return (modelo)

# -----------------------------------
# -----------------------------------

def sigmoideActivation(net):
	# return (1 / (1 + exp^-net))

def derivativeSigmoideActivation(net):
# f'(x) = f(x) * (1 - f(x)) 
	# valor = sigmoideActivation(net) * 
		# (1 - sigmoideActivation(net))
	# return (valor)

# -----------------------------------
# -----------------------------------

# modelo(Wh, Wo, (E,H,O))
def forward(Exemplo, Classe, modelo):

	# [+1 Exemplo] --> add o valor do bias da camada oculta
	# NetHidden = calcular os valores dos sinais dos 
	# 	neuronios da camada oculta
	# I_Hidden = sigmoideActivation(NetHidden)

	# [+1 I_Hidden] --> add o valor do bias da camada saida
	# NetOutpus = calcular os valores dos sinais dos neuronios
	#   da camada de saida
	# O_output = sigmoideActivation(NetOutpus)

	# return (O_outpus, NetHidden, NetOutputs, I_Hidden)

# -----------------------------------
# -----------------------------------

# modelo(Wh, Wo, (E,H,O))
# modelo = initilization(4, 3, 2)
# modelo.matrizPesosOcultos
# modelo.matrizPesosSaida

def MLPTrain(dataset, modelo, epocas, taxaAprendizgem, tolerancia):

	# ErrosEpocas = []

	# enqto erroEpoca > tolerancia & qtdeEpocas < epocas:

		#   epochError  = 0 --> erro da epoca
		# 	EPOCA - para todos os exemplos do dataset
		#  
		# 		X = exemplo do dataset
		#   	Y = classe de X

		#  		propagar o sinal na rede
		# 		valores = forward(X, Y, modelo)

			#   valores.NetHidden
			#   valores.I_Hidden
			#   valores.NetOutputs
			#   valores.O_outputs

			#   calcular o erro do exemplo (Real - Previsto)
			#   erroExemplo = Classe - O_outputs

			#   Calcular o gradiente da camada de Saída   
			# 	OutputGradient = ...

			#   Calcular o gradiente da camada Oculta
			# 	Hidden Gradient = ... [OutputGradient]

			#	Atualizar o pesos
			#	matrizPesosSaida = matrizPesosSaida + taxaAprendizado * I_Hidden 
			#       * OutputGradient 	

			# 	matrizPesosOculta = matrizPesosOculta + taxaAprendizado * Exemplo
		    #       * HiddenGradient

		    # epochError = epochError + erroExemplo

    	# ErrosEpocas.append(erroExemplo)

