import numpy as np

"""#### Definição da função de custo"""
def compute_cost(theta_0, theta_1, theta_2, data):
    """
    Calcula o erro quadratico medio
    
    Args:
        theta_0 (float): intercepto da reta 
        theta_1 (float): inclinacao da reta
        data (np.array): matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    
    Retorna:
        float: o erro quadratico medio
    """
    total_cost = 0

    # recebendo os elementos das colunas
    x = np.array(data[1:,46]) ### GrLivArea (pos 47-1)
    y = np.array(data[1:,80]) ### SalePrice (pos 81-1)
    w = np.array(data[1:,17]) ### OverallQual (pos 18-1)
    
    # para normalizar o valor de GrLivArea
    xmin = min(x)
    xmax = max(x)
    for i in range(0, len(x)):
        x[i] = (x[i]-xmin)/(xmax-xmin)

    # para normalizar o valor de OverallQual
    wmin = min(w)
    wmax = max(w)
    for i in range(0, len(w)):
        w[i] = (w[i]-wmin)/(wmax-wmin)

    # calculo h(casa)
    h = []
    for i in range(0, len(x)):
        h.append(theta_0_current + theta_1_current * x[i] + theta_2_current * w[i])
    
    # Calcula erro quadratico medio
    for i in range(0, len(x)):
        total_cost += pow(h[i] - y[i], 2)
    total_cost /= len(x)
    
    return total_cost


"""#### Define as funções de Gradiente Descendente"""

def step_gradient(theta_0_current, theta_1_current, theta_2_current, data, alpha):
    """Calcula um passo em direção ao EQM mínimo
    
    Args:
        theta_0_current (float): valor atual de theta_0
        theta_1_current (float): valor atual de theta_1
        data (np.array): vetor com dados de treinamento (x,y)
        alpha (float): taxa de aprendizado / tamanho do passo 

    """
    
    # recebendo os elementos das colunas
    x = np.array(data[1:,46]) ### GrLivArea (pos 47-1)
    w = np.array(data[1:,17]) ### OverallQual (pos 18-1)
    y = np.array(data[1:,80]) ### SalePrice (pos 81-1)

    # para normalizar o valor de GrLivArea
    xmin = min(x)
    xmax = max(x)
    for i in range(0, len(x)):
        x[i] = (x[i]-xmin)/(xmax-xmin)

    # para normalizar o valor de 
    wmin = min(w)
    wmax = max(w)
    for i in range(0, len(w)):
        w[i] = (w[i]-wmin)/(wmax-wmin)
    
    # calculo h(casa)
    h = []
    for i in range(0, len(x)):
        h.append(theta_0_current + theta_1_current * x[i] + theta_2_current * w[i])

    # resolvendo as derivadas, com o somatório
    D0 = 0
    D1 = 0
    D2 = 0
    for i in range(0, len(x)):
      D0 += h[i] - y[i]
      D1 += (h[i] - y[i]) * x[i]
      D2 += (h[i] - y[i]) * w[i]
    D0 *= (2 / len(x))
    D1 *= (2 / len(x))
    D2 *= (2 / len(x))

    # atualizando o valor de theta_0 e theta_1
    theta_0_updated = theta_0_current - (alpha * D0)
    theta_1_updated = theta_1_current - (alpha * D1)
    theta_2_updated = theta_2_current - (alpha * D2)

    return theta_0_updated, theta_1_updated, theta_2_updated


def gradient_descent(data, starting_theta_0, starting_theta_1, starting_theta_2, learning_rate, num_iterations):
    """executa a descida do gradiente
    
    Args:
        data (np.array): dados de treinamento, x na coluna 0 e y na coluna 1
        starting_theta_0 (float): valor inicial de theta0 
        starting_theta_1 (float): valor inicial de theta1
        learning_rate (float): hyperparâmetro para ajustar o tamanho do passo durante a descida do gradiente
        num_iterations (int): hyperparâmetro que decide o número de iterações que cada descida de gradiente irá executar
    
    Retorna:
        list : os primeiros dois parâmetros são o Theta0 e Theta1, que armazena o melhor ajuste da curva. O terceiro e quarto parâmetro, são vetores com o histórico dos valores para Theta0 e Theta1.
    """

    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1
    theta_2 = starting_theta_2
    
    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    for i in range(num_iterations):
        theta_0, theta_1, theta_2 = step_gradient(theta_0, theta_1, theta_2, data, learning_rate)
        
    return [theta_0, theta_1, theta_2]

"""#### Executa a função gradient_descent() para obter os parâmetros otimizados, Theta0 e Theta1."""

import sys

# Args
dataFile = sys.argv[1]
iterations = int(sys.argv[2])

# Open file
data = np.genfromtxt(dataFile, delimiter=',')

# Execute function
theta_0, theta_1, theta_2 = gradient_descent(data, starting_theta_0=0, starting_theta_1=0, starting_theta_2=0, learning_rate=0.0001, num_iterations=iterations)

#Imprimir parâmetros otimizados
print ('theta_0: ', theta_0)
print ('theta_1: ', theta_1)
print ('theta_2: ', theta_2)

#Imprimir erro com os parâmetros otimizados
print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1, theta_2, data))
