import numpy as np

"""#### Definição da função de custo"""
def compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data):
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
    v = np.array(data[1:,18]) ### OverallCond (19-1)
    u = np.array(data[1:,62]) ### GarageArea (63-1)
    z = np.array(data[1:,19]) ### YearBuilt (20-1)
    
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

    # para normalizar o valor de OverallCond
    vmin = min(v)
    vmax = max(v)
    for i in range(0, len(v)):
        v[i] = (v[i]-vmin)/(vmax-vmin)

    # para normalizar o valor de GarageArea
    umin = min(u)
    umax = max(u)
    for i in range(0, len(u)):
        u[i] = (u[i]-umin)/(umax-umin)

    # para normalizar o valor de YearBuilt
    zmin = min(z)
    zmax = max(z)
    for i in range(0, len(z)):
        z[i] = (z[i]-zmin)/(zmax-zmin)

    # calculo h(casa)
    h = []
    for i in range(0, len(x)):
        h.append(theta_0 + theta_1 * x[i] + theta_2 * w[i] + theta_3 * v[i] + theta_4 * u[i] + theta_5 * z[i])
    
    # Calcula erro quadratico medio
    for i in range(0, len(x)):
        total_cost += pow(h[i] - y[i], 2)
    total_cost /= len(x)
    
    return total_cost


"""#### Define as funções de Gradiente Descendente"""

def step_gradient(theta_0_current, theta_1_current, theta_2_current, theta_3_current, theta_4_current, theta_5_current, data, alpha):
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
    v = np.array(data[1:,18]) ### OverallCond (19-1)
    u = np.array(data[1:,62]) ### GarageArea (63-1)
    z = np.array(data[1:,19]) ### YearBuilt (20-1)

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
    
    # para normalizar o valor de OverallCond
    vmin = min(v)
    vmax = max(v)
    for i in range(0, len(v)):
        v[i] = (v[i]-vmin)/(vmax-vmin)

    # para normalizar o valor de GarageArea
    umin = min(u)
    umax = max(u)
    for i in range(0, len(u)):
        u[i] = (u[i]-umin)/(umax-umin)

    # para normalizar o valor de YearBuilt
    zmin = min(z)
    zmax = max(z)
    for i in range(0, len(z)):
        z[i] = (z[i]-zmin)/(zmax-zmin)

    # calculo h(casa)
    h = []
    for i in range(0, len(x)):
        h.append(theta_0_current + theta_1_current * x[i] + theta_2_current * w[i] + theta_3_current * v[i] + theta_4_current * u[i] + theta_5_current * z[i])

    # resolvendo as derivadas, com o somatório
    D0 = 0
    D1 = 0
    D2 = 0
    D3 = 0
    D4 = 0
    D5 = 0
    for i in range(0, len(x)):
      D0 += h[i] - y[i]
      D1 += (h[i] - y[i]) * x[i]
      D2 += (h[i] - y[i]) * w[i]
      D3 += (h[i] - y[i]) * v[i]
      D4 += (h[i] - y[i]) * u[i]
      D5 += (h[i] - y[i]) * z[i]
    D0 *= (2 / len(x))
    D1 *= (2 / len(x))
    D2 *= (2 / len(x))
    D3 *= (2 / len(x))
    D4 *= (2 / len(x))
    D5 *= (2 / len(x))

    # atualizando o valor de theta_0 e theta_1
    theta_0_updated = theta_0_current - (alpha * D0)
    theta_1_updated = theta_1_current - (alpha * D1)
    theta_2_updated = theta_2_current - (alpha * D2)
    theta_3_updated = theta_2_current - (alpha * D3)
    theta_4_updated = theta_2_current - (alpha * D4)
    theta_5_updated = theta_2_current - (alpha * D5)

    return theta_0_updated, theta_1_updated, theta_2_updated, theta_3_updated, theta_4_updated, theta_5_updated


def gradient_descent(data, starting_theta_0, starting_theta_1, starting_theta_2, starting_theta_3, starting_theta_4, starting_theta_5, learning_rate, num_iterations):
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
    theta_3 = starting_theta_3
    theta_4 = starting_theta_4
    theta_5 = starting_theta_5
    
    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    for i in range(num_iterations):
        theta_0, theta_1, theta_2, theta_3, theta_4, theta_5 = step_gradient(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data, learning_rate)
        
    return [theta_0, theta_1, theta_2, theta_3, theta_4, theta_5]

"""#### Executa a função gradient_descent() para obter os parâmetros otimizados, Theta0 e Theta1."""

import sys

# Args
dataFile = sys.argv[1]
iterations = int(sys.argv[2])

# Open file
data = np.genfromtxt(dataFile, delimiter=',')

# Execute function
theta_0, theta_1, theta_2, theta_3, theta_4, theta_5 = gradient_descent(data, starting_theta_0=0, starting_theta_1=0, starting_theta_2=0, starting_theta_3=0, starting_theta_4=0, starting_theta_5=0, learning_rate=0.0001, num_iterations=iterations)

#Imprimir parâmetros otimizados
print ('theta_0: ', theta_0)
print ('theta_1: ', theta_1)
print ('theta_2: ', theta_2)
print ('theta_3: ', theta_3)
print ('theta_4: ', theta_4)
print ('theta_5: ', theta_5)

#Imprimir erro com os parâmetros otimizados
print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data))
