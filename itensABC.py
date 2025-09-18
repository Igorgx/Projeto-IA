#Item A

import numpy as np

data = np.loadtxt("arsenio_dataset (1).csv", delimiter=",", skiprows=1)

X = data[:, [0, 2, 3, 4]]
X = np.column_stack((np.ones(X.shape[0]), X))

y = data[:, 5]

X_T = X.T
beta = np.linalg.inv(X_T @ X) @ (X_T @ y)

y_pred = X @ beta


equacao = (
    f"y = {beta[0]:.3f} "
    f"+ {beta[1]:.5f}* Idade "
    f"+ {beta[2]:.5f}* Uso_Beber "
    f"+ {beta[3]:.5f}* Uso_Cozinhar "
    f"+ {beta[4]:.3f}* Arsenio_Agua"
)

print("Coeficientes:", beta)
print("Equação do modelo:", equacao)

#Item B 

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)

beta = np.array([0.4875, -0.0008, -0.0227, -0.0415, 13.2400])

entrada = np.array([1, 30, 5, 5, 0.135])

y_pred = entrada @ beta
print(f"Previsão de arsênio nas unhas: {y_pred:.4f} ppm")


#Item C
r2 = 1 - (ss_res / ss_tot)
print("R²:", r2)