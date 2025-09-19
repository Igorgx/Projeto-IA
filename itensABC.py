#Item A

import numpy as np

data = np.loadtxt("arsenio_dataset (1).csv", delimiter=",", skiprows=1)

X = data[:, [0, 2, 3, 4]]
X = np.column_stack((np.ones(X.shape[0]), X))

y = data[:, 5]

X_T = X.T
beta = np.linalg.inv(X_T @ X) @ (X_T @ y)
y_pred_all = X @ beta 


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

beta = np.array([0.4875, -0.0008, -0.0227, -0.0415, 13.2400])

entrada = np.array([1, 30, 5, 5, 0.135])
y_pred_case = entrada @ beta
print(f"Previsão de arsênio nas unhas: {y_pred_case:.4f} ppm")

#Item C
ss_resid = np.sum((y - y_pred_all) ** 2)   
ss_total = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_resid / ss_total)
print("R²:", r2)