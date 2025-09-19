#Item E

import numpy as np

dados = np.loadtxt("arsenio_dataset (1).csv", delimiter=",", skiprows=1)

arsenio_agua = dados[:, 4]
y = dados[:, 5]

X = np.column_stack((np.ones(len(y)), arsenio_agua))

beta_simple = np.linalg.inv(X.T @ X) @ (X.T @ y)
y_pred_simple = X @ beta_simple

ss_total = np.sum((y - np.mean(y))**2)
ss_resid = np.sum((y - y_pred_simple)**2)
r2_simple = 1 - ss_resid/ss_total

print("Coeficientes modelo simples:", beta_simple)
print(f"R² modelo simples = {r2_simple:.4f}")

#Item F

y_f = np.array([0.12, 0.25, 0.07, 0.30, 0.22, 0.18, 0.40, 0.15, 0.28, 0.35,
              0.10, 0.20, 0.32, 0.27, 0.14, 0.26, 0.31, 0.11, 0.29, 0.23, 0.19])

idade = np.array([25, 34, 28, 45, 52, 31, 40, 29, 50, 60,
                  22, 36, 47, 53, 30, 41, 49, 27, 55, 39, 33])

beber = np.array([1, 2, 1, 3, 2, 2, 3, 1, 3, 2,
                  1, 2, 3, 2, 1, 3, 3, 1, 2, 2, 1])

cozinhar = np.array([2, 3, 2, 3, 2, 1, 2, 1, 3, 3,
                     1, 2, 3, 2, 2, 3, 2, 1, 3, 2, 1])

arsenio_agua = np.array([0.010, 0.025, 0.005, 0.030, 0.022, 0.018, 0.040, 0.015, 0.028, 0.035,
                         0.012, 0.020, 0.032, 0.027, 0.014, 0.026, 0.031, 0.011, 0.029, 0.023, 0.019])

X = np.column_stack([np.ones(len(y_f)), idade, beber, cozinhar, arsenio_agua])

beta = np.linalg.inv(X.T @ X) @ X.T @ y_f

#valor ajstado
y_hat = X @ beta

residuos = y_f - y_hat

#printar tabela
print("Obs | y_obs | y_ajs | Residuo")
for i in range(len(y_f)):
    print(f"{i+1:2d}  | {y_f[i]:.3f} | {y_hat[i]:.3f} | {residuos[i]:.3f}")

#Item G

y = dados[:, 5]

X_todo = np.column_stack((np.ones(len(y)), dados[:, [0, 2, 3, 4]]))
beta_todo = np.linalg.inv(X_todo.T @ X_todo) @ (X_todo.T @ y)
y_pred_todo = X_todo @ beta_todo

X_simple = np.column_stack((np.ones(len(y)), dados[:, 4]))
beta_simple = np.linalg.inv(X_simple.T @ X_simple) @ (X_simple.T @ y)
y_pred_simple = X_simple @ beta_simple

mse_todo = np.mean((y - y_pred_todo) ** 2)
rmse_todo = np.sqrt(mse_todo)
mae_todo = np.mean(np.abs(y - y_pred_todo))

mse_simple = np.mean((y - y_pred_simple) ** 2)
rmse_simple = np.sqrt(mse_simple)
mae_simple = np.mean(np.abs(y - y_pred_simple))

print("Modelo Completo - MSE:", mse_todo, "RMSE:", rmse_todo, "MAE:", mae_todo)
print("Modelo Simples  - MSE:", mse_simple, "RMSE:", rmse_simple, "MAE:", mae_simple)
