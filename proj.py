from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

file_path = r'C:\Users\JH\Programs\ML_proj\dados.xlsx'
df = pd.read_excel(file_path)

dados = df.values

x = np.delete(dados, [0, 3, 4], axis=1)
y = dados[:, [3, 4]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_test)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=50)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

# Combine predictions by averaging
y_pred_combined = (y_pred_linear + y_pred_rf) / 2

acertos = 0

for var in range(len(y_pred_combined)):
    previsao = 'Empate' if np.rint(y_pred_combined[var][0]) == np.rint(y_pred_combined[var][1]) else ('Vitória do adversário' if np.rint(y_pred_combined[var][0]) > np.rint(y_pred_combined[var][1]) else 'Vitória do flamengo')
    resultado = 'Empate' if np.rint(y_test[var][0]) == np.rint(y_test[var][1]) else ('Vitória do adversário' if np.rint(y_test[var][0]) > np.rint(y_test[var][1]) else 'Vitória do flamengo') 

    if previsao == resultado:
        acertos += 1     
         
    print(f'Previsão: {np.rint(y_pred_combined[var])}, Resultado real: {y_test[var]}, Previsão: {previsao}, Resultado real: {resultado}')

mse = mean_squared_error(y_test, y_pred_combined)
r2 = r2_score(y_test, y_pred_combined)

y_test_flat = y_test.flatten()
y_pred_combined_flat = y_pred_combined.flatten()

pearson_corr = np.corrcoef(y_test_flat, y_pred_combined_flat)[0, 1]

print(f'Pearson Correlation Coefficient: {pearson_corr}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Acertos de outcomes: {acertos} de {len(y_test)} jogos testados, Acurácia: {acertos/len(y_test)}')
