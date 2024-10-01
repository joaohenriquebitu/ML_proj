
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

file_path = r'C:\Users\JH\Programs\ML_proj\dados.xlsx'
df = pd.read_excel(file_path)

dados = df.values



x = np.delete(dados, [0, 3, 4], axis=1)
y = dados[:, [3, 4]]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

model = LinearRegression()
model.fit(x_train, y_train)
acertos = 0
y_pred = model.predict(x_test)

for var in range(len(y_pred)):
    previsao = 'Empate' if np.rint(y_pred[var][0]) == np.rint(y_pred[var][1]) else ('Vitória do adversário' if np.rint(y_pred[var][0]) > np.rint(y_pred[var][1]) else 'Vitória do flamengo')
    resultado = 'Empate' if np.rint(y_test[var][0]) == np.rint(y_test[var][1]) else ('Vitória do adversário' if np.rint(y_test[var][0]) > np.rint(y_test[var][1]) else 'Vitória do flamengo') 

    if previsao == resultado:
        acertos += 1     
         
    print(f'Previsão: {np.rint(y_pred[var])}, Resultado real: {y_test[var]}, Previsão: {previsao}, Resultado real: {resultado}')
          
          

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)




y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()


y_pred_int_flat = np.rint(y_pred_flat).astype(int)

pearson_corr = np.corrcoef(y_test_flat, y_pred_int_flat)[0, 1]

print(f'Pearson Correlation Coefficient: {pearson_corr}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Acertos de outcomes: {acertos} de {len(y_test)} jogos testados, Acurácia: {acertos/len(y_test)}')