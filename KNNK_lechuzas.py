# Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset desde un archivo CSV
# Cambia 'ruta_al_archivo.csv' por la ubicación de tu archivo
ruta_archivo = 'iris.csv'
data = pd.read_csv(ruta_archivo)

# Inspeccionar el dataset (opcional)
print(data.head())

# Seleccionar características (Radiacion, Temperatura, Temperatura panel) y la etiqueta (Potencia)
X = data[['Radiacion', 'Temperatura', 'Temperatura panel']]
y = data['Potencia']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo K-Nearest Neighbors
k = 5  # Número de vecinos
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = knn.predict(X_test)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Generar gráfico de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title(f'Matriz de Confusión para K = {k}')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.show()

# Mostrar el reporte de clasificación
classification_rep = classification_report(y_test, y_pred)
print(classification_rep)
