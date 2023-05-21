import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#importer les donnees
dataset = pd.read_csv(r"C:\Users\HAB SOLUTIONS\OneDrive\Documents\s6\Analyse des donnees\Automobile_data.csv",sep=";")
print(dataset)
#separer les donnees qu'on va exploiter des donnees du test
X = dataset.drop('price', axis=1) # toutes les colonnes sauf "price"
y = dataset['price']
n = X.shape[0]
train_size = int(n * 0.8)
X_train = X.iloc[:train_size,:]
X_test = X.iloc[train_size:,:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]
#centrer et reduire les donnees
Xcr = (X_train - X_train.mean()) / X_train.std()
y_train_cr = (y_train - y_train.mean()) / y_train.std()
# Calculer les valeurs et les vecteurs propres
matrice= Xcr.to_numpy()
matrice_correlation = np.corrcoef(matrice, rowvar=False)
valeurs_propres, vecteurs_propres = np.linalg.eigh(matrice_correlation)
#creer la matrice de design
X_design = np.hstack((np.ones((Xcr.shape[0], 1)), Xcr.dot(vecteurs_propres[:, -1:-3:-1])))
#calculer les parametres de regression
parametres = np.linalg.inv(X_design.T.dot(X_design)).dot(X_design.T).dot(y_train_cr)
#faire les predictions sur l'ensemble test
X_test_cr = (X_test - X_train.mean()) / X_train.std()
X_design_test = np.hstack((np.ones((X_test_cr.shape[0], 1)), X_test_cr.dot(vecteurs_propres[:, -1:-3:-1])))
y_pred_cr = X_design_test.dot(parametres)
#reconvertir les donnees a l'ensemble d'origine
y_pred = y_pred_cr * y_train.std() + y_train.mean()
#calculer l'erreur de prediction moyen
mae = np.mean(np.abs(y_test - y_pred))
print("Mean absolute error:", mae)
#affichage des resultats
plt.scatter(y_test, y_pred)
plt.xlabel("Vraies valeurs")
plt.ylabel("Prédictions")
plt.show()