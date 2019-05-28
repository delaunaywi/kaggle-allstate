# -*- coding: utf-8 -*-




# Importation des librairies utilisées  
import pandas as pd # Pour les dataframes
import seaborn as sns # Pour les graphiques
import matplotlib.pyplot as plt # Pour les graphiques
import numpy as np
#Pour encoder les variables catégorielles
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#Division des données en train - test
from sklearn.model_selection import train_test_split
# Régression linéaire
from sklearn.linear_model import LinearRegression
# Abre de décision
from sklearn.tree import DecisionTreeRegressor
# Forêts aléatoires
from sklearn.ensemble import RandomForestRegressor
#### Mesures utilisées pour calculer la précision ####
# Erreur quadratique moyenne
from sklearn.metrics import mean_squared_error
# Overfitting (sur ou sous-ajustement)
from sklearn.model_selection import cross_val_score
# Hyper-paramètres
from sklearn.model_selection import GridSearchCV


# Téléchargement des données 
data = pd.read_csv('data/train.csv')


################################
#### Découverte des données ####
################################

#Affichage des 5 premières lignes du dataframe pour observer les données
data.head(5)

#Suppression de la colonn id qui ne nous est pas utile
data = data.drop(['id'], axis=1)


# Observation des tendances et mesure de la dispersion des données numériques
data.describe()

colonnes = data.columns

colonnes_numeriques = data._get_numeric_data().columns

liste_col_num = list(colonnes_numeriques)

liste_colonnes_cat = list(set(colonnes) - set(colonnes_numeriques))

#################################
#### Exploration des données ####
#################################
 #_____________________#
 #                     #
 #      UNIVARIEE      #
 #_____________________#
 
 #Affichage du graphe de chaque variable
# Histogramme pour chaque variable.
data.hist(bins=50, figsize=(20,15))
plt.show()

# Histogramme de la variable loss
data["loss"].hist(bins=100, figsize=(20,15))
plt.show()

# Modification de la variable loss avec le log(1+x)
data['loss'] = np.log1p(data['loss']) 

# Histogramme de la variable loss après modification
data["loss"].hist(bins=100)
plt.show()

#Création d'une copie du data 
cp_data = data.copy()
# Suppression de la colonne "loss" pour travailler plus tard les données
cp_data = cp_data.drop(['loss'], axis=1)

#Valeurs numériques
sns.boxplot(data=cp_data, orient="h", palette="Set2")
plt.show()

#Boîte à moustaches du coût des sinistres
sns.boxplot(data=data['loss'], orient="h", palette="Set2")
plt.show()

#names of all the columns
colnames = data.columns

# Graphe de toutes les variables qualitatives
n_cols = 4
n_rows = 29

for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))
    for j in range(n_cols):
        sns.countplot(x=colnames[i*n_cols+j], data=data, ax=ax[j])

 #_____________________#
 #                     #
 #       BIVARIEE      #
 #_____________________#


#Affichage de la corrélation des variables de la copie du data_train 
sns.pairplot(data)
plt.show()

corr_matrix = data.corr()
sns.heatmap(corr_matrix, 
            xticklabels=corr_matrix.columns.values,
            yticklabels=corr_matrix.columns.values)


 #_____________________#
 #                     #
 # VALEURS MANQUANTES  #
 #_____________________#


#Fonction comptant le nombre de lignes vides dans une colonne
def row_emptiness(df, column):
    line_count = 0
    for line in df[column]:
        if(line != None):
            line_count = line_count + 1
    return line_count

#Fonction utilisant row_emptiness pour regarder le taux de lignes vides dans les colonnes d'un dataframe
def missing_values(df):
   # truth = 0
    for i in df.columns:
        if(row_emptiness(df, str(i)) != 188318):
            print(str(i) + str(row_emptiness(df, str(i))))

missing_values(data)

############################
####  DATA PREPARATION #####
############################

 #______________________________________#
 #                                      #
 # Encodage des variables qualitatives  #
 #______________________________________#
 
# Récupération de tous les noms des variables qualitatives
label = []
for i in range(0,116):
    label.append(list(set(data[colonnes[i]].unique()))) 
    
#One hot encode toutes les variables catégorielles
var_cat = []
for i in range(0, 116):
    #Label encode
    label_encode = LabelEncoder()
    label_encode.fit(label[i])
    cat = label_encode.transform(data.iloc[:,i])
    cat = cat.reshape(data.shape[0], 1)
    #One hot encode 
    feature = OneHotEncoder(sparse=False,n_values=len(label[i])).fit_transform(cat)
    var_cat.append(cat)
    
# Faire un tableau à deux dimensions à partir de la liste des tableaux des ID
var_cat_encodees = np.column_stack(var_cat)

#Concaténation des variables catégorielles avce les données des variables ordinales
data_encode = np.concatenate((var_cat_encodees,data.iloc[:,116:].values),axis=1)

 #________________________________________________________#
 #                                                        #
 # Séparation des données en apprentissage et validation  #
 #________________________________________________________#

#Récupération de la dimension du data frame
r, c = data_encode.shape

#Y c'est la variable target "loss", X le reste
X = data_encode[:,0:(c-1)]
Y = data_encode[:,(c-1)]

# Séparation des données en train - test
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#########################################
####  Apprentissage des algorithmes #####
#########################################
 #__________________________#
 #                          #
 #   Régression linéaire    #
 #__________________________#
 
# Lancement du modèle 
lin_reg = LinearRegression()
# Entraînement du modèle sur les données
lin_reg.fit(X_train, Y_train)

# Prévision des données sur les données
dataset_predictions = lin_reg.predict(X_train)

# Calcul de la RMSE (Erreur quadratique moyenne)
lin_mse = mean_squared_error(np.expm1(Y_train), np.expm1(dataset_predictions))
lin_rmse = np.sqrt(lin_mse)
lin_rmse

 #__________________________#
 #                          #
 #    Arbre de décision     #
 #__________________________#
 # Lancement du modèle 
tree_reg = DecisionTreeRegressor()
# Entraînement du modèle sur les données
tree_reg.fit(X_train, Y_train)

# Prévision des données sur les données
dataset_predictions = tree_reg.predict(X_train)

# Calcul de la RMSE (Erreur quadratique moyenne)
tree_mse = mean_squared_error(np.expm1(Y_train), np.expm1(dataset_predictions))
tree_rmse = np.sqrt(tree_mse)
tree_rmse

#Vérification du sur-ajustement ou du sous-ajustement
scores = cross_val_score(tree_reg, X_train, Y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

#La fonction display_scores : affiche les résultats sur les scores, leurs moyennes et leurs écarts types
def display_scores(scores):
    print("Scores:", scores)
    print("Moyenne:", scores.mean())
    print("Ecart-type:", scores.std())
    
display_scores(tree_rmse_scores)

 #__________________________#
 #                          #
 #    Forêts aléatoires     #
 #__________________________#
 # Lancement du modèle 
forest_reg = RandomForestRegressor()
# Entraînement du modèle sur les données
forest_reg.fit(X_train, Y_train)

# Prévision des données sur les données
dataset_predictions = forest_reg.predict(X_train)

# Calcul de la RMSE (Erreur quadratique moyenne)
forest_mse = mean_squared_error(np.expm1(Y_train), np.expm1(dataset_predictions))
forest_rmse = np.sqrt(forest_mse)
forest_rmse

#Vérification du sur-ajustement ou du sous-ajustement
forest_scores = cross_val_score(forest_reg, X_train, Y_train, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# Recherche des hyper-paramètres
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)

grid_search.best_params_

best_model = grid_search.best_estimator_


dataset_predictions = best_model.predict(X_train)


best_mse = mean_squared_error(np.expm1(Y_train), np.expm1(dataset_predictions))
best_rmse = np.sqrt(best_mse)
best_rmse
