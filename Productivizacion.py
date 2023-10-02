#CARGA DE DATOS Y PREPROCESADO------------------------------------------------------------

#importamos librerias para la carga de los datos csv
import pandas as pd
import numpy as np

#cargamos los csv en dos dataframes diferentes, fake vs real news. HAY QUE REVISAR COMO HACERLO EN GIT
dataset = pd.read_csv("./Data/WELFake_Dataset.csv")

#Eliminamos las columnas que no nos hacen falta en este análisis
dataset.drop("Unnamed: 0",axis=1,inplace=True)
dataset.drop("title",axis=1,inplace=True)

#Revisamos el estado de los datos
print("En dataset hay {} nulos".format(dataset.isnull().sum()))

#Eliminamos los null values, ya que contamos con registros suficientes
dataset = dataset.dropna()

#Se hace un shuffle de los registros
dataset = dataset.sample(frac=1, ignore_index=True, random_state=42)

#Generamos una función para eliminar el texto que figura antes del guión, el cual da información sobre el periódico del cual se ha extraido.

def eliminar_texto_previo(text):
    # Dividir el texto en función del guión y quedarse con la segunda parte (si existe)
    partes = text.split(" — ", 1)
    if len(partes) > 1:
        return partes[1]
    else:
        return text
    
dataset["text"] = dataset.text.apply(eliminar_texto_previo)

#en primer lugar, se debe separar la información en train y test
from sklearn.model_selection import train_test_split

X_data = dataset["text"] #Analizaremos el texto de la noticia
y_data = dataset["label"] #Dataframe para que mantenga el column name
X_train, X_test, y_train, y_test = train_test_split(
    X_data,y_data, test_size=0.25, random_state=42
)

#ELIMINACIÓN DE STOPWORDS Y TOKENIZACIÓN----------------------------------------------------

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import string 

def tokenizado_stopwords (texto):
    texto = texto.lower() #lo pasamos a minúsculas
    palabras = word_tokenize(texto) #tokenizado del texto
    stop_words = set(stopwords.words('english')) #cargamos el set de stopwords en inglés

    #eliminamos las stopwords del texto tokenizado
    palabras_filtradas = []
    for i in palabras:
        if i not in stop_words and i not in [".",",",":",";"," ","(",")","’","“","”","@","?","-","—","_"] and i not in string.punctuation : #He intentado eliminar todos los caracteres que pueda haber en el texto y que no sean palabras
            palabras_filtradas.append(i)
    return ' '.join(palabras_filtradas)
      
X_train_word_tokenize = X_train.apply(tokenizado_stopwords)
X_test_word_tokenize = X_test.apply(tokenizado_stopwords)

#MODELO DE VECTORIZACIÓN (TfidfVectorizer) ---------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

Tfid_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
Tfid_X_train = Tfid_vectorizer.fit_transform(X_train_word_tokenize)
Tfid_X_test = Tfid_vectorizer.transform(X_test_word_tokenize)



#PREDICCIÓN CON XGBOOST ------------------------------------------------------------

import xgboost as xgb
from xgboost import XGBClassifier  # Para clasificación
import time
from sklearn.metrics import accuracy_score

start_time = time.time()
XGBoost_model_Tfid = XGBClassifier(
    objective='binary:logistic',  # Para clasificación binaria
    max_depth=3,                 # Profundidad máxima del árbol. Escojo 3 para evitar overfitting
    learning_rate=0.1,           # Tasa de aprendizaje
    n_estimators=100             # Número de árboles (estimadores)
)

XGBoost_model_Tfid.fit(Tfid_X_train,y_train)
end_time = time.time()
training_time_xgboost_tfid = end_time - start_time
XGBoost_prediction_Tfid = XGBoost_model_Tfid.predict(Tfid_X_test)
XGBoost_Tfid_prediction_accuracy_train = XGBoost_model_Tfid.score(Tfid_X_train,y_train)
XGBoost_accuracy_Tfid = accuracy_score(y_test, XGBoost_prediction_Tfid) 

print("Tiempo de entrenamiento tfid: {}".format(training_time_xgboost_tfid))
print("XGBoost has an accuracty in train of {:.2f}".format(XGBoost_Tfid_prediction_accuracy_train))
print("XGBoost has an accuracty in test of {:.2f}".format(XGBoost_accuracy_Tfid))

#PICKLE Y GUARDADO DE MODELOS

import pickle

#Guardamos modelo con pickle
modelos_entrenados = {
    "Tfid_vectorizer":Tfid_vectorizer,
    "XGBoost":XGBoost_model_Tfid
}

with open("modelos_entrenados.pkl", 'wb') as file:
    pickle.dump(modelos_entrenados, file)

#PARTE DEL EDA -----------------------------------------------------------------------------
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr_xgboost_tfid, tpr_xgboost_tfid, thresholds_xgboost_tfid = roc_curve(y_test, XGBoost_prediction_Tfid)
roc_auc_xgboost_tfid = roc_auc_score(y_test, XGBoost_prediction_Tfid)

# Plot ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_xgboost_tfid, tpr_xgboost_tfid, label= "TfidVectorizer (AUC = {roc_auc_xgboost_tfid:.2f})")

# Plot the random guessing line
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)

# Customize the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Show the plot
plt.show()