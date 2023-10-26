import pickle
import nltk

def analizar_texto(texto_analizar):
    # Cargar modelos desde el archivo pickle
    with open("modelos_entrenados.pkl", 'rb') as file:
        modelos_cargados = pickle.load(file)
    
    # Acceder a modelos individuales
    tfidf_vectorizer = modelos_cargados["Tfid_vectorizer"]
    xgboost_model = modelos_cargados["XGBoost"]
    
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    from nltk.tokenize import word_tokenize
    import string
    
    def tokenizado_stopwords(texto):
        texto = texto.lower()
        palabras = word_tokenize(texto)
        stop_words = set(stopwords.words('english'))
        palabras_filtradas = []
        for i in palabras:
            if i not in stop_words and i not in [".", ",", ":", ";", " ", "(", ")", "’", "“", "”", "@", "?", "-", "—", "_"] and i not in string.punctuation:
                palabras_filtradas.append(i)
        return ' '.join(palabras_filtradas)
    
    texto_analizar_tokenizado = tokenizado_stopwords(texto_analizar)
    
    texto_vectorizado = tfidf_vectorizer.transform([texto_analizar_tokenizado])
    prediccion = xgboost_model.predict(texto_vectorizado)
    
    # Devuelve la predicción como resultado
    return "The information contained in this news article is probably true" if prediccion == 0 else "The information contained in this news article is probably false"
