from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from Productivizacion_prueba import analizar_texto
#from pyspark.sql import SparkSession

app = Flask(__name__)

'''# Crear SparkSession
spark = SparkSession.builder.appName('myApp').getOrCreate()'''

'''# Conectar a la base de datos
db_path = '/dbfs/path/to/predictions.db' #sustituir por la ruta de la base de datos en databricks
df = spark.read.format('jdbc').options(
          url='jdbc:sqlite:' + db_path,
          dbtable='predictions').load()'''

#ruta para la pagina principal
@app.route('/')
def index():
    return render_template('index.html')

#ruta para el formulario de predicción de noticias
@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        news = request.form['news']
        prediction = analizar_texto(news)
        return redirect(url_for('prediction', prediction=prediction))
    return render_template('form.html')

#ruta para mostrar el resultado de la predicción
@app.route('/prediction/<prediction>')
def prediction(prediction):
    return render_template('prediction.html', prediction=prediction)

#ruta para guardar la predicción en la base de datos
'''@app.route('/store_prediction', methods=['POST'])
def store_prediction():
    news = request.form['news']
    prediction = request.form['prediction']
    new_row = spark.createDataFrame([(news, prediction)], ['news', 'prediction'])
    df = df.union(new_row)
    df.write.format('jdbc').options(
          url='jdbc:sqlite:' + db_path,
          dbtable='predictions').mode('append').save()
    return redirect(url_for('read_predictions'))'''

#ruta para leer las predicciones almacenadas en la base de datos
'''@app.route('/read_predictions')
def read_predictions():
    df = spark.read.format('jdbc').options(
          url='jdbc:sqlite:' + db_path,
          dbtable='predictions').load()
    predictions = df.collect()
    return render_template('read_prediction.html', predictions=predictions)'''

#ruta para leer las noticias según su tipo (fake o true)
'''@app.route('/news/<type>')
def news(type):
    df = spark.read.format('jdbc').options(
          url='jdbc:sqlite:' + db_path,
          dbtable='dataset').load()
    if type == 'fake':
        news = df.filter(df.label == 0).collect()
    elif type == 'true':
        news = df.filter(df.label == 1).collect()
    return render_template('news.html', news=news)'''


if __name__ == '__main__':
    app.run(debug=True, port=3500)

