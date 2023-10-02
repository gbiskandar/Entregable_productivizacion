from flask import Flask, render_template, request, redirect, url_for
from Productivizacion_prueba import analizar_texto  # Importa la función desde Productivizacion_prueba

app = Flask(__name__)

@app.route('/')
def pagina_principal():
    return render_template('pagina_principal.html')

@app.route('/formulario', methods=['GET', 'POST'])
def mostrar_formulario():
    if request.method == 'POST':
        texto_analizar = request.form['texto']
        resultado = analizar_texto(texto_analizar)  # Utiliza la función importada
        return redirect(url_for('mostrar_resultado', resultado=resultado))

    return render_template('formulario.html')

@app.route('/resultado/<resultado>')
def mostrar_resultado(resultado):
    return render_template('resultado.html', resultado=resultado)

'''
#Ruta para entrenar el modelo
@app.route('/entrenar_modelo', methods=['GET', 'POST'])
def entrenar_modelo():
    if request.method == 'POST':
        #logica para entrenar el modelo

    return render_template('entrenar_modelo.html')

#Ruta para guardar la predicción en la base de datos
@app.route('/guardar_prediccion', methods=['POST'])
def guardar_prediccion():
    if request.method == 'POST':
        texto_analizar = request.form['texto']
        resultado = analizar_texto(texto_analizar)
        
        db = get_db()
        db.execute("INSERT INTO predicciones (texto, resultado) VALUES (?, ?)", (texto_analizar, resultado))
        db.commit()
        
        return redirect(url_for('mostrar_resultado', resultado=resultado))

#Ruta para leer las predicciones almacenadas en la base de datos
@app.route('/leer_predicciones', methods=['GET'])
def leer_predicciones():
    db = get_db()
    cursor = db.execute("SELECT texto, resultado FROM predicciones")
    predicciones = cursor.fetchall()
    
    return render_template('leer_predicciones.html', predicciones=predicciones)
'''

if __name__ == '__main__':
    app.run(debug=True, port=3500)
