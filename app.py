from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
import numpy as np

import nltk
nltk.download('omw-1.4')
nltk.download('punkt')

lematizador = WordNetLemmatizer()
intenciones = json.loads(open('intenciones.json').read())

palabras = pickle.load(open('palabras.pkl', 'rb'))
clases = pickle.load(open('clases.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def limpiar_sentencia(sentencia):
    palabras_sentencia = nltk.word_tokenize(sentencia)
    palabras_sentencia = [lematizador.lemmatize(
        word) for word in palabras_sentencia]
    return palabras_sentencia


def bag_of_words(sentencia):
    palabras_sentencia = limpiar_sentencia(sentencia)
    bag = [0] * len(palabras)
    for w in palabras_sentencia:
        for i, word in enumerate(palabras):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predecir_clase(sentencia):
    bow = bag_of_words(sentencia)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r]for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append(
            {'itencion': clases[r[0]], 'probabilidad': str(r[1])})
        return return_list


def get_respuesta(intenciones_list, intenciones_json):
    tag = intenciones_list[0]['itencion']
    lista_de_intenciones = intenciones_json['intenciones']
    result = 'No entendi :('
    for i in lista_de_intenciones:
        if i['tag'] == tag:
            result = random.choice(i['respuestas'])
            break
    return result


def RecognizeColection(k):
    Colection = {
        'proyectos',
        'proyecto',
        'usuarios',
        'usuario',
        'reportes',
        'reporte',
        'modulos',
        'modulo'
    }
    for y in Colection:
        if (y in k):
            return True
    return False


def GetBeneficiario(message):
    EvitateWords = {
        'beneficiarios',
        'beneficiario',
        'tambogrande',
        'apellidado',
        'proyectos',
        'encuentra',
        'principal',
        'proyecto',
        'usuarios',
        'apellido',
        'proyecto',
        'nombrado',
        'reportes',
        'reporte',
        'modulos',
        'usuario',
        'sullana',
        'llamado',
        'nombres',
        'enlista',
        'muestra',
        'modulo',
        'nombre',
        'metros',
        'senora',
        'señora',
        ' cuyo ',
        ' como ',
        'senor',
        'piura',
        'señor',
        ' con ',
        'busca',
        'halla',
        'lista',
        ' fin ',
        ' de ',
        ' en ',
        ' el ',
        ' la '
    }
    message = ' ' + message + ' '
    for elem in EvitateWords:
        message = message.replace(elem, '')
        message = ' ' + message + ' '
    return message.strip()


def BusquedaDeNumbers(Text):
    Sintext = Text.split()
    Num = ""
    for NWord in Sintext:
        if (NWord.isdigit()):
            Num = Num + NWord
    if (Num != ""):
        if (len(Num) == 8):
            return {'DNI': Num}
        elif (len(Num) == 9 or len(Num) == 6):
            return {'Telefono': Num}
    else:
        pass


app = Flask(__name__)


@app.route("/ApiQuestIA", methods=['GET'])
def home():
    JsonAws = {}
    mensaje = str(request.args['Query'])
    mensaje = mensaje.strip().lower()
    JsonAws['Query'] = str(mensaje)
    ContainsNumbers = BusquedaDeNumbers(mensaje)
    if (ContainsNumbers == None):
        ints = predecir_clase(mensaje)
        res = get_respuesta(ints, intenciones)
        if 'beneficiario' in res["campos"]:
            res["campos"]["beneficiario"] = GetBeneficiario(mensaje)
        JsonAws['Answer'] = res
    else:
        ListaDeKeys = list(ContainsNumbers.keys())
        if ListaDeKeys[0] == 'DNI':
            JsonAws['Answer'] = {
                    "collection": "proyectos",
                    "campos": {
                        "dni": {
                            "Value": ContainsNumbers['DNI'],
                            "Operator": "="
                        }
                    }
                }
        elif ListaDeKeys[0] == 'Telefono':
            JsonAws['Answer'] = {
                    "collection": "proyectos",
                    "campos": {
                        "telefono": {
                            "Value": ContainsNumbers['Telefono'],
                            "Operator": "="
                        }
                    }
                }
    return jsonify(JsonAws)

if __name__ == '__main__':
    app.run()
