
import re
import unicodedata
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



correcciones = {
    "ecxelente": "excelente",
    "exelente": "excelente",
    "escelente": "excelente",
    "exelentes": "excelentes",
    "pesimo": "pesimo",
    "pesima": "pesima",
    "delisioso": "delicioso",
    "delisiosa": "deliciosa",
    "horrivle": "horrible",
    "malisimo": "pesimo",
    "malisima": "pesima",
    "atencion": "atencion"
}


def normalizar_texto(texto):
    texto = str(texto).lower().strip()

    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    texto = re.sub(r"[^a-z0-9\s]", " ", texto)

    texto = re.sub(r"\s+", " ", texto).strip()

    palabras = texto.split()
    palabras = [correcciones.get(p, p) for p in palabras]

    return " ".join(palabras)


datos = {

    "comentario": [


        "La comida estuvo excelente",
        "El servicio fue muy bueno",
        "Me encantó la atención del personal",
        "Los precios son bastante accesibles",
        "El lugar es muy bonito y agradable",
        "La experiencia fue maravillosa",
        "La comida estaba deliciosa",
        "El personal fue amable y atento",
        "La entrega fue rápida y eficiente",
        "Estoy muy satisfecho con el servicio",
        "La calidad de los productos es excelente",
        "Recomiendo totalmente este negocio",
        "El ambiente es cómodo y agradable",
        "Todo estuvo perfecto",
        "La atención fue excelente",
        "Los empleados fueron muy profesionales",
        "El producto superó mis expectativas",
        "La experiencia fue increíble",
        "El pedido llegó a tiempo",
        "Muy buena calidad y excelente atención",
        "El restaurante tiene un ambiente fantástico",
        "Me gustó mucho la comida",
        "El servicio fue rápido y eficiente",
        "El personal fue muy amable",
        "Volvería a comprar sin duda",


        "La comida estuvo horrible",
        "El servicio fue pésimo",
        "La atención del personal fue muy mala",
        "Los precios son demasiado elevados",
        "El lugar estaba sucio y desordenado",
        "Tuve una experiencia terrible",
        "La comida estaba fría y sin sabor",
        "El personal fue grosero",
        "La entrega fue demasiado lenta",
        "Estoy muy insatisfecho con el servicio",
        "La calidad de los productos es pésima",
        "No recomiendo este negocio",
        "El ambiente es incómodo",
        "Todo estuvo mal",
        "La atención fue deficiente",
        "Los empleados fueron poco profesionales",
        "El producto llegó dañado",
        "La experiencia fue desagradable",
        "El pedido llegó tarde",
        "Muy mala calidad y pésima atención",
        "El restaurante estaba muy sucio",
        "No me gustó la comida",
        "El servicio fue lento y malo",
        "Los meseros fueron poco atentos",
        "No volvería a comprar",


        "El restaurante está ubicado en el centro",
        "El establecimiento abre a las ocho",
        "El menú contiene varias opciones",
        "El negocio tiene dos sucursales",
        "La tienda vende diferentes productos",
        "El local cuenta con estacionamiento",
        "El pedido fue de dos productos",
        "El restaurante tiene mesas disponibles",
        "La empresa ofrece varios servicios",
        "El establecimiento está cerca del parque",
        "El horario de atención es de lunes a sábado",
        "La tienda tiene una página en internet",
        "El menú incluye bebidas y alimentos",
        "El negocio acepta pagos con tarjeta",
        "El producto viene en diferentes tamaños",
        "La sucursal está en una avenida principal",
        "El local tiene capacidad para cincuenta personas",
        "La empresa fue fundada hace varios años",
        "El pedido contiene tres artículos",
        "El establecimiento tiene servicio a domicilio",
        "La tienda vende ropa y accesorios",
        "El restaurante ofrece desayunos y almuerzos",
        "El negocio cuenta con varias promociones",
        "El producto está disponible en diferentes colores",
        "La empresa tiene atención por teléfono"

    ],

    "sentimiento": (
        ["Positivo"] * 25 +
        ["Negativo"] * 25 +
        ["Neutro"] * 25
    )
}


df = pd.DataFrame(datos)

df["comentario"] = df["comentario"].apply(normalizar_texto)



expresiones = {


    "excelente": 2.5,
    "excelentes": 2.5,
    "maravilloso": 2.5,
    "maravillosa": 2.5,
    "increible": 2.5,
    "delicioso": 2.5,
    "deliciosa": 2.5,
    "fantastico": 2.5,
    "fantastica": 2.5,
    "perfecto": 2.5,
    "perfecta": 2.5,
    "bueno": 1.5,
    "buena": 1.5,
    "amable": 1.5,
    "agradable": 1.5,
    "rapido": 1.5,
    "rapida": 1.5,
    "eficiente": 2.0,
    "satisfecho": 2.0,
    "satisfecha": 2.0,
    "recomiendo": 2.0,
    "accesible": 1.5,
    "encanto": 2.0,
    "encanto mucho": 2.5,
    "me gusto": 2.0,
    "me encanta": 2.5,
    "supero mis expectativas": 3.0,
    "volveria a comprar": 2.5,
    "sin duda": 1.0,


    "horrible": -3.0,
    "pesimo": -3.0,
    "pesima": -3.0,
    "terrible": -3.0,
    "desagradable": -2.5,
    "malo": -2.0,
    "mala": -2.0,
    "sucio": -2.0,
    "sucia": -2.0,
    "grosero": -2.5,
    "grosera": -2.5,
    "lento": -1.5,
    "lenta": -1.5,
    "deficiente": -2.5,
    "insatisfecho": -2.5,
    "insatisfecha": -2.5,
    "dañado": -2.0,
    "fria": -1.5,
    "incomodo": -1.5,
    "incomoda": -1.5,
    "caro": -1.5,
    "cara": -1.5,
    "elevado": -1.5,
    "elevada": -1.5,


    "no me gusto": -3.0,
    "no me gusta": -3.0,
    "no me agrado": -2.5,
    "no me encanto": -2.5,
    "no fue bueno": -2.5,
    "no fue buena": -2.5,
    "no estuvo bueno": -2.5,
    "no estuvo buena": -2.5,
    "no estoy satisfecho": -2.5,
    "no estoy satisfecha": -2.5,
    "no recomiendo": -3.0,
    "no volveria a comprar": -3.0,
    "no volveria": -2.5,

    "no estuvo mal": 2.5,
    "no fue tan malo": 1.5,
    "no fue tan mala": 1.5,
    "no estuvo nada mal": 2.5,


    "poco atento": -2.0,
    "poco atenta": -2.0,
    "poco atentos": -2.5,
    "poco atentas": -2.5,
    "atencion deficiente": -2.5,
    "trato desagradable": -2.5,
    "personal grosero": -2.5,
    "personal poco profesional": -2.5,
    "comida horrible": -3.0,
    "comida desagradable": -2.5,
    "comida sin sabor": -2.0,
    "comida fria": -2.0,
    "mal sabor": -2.0,
    "comida de mala calidad": -3.0,
    "servicio deficiente": -2.5,
    "pedido incorrecto": -2.0,
    "entrega tardia": -2.0,
    "mala experiencia": -2.5,
    "servicio lento": -2.0,
    "muy mala calidad": -3.0,
    "pesima atencion": -3.0,
    "muy buena calidad": 2.5,
    "excelente atencion": 3.0,
    "muy buen servicio": 2.5
}



def analizar_expresiones(comentario):

    texto = normalizar_texto(comentario)
    puntuacion = 0
    palabras_usadas = set()

    expresiones_ordenadas = sorted(
        expresiones.items(),
        key=lambda elemento: len(elemento[0].split()),
        reverse=True
    )

    for frase, valor in expresiones_ordenadas:

        patron = r"\b" + re.escape(frase) + r"\b"

        coincidencias = list(re.finditer(patron, texto))

        for coincidencia in coincidencias:

            inicio = coincidencia.start()
            final = coincidencia.end()

            posiciones = set(range(inicio, final))

            if any(posiciones.intersection(usadas)
                   for usadas in palabras_usadas):
                continue

            puntuacion += valor
            palabras_usadas.add(posiciones)

    return puntuacion




X = df["comentario"]
y = df["sentimiento"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


vectorizador = FeatureUnion([

    ("palabras", TfidfVectorizer(
        ngram_range=(1, 2),
        strip_accents="unicode",
        sublinear_tf=True
    )),

    ("caracteres", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        strip_accents="unicode",
        sublinear_tf=True
    ))

])


modelo = Pipeline([

    ("vectorizador", vectorizador),

    ("clasificador", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))

])



modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

print("\n====================================")
print("       EVALUACIÓN DE RESEÑAIA")
print("====================================")

print("\nExactitud del modelo:")
print(f"{accuracy_score(y_test, predicciones) * 100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones, zero_division=0))

print("\nMatriz de confusión:")
print(confusion_matrix(
    y_test,
    predicciones,
    labels=["Negativo", "Neutro", "Positivo"]
))


modelo.fit(X, y)



def clasificar_comentario(comentario):

    comentario_limpio = normalizar_texto(comentario)

    puntuacion = analizar_expresiones(comentario_limpio)

    prediccion_ml = modelo.predict([comentario_limpio])[0]

    probabilidades = modelo.predict_proba([comentario_limpio])[0]

    clases = modelo.named_steps["clasificador"].classes_

    probabilidades_dict = dict(zip(clases, probabilidades))

    if puntuacion >= 1.25:
        sentimiento_final = "Positivo"
        metodo = "Análisis léxico"

    elif puntuacion <= -1.25:
        sentimiento_final = "Negativo"
        metodo = "Análisis léxico"

    else:
        sentimiento_final = prediccion_ml
        metodo = "Machine Learning"

    return (
        sentimiento_final,
        metodo,
        puntuacion,
        probabilidades_dict
    )



print("\n====================================")
print("          RESEÑAIA 1.0")
print("     ANÁLISIS DE SENTIMIENTOS")
print("====================================")

print("\nSistema listo para analizar reseñas.")
print("Escribe 'salir' para cerrar el programa.")


while True:

    comentario = input("\nEscribe una reseña: ")

    if comentario.lower().strip() == "salir":
        print("\nGracias por utilizar RESEÑAIA.")
        break

    if not comentario.strip():
        print("Por favor, escribe un comentario válido.")
        continue

    sentimiento, metodo, puntuacion, probabilidades = (
        clasificar_comentario(comentario)
    )

    print("\n------------------------------------")
    print("         RESULTADO DEL ANÁLISIS")
    print("------------------------------------")

    print(f"\nComentario: {comentario}")

    print(f"\nSentimiento detectado: {sentimiento}")

    print(f"Método utilizado: {metodo}")

    print(f"Puntuación léxica: {puntuacion:.2f}")

    print("\nProbabilidades del modelo ML:")

    print(
        f"Positivo: {probabilidades['Positivo'] * 100:.2f}%"
    )

    print(
        f"Negativo: {probabilidades['Negativo'] * 100:.2f}%"
    )

    print(
        f"Neutro: {probabilidades['Neutro'] * 100:.2f}%"
    )

    print("------------------------------------")
