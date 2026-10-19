import io
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Inicializar la aplicación de FastAPI
app = FastAPI(
    title="API de Reconocimiento de Imágenes",
    description=(
        "API para clasificar imágenes utilizando modelo entrenado con 210"
        " clases."
    ),
    version="1.0",
)

# Configurar CORS (Permite que aplicaciones web externas o frontends se conecten sin problemas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringirlo si lo necesitas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo y las clases
model = None
class_names = []


@app.on_event("startup")
async def load_model_and_classes():
  global model, class_names
  try:
    model = tf.keras.models.load_model("modeloDL.keras")

    with open("clases.json", "r", encoding="utf-8") as f:
      class_names = json.load(f)

    print(
        "¡Modelo y clases cargados con éxito! Clases totales:"
        f" {len(class_names)}"
    )
  except Exception as e:
    print(f"Error al cargar el modelo o las clases: {e}")

@app.get("/")
def home():
  return {
      "estado": "activo",
      "mensaje": "La API de reconocimiento de 210 clases está funcionando.",
  }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  # 1. Validar que el archivo sea una imagen
  if not file.content_type.startswith("image/"):
    raise HTTPException(
        status_code=400,
        detail=(
            "El archivo proporcionado no es una imagen válida (debe ser JPEG,"
            " PNG, etc.)."
        ),
    )

  if model is None or not class_names:
    raise HTTPException(
        status_code=500,
        detail="El modelo o las clases no se cargaron correctamente en el servidor.",
    )

  try:
    # 2. Leer la imagen enviada
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # 3. Aplicar el mismo preprocesamiento del entrenamiento:
    # - Redimensionar a 128x128
    image = image.resize((128, 128))
    # - Convertir a escala de grises ('L')
    image = image.convert("L")

    # - Convertir a arreglo numpy y normalizar a [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0

    # - Ajustar dimensiones a la forma esperada por el modelo: (1, 128, 128, 1)
    image_array = np.expand_dims(image_array, axis=-1)  # Añade canal (128,128,1)
    image_array = np.expand_dims(image_array, axis=0)  # Añade batch (1,128,128,1)

    # 4. Realizar la predicción
    predictions = model.predict(image_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    # 5. Buscar el nombre real de la clase
    predicted_class_name = class_names[predicted_index]

    return {
        "clase_predicha": predicted_class_name,
        "probabilidad": round(confidence * 100, 2),  # Porcentaje de confianza
    }

  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Ocurrió un error al procesar la imagen: {str(e)}"
    ) 