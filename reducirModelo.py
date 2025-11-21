import tensorflow as tf
import numpy as np
import os

# --- 1. CONFIGURACIÓN ---
KERAS_MODEL_PATH = 'VOWEL_GRAY_96x96_CustomCNN_FINAL_ACC80.72.keras' # ¡Pon el nombre de tu mejor modelo!
VALIDATION_DIR = 'VOCALES/test' # ¡Tu carpeta de validación!
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
CLASSES = ['A', 'E', 'I', 'O', 'U']

print(f"Cargando modelo Keras desde: {KERAS_MODEL_PATH}")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
model.summary()

# --- 2. DATASET DE REPRESENTACIÓN (VITAL PARA CUANTIZAR) ---
# El conversor necesita ver datos del "mundo real" para saber
# cómo mapear de float32 a int8. Usamos el set de validación.

print(f"Cargando dataset de validación desde: {VALIDATION_DIR}")

# Cargamos el dataset DE FORMA SIMPLE (sin fondos, sin chroma key)
# PERO SÍ EN ESCALA DE GRISES
validation_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    color_mode='grayscale', # <-- ¡IMPORTANTE!
    shuffle=False # No es necesario barajar para esto
)

# El script de entrenamiento se encargaba de los fondos. Aquí, 
# para la CUANTIZACIÓN, solo necesitamos imágenes en grises 
# que se parezcan a la ENTRADA FINAL del modelo (que era en grises).
# Esta es la forma más simple de proveer esos datos.

def representative_data_gen():
  """Generador que produce datos de validación para el conversor TFLite."""
  print("Generando datos de representación...")
  # Usamos .take(100) para no usar todo el dataset, 100-200 lotes son suficientes
  for images, _ in validation_ds.take(100):
    # La entrada del modelo es (batch_size, 96, 96, 1)
    # y debe ser float32
    yield [tf.cast(images, tf.float32)]

print("✅ Dataset de representación listo.")

# --- 3. CONVERSIÓN Y CUANTIZACIÓN ---
print("Iniciando conversión a TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Activar optimizaciones (incluye cuantización)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Proveer el dataset de representación
converter.representative_dataset = representative_data_gen

# Forzar a que la entrada y salida sean int8 (común en micros)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8  # Entrada será int8
converter.inference_output_type = tf.int8 # Salida será int8

# Convertir
tflite_model_quant = converter.convert()

# --- 4. GUARDAR EL MODELO .tflite ---
TFLITE_MODEL_PATH = 'vowel_model_quant_int8.tflite'
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model_quant)

print(f"\n{'='*50}")
print(f"✅ ¡Conversión Exitosa!")
print(f"Modelo Keras original: {os.path.getsize(KERAS_MODEL_PATH) / 1024:.2f} KB")
print(f"Modelo TFLite cuantizado: {len(tflite_model_quant) / 1024:.2f} KB")
print(f"Guardado en: {TFLITE_MODEL_PATH}")
print(f"{'='*50}")