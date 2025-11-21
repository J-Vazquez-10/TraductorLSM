import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
import json
import matplotlib.pyplot as plt

# ====================================================================
# --- 0. CONFIGURACIÓN INICIAL Y OPTIMIZACIÓN DE HARDWARE ---
# ====================================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
NUM_CPU_CORES = os.cpu_count()

tf.config.threading.set_intra_op_parallelism_threads(NUM_CPU_CORES)
tf.config.threading.set_inter_op_parallelism_threads(NUM_CPU_CORES)
print(f"✅ Configuración de CPU: {NUM_CPU_CORES} núcleos disponibles.")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU detectada y VRAM optimizada.")
    except RuntimeError as e:
        print(f"⚠️ Error en configuración de GPU: {e}")
else:
    print("ℹ️ No se detectó GPU. Usando CPU.")

# --- 1. PARÁMETROS CLAVE Y AJUSTE DE ESTABILIDAD ---
TRAIN_DIR = 'VOCALES/train'
TEST_DIR = 'VOCALES/test'

# ==================== SOLUCIÓN FASE 2: Ruta de Fondos ====================
BACKGROUNDS_DIR = 'backgrounds_pexels_400' # ¡Asegúrate de que esta carpeta exista y tenga imágenes!
# =======================================================================

IMAGE_SIZE = (96, 96) 
BATCH_SIZE = 64 

VOWEL_CLASSES = ['A', 'E', 'I', 'O', 'U']
NUM_CLASSES = len(VOWEL_CLASSES) # Ahora es 5

EPOCHS = 50 

# ==================== SOLUCIÓN FASE 1: Tasa de Aprendizaje Segura ====================
LEARNING_RATE = 0.0001 
# ==================================================================================
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999

# ==================== SOLUCIÓN FASE 2: Parámetros del Chroma Key (Fondo Verde) ====================
# Estos valores (en HSV) definen tu fondo verde. ¡AJÚSTALOS SI ES NECESARIO!
# H (Tono): 0.25 - 0.45 (rango para verdes)
# S (Saturación): > 0.4 (ignora verdes pálidos)
# V (Valor/Brillo): > 0.2 (ignora verdes oscuros)
HSV_GREEN_MIN = tf.constant([0.25, 0.4, 0.2], dtype=tf.float32)
HSV_GREEN_MAX = tf.constant([0.45, 1.0, 1.0], dtype=tf.float32)
# ================================================================================================


# ====================================================================
# --- 2. NUEVO PIPELINE DE DATOS (Soluciones Fases 1 y 2) ---
# ====================================================================

def create_soft_augmentation_pipeline():
    """
    SOLUCIÓN FASE 1: Crea un pipeline de aumentación suave que no
    "destruye" la imagen.
    """
    return tf.keras.Sequential([
        layers.RandomRotation(0.1),       # Rotación sutil (máx 36 grados)
        layers.RandomZoom(0.1),         # Zoom sutil (máx 10%)
        layers.RandomTranslation(0.05, 0.05), # Traslación sutil (~4-5 píxeles)
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1), 
        layers.RandomBrightness(0.1),
        layers.GaussianNoise(0.05),
    ], name="soft_augmentation_pipeline")


def chroma_key_blend(image, background):
    """
    SOLUCIÓN FASE 2: Reemplaza el fondo verde de 'image' con 'background'.
    Esto se ejecuta 100% en TensorFlow para alta eficiencia en la GPU.
    """
    # Convertir la imagen a HSV (es más fácil aislar el verde)
    # Las imágenes de Keras están en [0, 255], HSV de TF espera [0, 1]
    image_hsv = tf.image.rgb_to_hsv(image / 255.0)
    
    # Dividir en canales H, S, V
    h, s, v = tf.split(image_hsv, 3, axis=-1)
    
    # Crear la máscara: Comprobar qué píxeles están DENTRO del rango verde
    mask_h = tf.logical_and(h >= HSV_GREEN_MIN[0], h <= HSV_GREEN_MAX[0])
    mask_s = tf.logical_and(s >= HSV_GREEN_MIN[1], s <= HSV_GREEN_MAX[1])
    mask_v = tf.logical_and(v >= HSV_GREEN_MIN[2], v <= HSV_GREEN_MAX[2])
    
    green_mask_bool = tf.logical_and(mask_h, tf.logical_and(mask_s, mask_v))
    
    # Invertir la máscara: queremos la MANO (True), no el fondo (False)
    hand_mask_bool = tf.logical_not(green_mask_bool)
    
    # Convertir de booleano a float32 (0.0 o 1.0)
    hand_mask = tf.cast(hand_mask_bool, dtype=tf.float32)
    
    # Suavizar los bordes de la máscara (truco rápido de "blur")
    # Esto evita bordes duros y pixelados
    mask_small = tf.image.resize(hand_mask, (24, 24), method='bilinear')
    hand_mask_smooth = tf.image.resize(mask_small, IMAGE_SIZE, method='bilinear')

    # Normalizar el fondo también a [0, 255] (si no lo está ya)
    # y asegurarse de que tenga el tamaño correcto
    background_resized = tf.image.resize(background, IMAGE_SIZE)
    
    # Combinar: (Mano * Máscara) + (Fondo * (1.0 - Máscara))
    # 'image' y 'background_resized' deben estar en el mismo rango [0, 255]
    blended_image = (image * hand_mask_smooth) + (background_resized * (1.0 - hand_mask_smooth))
    
    return blended_image


def load_backgrounds(directory, shuffle_buffer_size=1000):
    """Carga, redimensiona y repite infinitamente las imágenes de fondo."""
    print(f"Cargando fondos desde {directory}...")
    bg_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=None, # No hay etiquetas
        image_size=IMAGE_SIZE, # Redimensionar fondos
        interpolation='bilinear',
        batch_size=None # Cargar imágenes individuales
    )
    
    # Crear un stream infinito y aleatorio de fondos
    bg_ds = bg_ds.shuffle(shuffle_buffer_size).repeat()
    return bg_ds


def load_and_optimize_data(directory, augment_pipeline, background_ds, augment=False, class_filter=None):
    """
    Pipeline de carga NUEVO Y COMPLETO.
    1. Carga imágenes (Color)
    2. Descomprime lotes
    3. Combina cada imagen con un fondo aleatorio
    4. Aplica el chroma key (Color)
    5. *** CONVIERTE A ESCALA DE GRISES ***
    6. Aplica aumentación suave (Grises)
    7. Re-empaqueta en lotes y optimiza
    """
    
    print(f"Cargando desde {directory}...")
    print(f"Filtrando solo clases: {class_filter}")
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=class_filter, 
        image_size=IMAGE_SIZE,
        interpolation='bilinear',
        batch_size=BATCH_SIZE, # Carga en lotes (más rápido)
        shuffle=augment
    )
    
    class_names = dataset.class_names
    
    # Deshacer los lotes para procesar imagen por imagen
    dataset = dataset.unbatch()
    
    # Combinar el dataset de señas con el stream de fondos
    # Ahora cada elemento es: ((imagen, etiqueta), fondo)
    zipped_ds = tf.data.Dataset.zip((dataset, background_ds))
    
    # Función de mapeo principal
    @tf.function
    def process_image(data, background):
        image, label = data
        
        # SOLUCIÓN FASE 2: Reemplazar el fondo verde (sigue en RGB)
        blended_image_rgb = chroma_key_blend(image, background)
        
        # ===================================================================
        # --- ¡CAMBIO AQUÍ! CONVERTIR A ESCALA DE GRISES ---
        # ===================================================================
        blended_image = tf.image.rgb_to_grayscale(blended_image_rgb)
        # ===================================================================

        # SOLUCIÓN FASE 1: Aplicar aumentación suave (ahora sobre grises)
        if augment:
            blended_image = augment_pipeline(blended_image, training=True)
            
        return blended_image, label
    
    dataset = zipped_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Re-empaquetar y optimizar
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset, class_names

# --- Crear los pipelines ---
print("\n📁 Cargando datos y pipelines...")

# 1. Cargar el stream de fondos
background_dataset = load_backgrounds(BACKGROUNDS_DIR)

# 2. Crear el pipeline de aumentación suave
soft_augment_pipeline = create_soft_augmentation_pipeline()

# 3. Cargar el dataset de entrenamiento
train_dataset, class_names = load_and_optimize_data(
    TRAIN_DIR,
    augment_pipeline=soft_augment_pipeline,
    background_ds=background_dataset,
    augment=True,
    class_filter=VOWEL_CLASSES
)

# 4. Cargar el dataset de validación
# (Nota: También reemplazamos el fondo en validación para ser consistentes)
validation_dataset, _ = load_and_optimize_data(
    TEST_DIR,
    augment_pipeline=soft_augment_pipeline,
    background_ds=background_dataset,
    augment=False, # Sin aumentación en validación
    class_filter=VOWEL_CLASSES
)

print(f"✅ Orden de clases detectado por Keras: {class_names}")
print(f"Total de clases: {len(class_names)}")


# ====================================================================
# --- 3. ARQUITECTURA CNN LIGERA PROPIA (Modificada para Grises) ---
# ====================================================================

print("\n🏗️ Construyendo modelo CNN LIGERA (SeparableConv2D) con penalización L2...")

strong_l2_reg = tf.keras.regularizers.l2(0.01)
model = models.Sequential([
    # ===================================================================
    # --- ¡CAMBIO AQUÍ! ACEPTAR 1 CANAL DE ENTRADA ---
    # ===================================================================
    layers.SeparableConv2D(32, (3, 3), padding='same', 
                           input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)), # <-- CAMBIADO A 1 CANAL
    # ===================================================================
    
    layers.BatchNormalization(), 
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2, 2)), 
    
    layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2), 
    
    layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)), 
    layers.Dropout(0.3), 
    
    layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(), 
    
    layers.Dropout(0.5), 
    layers.Dense(NUM_CLASSES, 
                 activation='softmax', 
                 dtype='float32',
                 kernel_regularizer=strong_l2_reg)
])

# Compilación (con el LR de la Fase 1)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
total_params = model.count_params()
print(f"\n📊 Parámetros totales: {total_params:,}")
# El tamaño en MB será incluso menor ahora
print(f"✅ Modelo ligero ({total_params * 4 / (1024**2):.2f} MB) y apto para cuantización TFLite.")

# ====================================================================
# --- 4. CALLBACKS Y ENTRENAMIENTO (Con ajuste FASE 1) ---
# ====================================================================

# (Callback personalizado sin cambios)
class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold, filepath):
        super().__init__()
        self.threshold = threshold
        self.filepath = filepath
        self.best_acc_above_threshold = 0.0

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get('val_accuracy')
        if (current_acc is not None and 
            current_acc >= self.threshold and 
            current_acc > self.best_acc_above_threshold):
            
            self.best_acc_above_threshold = current_acc
            filename = self.filepath.format(epoch=epoch + 1, a=current_acc * 100)
            self.model.save(filename, overwrite=True)
            print(f"\nÉpoca {epoch + 1}: ¡Guardado por Umbral! val_accuracy={current_acc:.4f} > {self.threshold*100}% en {filename}")


callbacks = [
    # ==================== SOLUCIÓN FASE 1: Callback de seguridad ====================
    tf.keras.callbacks.TerminateOnNaN(), # <-- AÑADIDO
    # ==============================================================================
    
    EarlyStopping(
        monitor='val_accuracy', 
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=12, 
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_VOWEL_CustomCNN_TinyML_GRAYSCALE.keras', # Nombre de archivo actualizado
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    AccuracyThresholdCallback(
        threshold=0.95, 
        filepath='VOWEL_GRAY_ACC95_E{epoch:02d}_ACC{a:.2f}.keras' # Nombre de archivo actualizado
    )
]

# --- 5. ENTRENAMIENTO ---
print("\n🚀 Iniciando entrenamiento (CON FONDOS ALEATORIOS Y SALIDA GRIS)...")

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks,
    verbose=1
)

# --- 6. EVALUACIÓN FINAL Y GUARDADO ---
print("\n📈 Evaluando modelo final (VOCALES - GRISES)...")
# (Resto del código sin cambios)
loss, accuracy = model.evaluate(validation_dataset, verbose=0)

print(f"\n{'='*50}")
print(f"RESULTADOS FINALES (VOCALES - GRISES)")
print(f"{'='*50}")
print(f"Pérdida (Loss):       {loss:.4f}")
print(f"Exactitud (Accuracy): {accuracy*100:.2f}%")
print(f"{'='*50}")

model_name = f'VOWEL_GRAY_96x96_CustomCNN_FINAL_ACC{accuracy*100:.2f}.keras'
model_path = os.path.join(os.getcwd(), model_name)
model.save(model_path)
print(f"\n💾 Modelo guardado: {model_name}")

# --- 7. GUARDAR HISTORIAL DE ENTRENAMIENTO ---
history_data = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']]
}

with open('training_history_vowel_cnn_grayscale.json', 'w') as f:
    json.dump(history_data, f, indent=4)
print("📊 Historial guardado en: training_history_vowel_cnn_grayscale.json")

print("\n✅ Proceso completado exitosamente.")


# ====================================================================
# --- 8. (OPCIONAL) CÓDIGO DE DIAGNÓSTICO (AJUSTADO PARA GRISES) ---
# ====================================================================

# NOTA: Como el 'train_dataset' ahora contiene imágenes en GRISES,
# este bloque de diagnóstico debe cargar un lote de COLOR por separado
# para poder probar la lógica del Chroma Key (que SÍ usa color).

print("\n🩺 Iniciando Diagnóstico de Chroma Key (Fondo Verde)...")

try:
    # Cargar un lote de IMÁGENES ORIGINALES (en color)
    color_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=VOWEL_CLASSES,
        image_size=IMAGE_SIZE,
        interpolation='bilinear',
        batch_size=BATCH_SIZE, # Cargar un lote
        shuffle=True
    )

    # Combinar con un lote de fondos
    # (background_dataset ya fue creado y es infinito)
    diagnostic_ds = tf.data.Dataset.zip((color_ds, background_dataset.batch(BATCH_SIZE)))

    # Tomar el primer (y único) lote
    for (images_color, labels), backgrounds_color in diagnostic_ds.take(1):
        
        # Tomar la primera imagen y fondo del lote
        img_orig = images_color[0] # Esta SÍ es a color
        bg_orig = backgrounds_color[0] 
        
        # --- Aplicar lógica de Chroma Key ---
        
        # Aplicar el chroma key
        img_blended_rgb = chroma_key_blend(img_orig, bg_orig)
        
        # Aplicar la conversión a grises (para mostrar)
        img_blended_gray = tf.image.rgb_to_grayscale(img_blended_rgb)
        
        # --- Recrear la máscara solo para visualización ---
        img_hsv = tf.image.rgb_to_hsv(img_orig / 255.0)
        h, s, v = tf.split(img_hsv, 3, axis=-1)
        mask_h = tf.logical_and(h >= HSV_GREEN_MIN[0], h <= HSV_GREEN_MAX[0])
        mask_s = tf.logical_and(s >= HSV_GREEN_MIN[1], s <= HSV_GREEN_MAX[1])
        mask_v = tf.logical_and(v >= HSV_GREEN_MIN[2], v <= HSV_GREEN_MAX[2])
        green_mask_bool = tf.logical_and(mask_h, tf.logical_and(mask_s, mask_v))
        hand_mask_bool = tf.logical_not(green_mask_bool)
        hand_mask_float = tf.cast(hand_mask_bool, dtype=tf.float32)
        # --- Fin de recreación de máscara ---
        
        # --- Mostrar resultados ---
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(img_orig.numpy().astype("uint8"))
        plt.title("1. Original (Color)")
        plt.axis("off")
        
        plt.subplot(2, 3, 2)
        plt.imshow(bg_orig.numpy().astype("uint8"))
        plt.title("2. Fondo Aleatorio (Color)")
        plt.axis("off")
        
        plt.subplot(2, 3, 3)
        # .squeeze() elimina el canal '1' para que imshow sepa mostrarlo como gris
        plt.imshow(hand_mask_float.numpy().squeeze(), cmap='gray')
        plt.title("3. Máscara de la Mano (Blanca)")
        plt.axis("off")
        
        plt.subplot(2, 3, 4)
        plt.imshow(img_blended_rgb.numpy().astype("uint8"))
        plt.title("4. Fusión (en Color)")
        plt.axis("off")
        
        plt.subplot(2, 3, 5)
        plt.imshow(img_blended_gray.numpy().squeeze(), cmap='gray') 
        plt.title("5. Resultado Final (Grises)")
        plt.axis("off")
        
        plt.subplot(2, 3, 6)
        plt.axis("off") # Espacio vacío
        
        plt.suptitle("Diagnóstico del Chroma Key (Versión Grises)", fontsize=16)
        plt.show()

except Exception as e:
    print(f"⚠️ Error al generar diagnóstico: {e}")
    print("Asegúrate de que las carpetas 'VOCALES/train' y 'backgrounds' existan.")