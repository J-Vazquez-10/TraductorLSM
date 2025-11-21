/**
 * @file prototipo_corregido.ino
 * @brief Clasificador de Vocales con ESP32-CAM usando TFLite Micro
 * @version 2.0 - Corregido con diagnóstico completo
 * 
 * CORRECCIONES PRINCIPALES:
 * - Constructor del intérprete corregido (5 argumentos)
 * - Diagnóstico detallado de memoria
 * - Validación exhaustiva de errores
 * - Mensajes de debug mejorados
 */

// --- LIBRERÍAS ESENCIALES ---
#include "esp_camera.h"

// LCD I2C
#include <Wire.h> 
#include <LiquidCrystal_I2C.h>

// TensorFlow Lite Micro
#include "micro_interpreter.h" 
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h" 
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

#include "model_data.h" // Tu archivo del modelo

// --- CONFIGURACIÓN DE PINES Y PERIFÉRICOS ---

// Configuración I2C para el LCD
#define SDA_PIN 13 
#define SCL_PIN 14 
#define I2C_ADDR 0x27 
LiquidCrystal_I2C lcd(I2C_ADDR, 16, 2);

// Configuración de Pines de la Cámara (WROVER-KIT)
#define PWDN_GPIO_NUM -1 
#define RESET_GPIO_NUM -1 
#define XCLK_GPIO_NUM 21
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 19
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 5
#define Y2_GPIO_NUM 4
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// --- TFLITE y VARIABLES GLOBALES ---
namespace {
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    constexpr int kTensorArenaSize = 1500 * 1024; // 120KB - Ajustar si es necesario
    uint8_t *tensor_arena = nullptr;
}

const char* VOWEL_CLASSES[] = {"A", "E", "I", "O", "U"};

const long captureInterval = 20000; // 20 segundos
unsigned long lastCaptureTime = 0;
String lastPrediction = "Iniciando..."; 

// --- FUNCIONES ---

/**
 * @brief Inicializa la cámara con las configuraciones deseadas.
 */
bool init_camera() {
    camera_config_t config;
    
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE; 
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    
    if(psramFound()){
        config.fb_location = CAMERA_FB_IN_PSRAM;
    } else {
        config.fb_location = CAMERA_FB_IN_DRAM;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("  ✗ Error al inicializar cámara: 0x%x\n", err);
        return false;
    }
    
    return true;
}

/**
 * @brief Ejecuta la inferencia del modelo con una imagen capturada
 */
void run_inference() {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
        lcd.setCursor(0, 1);
        lcd.print("Error Captura ");
        Serial.println("ERROR: Falló captura de imagen");
        return;
    }

    if (fb->width != 96 || fb->height != 96 || fb->format != PIXFORMAT_GRAYSCALE) {
        lcd.setCursor(0, 1);
        lcd.print("Error Formato   ");
        Serial.printf("ERROR: Formato incorrecto - W:%d H:%d Fmt:%d\n", 
                      fb->width, fb->height, fb->format);
        esp_camera_fb_return(fb);
        return;
    }

    // --- PREPROCESAMIENTO DINÁMICO ---
    // Usamos los parámetros de calibración que el modelo trae dentro
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;

    // Verificar cuántos canales espera el modelo
    int num_channels = 1; 
    if (input->dims->size == 4) { // Formato [Batch, Height, Width, Channels]
       num_channels = input->dims->data[3];
    }

    int8_t* input_data = input->data.int8;

    // Bucle para procesar la imagen
    for (int i = 0; i < (96 * 96); i++) {
        // 1. Obtener pixel crudo de la cámara (0 a 255)
        float pixel = (float)fb->buf[i]; 
    
        // 2. Convertir usando la fórmula oficial de TensorFlow:
        //    Valor_Quantizado = (Valor_Real / Scale) + Zero_Point
        int32_t val = (int32_t)(pixel / input_scale) + input_zero_point;

        // 3. Asegurar que no nos salimos del rango -128 a 127 (Clamping)
        if (val < -128) val = -128;
        if (val > 127) val = 127;

        // 4. Llenar el tensor. 
        // SI el modelo espera RGB (3 canales) pero tenemos Grises (1 canal),
        // repetimos el mismo pixel 3 veces.
        if (num_channels == 3) {
            int index = i * 3;
            input_data[index] = (int8_t)val;     // R
            input_data[index + 1] = (int8_t)val; // G
            input_data[index + 2] = (int8_t)val; // B
        } else {
        // Caso normal: Grayscale
            input_data[i] = (int8_t)val;
        }
    }
    // ----------------------------------------
    
    esp_camera_fb_return(fb);
    Serial.println("Imagen preprocesada (96x96 -> int8)");

    // Ejecutar inferencia
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        lcd.setCursor(0, 1);
        lcd.print("Error Invoke    ");
        Serial.println("ERROR: Falló la inferencia");
        return;
    }
    
    Serial.println("Inferencia ejecutada exitosamente");

    // Procesar resultados
    int8_t* output_data = output->data.int8;
    int8_t best_score = -128;
    int best_index = -1;

    // Definimos el umbral de confianza (Ajustable: -128 a 127)
    // 20 es un buen punto de partida para filtrar ruido
    const int8_t CONFIDENCE_THRESHOLD = 20; 

    Serial.print("Scores: ");
    for (int i = 0; i < 5; i++) {
        Serial.printf("%s:%d ", VOWEL_CLASSES[i], output_data[i]);
        
        if (output_data[i] > best_score) {
            best_score = output_data[i];
            best_index = i;
        }
    }
    Serial.println();

    // --- LÓGICA DE FILTRADO MEJORADA ---
    if (best_index >= 0 && best_score > CONFIDENCE_THRESHOLD) {
        // CALCULO DE CONFIANZA (0% a 100%) basado en int8
        float confidence = ((float)(best_score + 128) / 255.0) * 100.0;
        
        // Formatear texto para LCD
        lastPrediction = "Pred: " + String(VOWEL_CLASSES[best_index]) + " (" + String((int)confidence) + "%)"; 
        
        lcd.setCursor(0, 0);
        lcd.print(lastPrediction);
        // Imprimir espacios extra para borrar caracteres viejos si el texto es corto
        lcd.print("   "); 
        
        Serial.printf(">>> PREDICCIÓN VÁLIDA: %s (Score: %d, Confianza: %.1f%%)\n\n", 
                      VOWEL_CLASSES[best_index], best_score, confidence);
    } else {
        // Si el score es muy bajo (ej. ruido o la "I" falsa de -40)
        lcd.setCursor(0, 0);
        lcd.print("Incierto...     "); // Feedback visual de "no sé qué es"
        
        Serial.printf(">>> PREDICCIÓN DESCARTADA: %s (Score %d es muy bajo)\n\n", 
                      (best_index >= 0 ? VOWEL_CLASSES[best_index] : "N/A"), best_score);
    }
}

// --- FUNCIÓN SETUP ---
void setup() {
    // ============================================
    // 1. INICIALIZAR SERIAL CON DIAGNÓSTICO
    // ============================================
    Serial.begin(115200);
    delay(1000); // Esperar estabilización
    
    Serial.println();
    Serial.println();
    Serial.println("===========================================");
    Serial.println("   ESP32-CAM CLASIFICADOR DE VOCALES");
    Serial.println("   TensorFlow Lite Micro - DEBUG MODE");
    Serial.println("===========================================");
    Serial.printf("Reset reason: %d\n", esp_reset_reason());
    Serial.printf("CPU Freq: %d MHz\n", ESP.getCpuFreqMHz());
    Serial.println();

    // ============================================
    // 2. INICIALIZAR LCD
    // ============================================
    Serial.println("[1/7] Inicializando LCD...");
    Wire.begin(SDA_PIN, SCL_PIN);
    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("Iniciando...");
    lcd.setCursor(0, 1);
    lcd.print("Serial OK");
    Serial.println("  ✓ LCD inicializado");
    delay(1000);

    // ============================================
    // 3. VERIFICAR PSRAM
    // ============================================
    Serial.println("\n[2/7] Verificando memoria...");
    lcd.setCursor(0, 1);
    lcd.print("Check Memory...");
    
    if (psramFound()) {
        Serial.printf("  ✓ PSRAM detectada\n");
        Serial.printf("  - Total: %u bytes (%.2f MB)\n", 
                      ESP.getPsramSize(), 
                      ESP.getPsramSize() / 1048576.0);
        Serial.printf("  - Libre: %u bytes (%.2f MB)\n", 
                      ESP.getFreePsram(),
                      ESP.getFreePsram() / 1048576.0);
    } else {
        Serial.println("  ✗ ADVERTENCIA: PSRAM no detectada");
        Serial.println("    El sistema usará DRAM (limitado)");
    }
    
    Serial.printf("  Heap libre: %u bytes\n", ESP.getFreeHeap());

    // ============================================
    // 4. INICIALIZAR CÁMARA
    // ============================================
    Serial.println("\n[3/7] Inicializando cámara...");
    lcd.setCursor(0, 1);
    lcd.print("Init Camera...  ");
    
    if (!init_camera()) {
        Serial.println("  ✗ ERROR FATAL: No se pudo inicializar cámara");
        lcd.clear();
        lcd.print("Error Camera!");
        while(1) {
            delay(1000);
        }
    }
    Serial.println("  ✓ Cámara configurada (96x96 GRAY)");

    // ============================================
    // 5. CARGAR MODELO TFLITE
    // ============================================
    Serial.println("\n[4/7] Cargando modelo TensorFlow Lite...");
    lcd.setCursor(0, 1);
    lcd.print("Load Model...   ");
    
    static tflite::MicroErrorReporter static_error_reporter;
    error_reporter = &static_error_reporter;
    
    model = tflite::GetModel(vowel_model);
    
    Serial.printf("  Tamaño: %u bytes (%.2f KB)\n", 
                  vowel_model_len, 
                  vowel_model_len / 1024.0);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("  ✗ ERROR: Versión incompatible\n");
        Serial.printf("    Modelo: v%d\n", model->version());
        Serial.printf("    Schema: v%d\n", TFLITE_SCHEMA_VERSION);
        lcd.clear();
        lcd.print("Error Version!");
        while(1) delay(1000);
    }
    Serial.println("  ✓ Modelo cargado y validado");

    // ============================================
    // 6. ASIGNAR MEMORIA PARA TENSOR ARENA
    // ============================================
    Serial.println("\n[5/7] Asignando memoria para tensores...");
    lcd.setCursor(0, 1);
    lcd.print("Alloc Memory... ");
    
    Serial.printf("  Solicitando: %d KB\n", kTensorArenaSize / 1024);
    
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    
    if (tensor_arena == nullptr) {
        Serial.println("  ✗ ERROR: Falló asignación en PSRAM");
        Serial.println("  Intentando con DRAM...");
        
        tensor_arena = (uint8_t*) malloc(kTensorArenaSize);
        
        if (tensor_arena == nullptr) {
            Serial.println("  ✗ ERROR FATAL: Sin memoria disponible");
            lcd.clear();
            lcd.print("Out of Memory!");
            while(1) delay(1000);
        }
        
        Serial.println("  ⚠ Arena asignada en DRAM (limitado)");
    } else {
        Serial.println("  ✓ Arena asignada en PSRAM");
    }
    
    Serial.printf("  Memoria post-asignación:\n");
    Serial.printf("    - PSRAM libre: %u bytes\n", ESP.getFreePsram());
    Serial.printf("    - Heap libre: %u bytes\n", ESP.getFreeHeap());

    // ============================================
    // 7. CREAR INTÉRPRETE - CONSTRUCTOR CORREGIDO
    // ============================================
    Serial.println("\n[6/7] Creando intérprete TFLite...");
    lcd.setCursor(0, 1);
    lcd.print("Create Interp...");
    
    // Usar AllOpsResolver (incluye todas las operaciones)
    static tflite::AllOpsResolver resolver;
    
    // CONSTRUCTOR DE 5 ARGUMENTOS (COMPATIBLE)
    static tflite::MicroInterpreter static_interpreter(
        model, 
        resolver, 
        tensor_arena, 
        kTensorArenaSize
    );
    interpreter = &static_interpreter;
    Serial.println("  ✓ Intérprete creado");

    // ============================================
    // 8. ASIGNAR TENSORES (PUNTO CRÍTICO)
    // ============================================
    Serial.println("\n[7/7] Asignando tensores...");
    lcd.setCursor(0, 1);
    lcd.print("Alloc Tensors...");
    
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    
    if (allocate_status != kTfLiteOk) {
        Serial.println("  ✗✗✗ ERROR FATAL: AllocateTensors() FALLÓ ✗✗✗");
        Serial.printf("    Status code: %d\n", allocate_status);
        Serial.printf("    Arena necesaria: %zu bytes (%.2f KB)\n", 
                      interpreter->arena_used_bytes(),
                      interpreter->arena_used_bytes() / 1024.0);
        Serial.printf("    Arena disponible: %d bytes (%.2f KB)\n",
                      kTensorArenaSize,
                      kTensorArenaSize / 1024.0);
        
        lcd.clear();
        lcd.print("Tensor Error!");
        lcd.setCursor(0, 1);
        lcd.printf("Need %zuKB", interpreter->arena_used_bytes() / 1024);
        
        while(1) delay(1000);
    }
    
    Serial.println("  ✓ Tensores asignados exitosamente");
    Serial.printf("    Arena usada: %zu / %d bytes (%.1f%%)\n", 
                  interpreter->arena_used_bytes(),
                  kTensorArenaSize,
                  (float)interpreter->arena_used_bytes() / kTensorArenaSize * 100);
    
    // ============================================
    // 9. OBTENER TENSORES DE ENTRADA/SALIDA
    // ============================================
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    if (input == nullptr || output == nullptr) {
        Serial.println("  ✗ ERROR: Tensores I/O son nullptr");
        while(1) delay(1000);
    }
    
    Serial.println("\n  Información de tensores:");
    Serial.printf("    Input:  dims=%d, type=%d, bytes=%zu\n",
                  input->dims->size,
                  input->type,
                  input->bytes);
    Serial.printf("    Output: dims=%d, type=%d, bytes=%zu\n",
                  output->dims->size,
                  output->type,
                  output->bytes);

    // ============================================
    // 10. SISTEMA LISTO
    // ============================================
    lcd.clear();
    lcd.print("Sistema Listo!");
    Serial.println("\n===========================================");
    Serial.println("  ✓✓✓ SISTEMA COMPLETAMENTE OPERATIVO ✓✓✓");
    Serial.println("  Captura automática cada 20 segundos");
    Serial.println("===========================================\n");
    
    lastCaptureTime = millis();
    delay(2000);
    
    lcd.clear();
    lcd.print("Esperando...");
}

// --- FUNCIÓN LOOP ---
void loop() {
    // Verificación de seguridad
    if (!interpreter || !input || !output) {
        delay(100);
        return;
    }

    unsigned long currentTime = millis();

    long timeRemaining = (lastCaptureTime + captureInterval) - currentTime;
    int secondsRemaining = max(0, (int)(timeRemaining / 1000));

    lcd.setCursor(0, 1);
    String countdown = "Captura: " + String(secondsRemaining) + "s   ";
    lcd.print(countdown);

    if (currentTime - lastCaptureTime >= captureInterval) {
        Serial.println("\n>>> NUEVA CAPTURA <<<");
        lastCaptureTime = currentTime;
        run_inference();
    }

    delay(100);
}