import os

# --- Configuración ---
TFLITE_FILE = 'vowel_model_quant_int8.tflite'
HEADER_FILE = 'model_data.h'
ARRAY_NAME = 'vowel_model'
# -------------------

def convert_to_c_array(tflite_path, header_path, array_name):
    print(f"Abriendo archivo binario: {tflite_path}")
    try:
        with open(tflite_path, 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo {tflite_path}")
        return

    array_len = len(data)
    print(f"Archivo leído. Tamaño: {array_len} bytes.")

    print(f"Escribiendo archivo de cabecera: {header_path}...")

    # Crear el array de C
    # Escribimos de 12 en 12 bytes por línea para que sea legible
    bytes_per_line = 12
    try:
        with open(header_path, 'w') as f:
            f.write(f"// Archivo generado automáticamente desde {tflite_path}\n\n")

            # Definición de la longitud
            f.write(f"const unsigned int {array_name}_len = {array_len};\n\n")

            # Definición del array
            f.write(f"const unsigned char {array_name}[] __attribute__((aligned(4))) = {{\n  ")

            for i, byte in enumerate(data):
                f.write(f"0x{byte:02x}, ") # Escribe el byte en formato hex (ej. 0x1c,)

                # Salto de línea cada 12 bytes
                if (i + 1) % bytes_per_line == 0:
                    f.write("\n  ")

            f.write("\n};\n") # Cierra el array

        print(f"\n✅ ¡Éxito! Archivo {header_path} generado.")
        print(f"   Array: {array_name}[]")
        print(f"   Longitud: {array_name}_len")

    except Exception as e:
        print(f"❌ ERROR: No se pudo escribir el archivo {header_path}. {e}")

# --- Ejecutar la conversión ---
if __name__ == "__main__":
    convert_to_c_array(TFLITE_FILE, HEADER_FILE, ARRAY_NAME)