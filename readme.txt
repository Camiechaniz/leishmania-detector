Detector Automático de Leishmania y Macrófagos

Aplicación web desarrollada como parte de nuestra tesis de grado para el análisis automatizado de imágenes microscópicas de Leishmania spp. mediante modelos de detección de objetos.

Objetivo

Desarrollar una herramienta accesible y eficiente que permita cuantificar automáticamente la cantidad de macrófagos y parásitos de Leishmania en imágenes microscópicas, con el fin de evaluar el efecto de diferentes concentraciones de droga a través de curvas dosis-respuesta.

Características Principales

- Detección simultánea de macrófagos y parásitos utilizando dos modelos YOLOv11n entrenados específicamente.
- Soporte para dos tipos de adquisición de imágenes:
  - Imágenes capturadas con cámara de microscopio (rectangulares)
  - Imágenes capturadas con teléfono móvil (generalmente con formato circular)
- Recorte manual o automático de la región de interés
- Corrección de imperfecciones ópticas mediante máscara de hoja blanca (inpainting)
- Detección y exclusión automática de macrófagos que tocan el borde
- Cálculo automático de la tasa de infección (parásitos por 200 macrófagos válidos)
- Generación interactiva de la curva dosis-respuesta
- Exportación completa de resultados (imágenes anotadas, CSV y ZIP)

Tecnologías Utilizadas

- Python 3.12.13
- YOLOv11n (Ultralytics) – Modelos de detección de objetos
- Streamlit – Framework para la interfaz web interactiva
- OpenCV – Procesamiento y análisis de imágenes
- Pandas y Matplotlib – Manejo de datos y visualizaciones

Cómo Usar la Aplicación

1. Acceso
La aplicación se encuentra desplegada en: 
https://leishmania-detector.streamlit.app/ 

2. Flujo de Trabajo Recomendado

1. Ingresar un ID del dataset y la concentración de droga (μM)
2. Subir entre 15 y 20 imágenes microscópicas
3. Seleccionar el origen de las imágenes (Microscopio o Teléfono móvil)
4. Configurar recorte (opcional) y máscara de corrección de imperfecciones
5. Presionar "Procesar todas las imágenes"
6. Revisar los resultados visuales (macrófagos verdes = válidos / rojos = inválidos)
7. Cuando se alcancen ≥ 200 macrófagos válidos, presionar "Agregar al gráfico" para incorporar el punto a la curva dosis-respuesta

Estructura del Repositorio

streamlit_app.py                 # Archivo principal de la aplicación
modelos/
   ├── best_macrofagos.pt        # Modelo YOLOv11n entrenado
   └── best_parasitos.pt
.streamlit/config.toml
requirements.txt
packages.txt
runtime.txt
assets/
   └── FLUJOGRAMA_PAGINA_WEB.jpg
salidas/                         # Carpeta generada automáticamente con resultados
README.txt

Resultados y Visualizaciones

- Imágenes procesadas con anotaciones de color (verde = válido, rojo = descartado)
- Tabla resumen por imagen y por dataset
- Curva dosis-respuesta interactiva
- Historial completo de todos los experimentos procesados
- Descarga de ZIP con todos los archivos generados

Consideraciones Importantes

- Se recomienda procesar entre 15 y 20 imágenes por dataset para alcanzar un mínimo confiable de 200 macrófagos válidos.
- Los macrófagos que tocan el borde o se encuentran dentro de macrófagos inválidos son descartados automáticamente.
- La opción “Sin máscara” para imágenes de microscopio se encuentra aún en fase de desarrollo/experimental.

Autores

Camila Belén Echaniz
Micaela Tajchman
Carrera: Ingeniería Biomédica
Universidad: Facultad de Ciencias Exactas, Físicas y Naturales - UNC
Año: 2026

Trabajo Final de Grado / Tesis

Este proyecto forma parte de un trabajo académico. Su uso está restringido a fines educativos y de investigación.
