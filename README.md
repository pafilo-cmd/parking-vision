# Parking Vision (Vigo)

Aplicación de escritorio en **Python + PyQt6 + YOLOv8 + Shapely** para monitorizar plazas de aparcamiento en tiempo real utilizando la red de cámaras de tráfico del **Concello de Vigo**.

## Características principales

- **Catálogo oficial de 59 cámaras de Vigo (`camara_vigo.py`)**: Conexión en tiempo real con `camaras.vigo.org` con parámetro anti-caché, buscador instantáneo de calles y cambio de cámara en caliente.
- **Motor de Visión Artificial (`detector.py`)**:
  - Detección de vehículos con **YOLOv8 (`yolov8m.pt`)** y geometría espacial con **Shapely**.
  - Filtro inteligente anti-semáforos y exclusión automática de carriles de circulación.
  - Ajuste en caliente del umbral de confianza (`conf`, por defecto `0.18`) y del umbral de solapamiento de ocupación (`UMBRAL_OCUPACION`, por defecto `0.25`).
- **Interfaz Gráfica Profesional (`app_parking.py`)**:
  - Procesamiento en segundo plano mediante `QThread` con reconexión automática ante fallos de red.
  - Edición directa sobre el visor de vídeo: **clic izquierdo + arrastrar** para añadir plazas y **clic derecho** sobre una plaza para eliminarla.
  - Botón **📸 Guardar Snapshot** para exportar capturas procesadas a `capturas/`.
- **Calibrador de 4 puntos (`calibrar_plazas.py`)**: Herramienta complementaria con barra de botones superior para delimitar polígonos de 4 esquinas por cámara.

## Instalación y ejecución en Windows

```powershell
# 1. Crear entorno virtual e instalar dependencias
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 2. Lanzar la aplicación principal
.\.venv\Scripts\python.exe app_parking.py
```

También puedes iniciar la aplicación haciendo doble clic en `iniciar_parking_vision.bat`.
