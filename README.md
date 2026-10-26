# Parking Vision (Vigo)

Aplicación de escritorio (**PyQt6**) y web (**Streamlit**) con **YOLOv8 + Shapely** para monitorizar plazas de aparcamiento en tiempo real utilizando la red de 59 cámaras de tráfico del **Concello de Vigo**.

## Estructura del Proyecto

- **`app_parking.py`**: Aplicación principal de escritorio en PyQt6 con hilo en segundo plano (`QThread`), ajuste de umbrales en caliente (`conf` y `UMBRAL_OCUPACION`), edición directa con ratón sobre el vídeo y guardado de capturas en `capturas/`.
- **`app_streamlit.py`**: Aplicación web preparada para ejecutarse localmente o en **Streamlit Cloud**.
- **`detector.py`**: Motor de visión artificial (`AnalizadorParking`) con YOLOv8 (`yolov8m.pt`), cálculo geométrico con Shapely y filtro inteligente anti-semáforos y carriles de circulación.
- **`camara_vigo.py`**: Cliente HTTP (`CamaraVigo`) con el catálogo oficial de las 59 cámaras de Vigo, cabeceras de navegador y enrutamiento de respaldo por pasarelas/proxies en España.
- **`calibrar_plazas.py`**: Herramienta interactiva de calibración de polígonos de 4 puntos.
- **`plazas_vigo.json`**: Archivo de persistencia multicámara con las plazas y zonas excluidas.

## Ejecución en Windows (Escritorio PyQt6)

Puedes hacer doble clic en `iniciar_parking_vision.bat` o ejecutar en terminal:

```powershell
.\.venv\Scripts\python.exe app_parking.py
```

## Ejecución Web (Streamlit)

```powershell
.\.venv\Scripts\streamlit.exe run app_streamlit.py
```
