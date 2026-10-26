"""
Aplicación web "Parking Vision" desarrollada con Streamlit, preparada para Streamlit Cloud.
Reutiliza íntegramente la lógica de captura (`camara_vigo.py`) y el motor de visión artificial
YOLOv8 + Shapely con filtro anti-semáforos (`detector.py`).
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from camara_vigo import CamaraVigo, obtener_catalogo_camaras, obtener_nombre_camara
from detector import AnalizadorParking

RUTA_CONFIG = Path("plazas_vigo.json")
RUTA_MODELO = "yolov8m.pt"


# =====================================================================
# Configuración de página de Streamlit
# =====================================================================
st.set_page_config(
    page_title="Parking Vision - Concello de Vigo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados para dar aspecto de dashboard profesional
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: #161922;
            border: 1px solid #2b3140;
            border-radius: 10px;
            padding: 12px 16px;
        }
        .cam-banner {
            background-color: #1e222d;
            border-left: 4px solid #3b82f6;
            border-radius: 6px;
            padding: 10px 16px;
            margin-bottom: 12px;
            font-size: 1.05rem;
            font-weight: 600;
        }
        .log-box {
            background-color: #0f1116;
            border: 1px solid #282c37;
            border-radius: 8px;
            padding: 10px 14px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.84rem;
            color: #c5cddb;
            max-height: 240px;
            overflow-y: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Funciones en caché y gestión de estado (Session State)
# =====================================================================
@st.cache_resource(show_spinner="Cargando modelo YOLOv8 (yolov8m.pt)...")
def obtener_analizador_cache() -> AnalizadorParking:
    """
    Instancia y mantiene en caché el motor `AnalizadorParking` para no recargar
    los pesos de YOLOv8 en cada interacción del usuario en Streamlit Cloud.
    """
    return AnalizadorParking(
        ruta_config=RUTA_CONFIG,
        ruta_modelo=RUTA_MODELO,
        cam_id="05",
        conf_threshold=0.18,
        overlap_threshold=0.25,
    )


@st.cache_data(ttl=3600)
def cargar_catalogo() -> dict[str, str]:
    """Devuelve el catálogo oficial de las 59 cámaras de tráfico de Vigo."""
    return obtener_catalogo_camaras()


def inicializar_estado() -> None:
    """Inicializa las variables de sesión de Streamlit."""
    if "logs" not in st.session_state:
        st.session_state.logs = [
            f"[{datetime.now().strftime('%H:%M:%S')}] Aplicación web Parking Vision iniciada."
        ]
    if "cam_id_actual" not in st.session_state:
        st.session_state.cam_id_actual = "05"
    if "cliente_camara" not in st.session_state:
        st.session_state.cliente_camara = CamaraVigo(
            cam_id="05",
            scale_factor=2.0,
            timeout=10.0,
            max_retries=3,
        )
    if "ultimo_jpg_bytes" not in st.session_state:
        st.session_state.ultimo_jpg_bytes = None
    if "ultima_img_rgb" not in st.session_state:
        st.session_state.ultima_img_rgb = None
    if "metricas" not in st.session_state:
        st.session_state.metricas = {
            "libres": 0,
            "total": 0,
            "ocupadas": 0,
            "vehiculos": 0,
            "observacion": 0,
            "hora": "--:--:--",
        }


def registrar_log(mensaje: str) -> None:
    """Añade una entrada con marca de tiempo al registro de actividad."""
    marca = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{marca}] {mensaje}")
    # Mantener las últimas 60 entradas
    if len(st.session_state.logs) > 60:
        st.session_state.logs = st.session_state.logs[-60:]


def obtener_camaras_con_plazas() -> set[str]:
    """Consulta qué cámaras tienen plazas guardadas en `plazas_vigo.json`."""
    calibradas: set[str] = set()
    if not RUTA_CONFIG.exists():
        return calibradas
    try:
        datos = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
        if datos.get("cam_id") and datos.get("plazas"):
            calibradas.add(str(datos["cam_id"]))
        camaras_dict = datos.get("camaras", {})
        if isinstance(camaras_dict, dict):
            for cid, info in camaras_dict.items():
                if isinstance(info, dict) and info.get("plazas"):
                    calibradas.add(str(cid))
    except Exception:
        pass
    return calibradas


def capturar_y_analizar(
    analizador: AnalizadorParking,
    cam_id: str,
    conf_yolo: float,
    umbral_ocupacion: float,
    mostrar_cajas: bool,
    auto_calibrar: bool,
) -> bool:
    """
    Descarga un fotograma en vivo de la cámara de Vigo indicada, lo procesa con
    `AnalizadorParking` y actualiza el estado de la sesión.
    """
    if analizador.cam_id != cam_id:
        analizador.cambiar_camara(cam_id)
        registrar_log(f"Cambiada cámara activa a [{cam_id}] {obtener_nombre_camara(cam_id)}.")

    analizador.actualizar_umbrales(conf_yolo, umbral_ocupacion)
    analizador.dibujar_detecciones = mostrar_cajas
    analizador.auto_calibrar = auto_calibrar

    cam: CamaraVigo = st.session_state.cliente_camara
    cam.cambiar_camara(cam_id, scale_factor=analizador.scale_factor)
    frame_bgr = cam.obtener_frame()

    if frame_bgr is None:
        registrar_log(
            f"⚠️ [RED] Sin respuesta temporal de la cámara [{cam_id}] "
            f"{obtener_nombre_camara(cam_id)}."
        )
        return False

    frame_proc, libres = analizador.procesar_frame(frame_bgr)
    total = analizador.total_plazas
    ocupadas = max(0, total - libres)
    vehiculos = analizador.vehiculos_detectados
    en_obs = analizador.candidatos_en_observacion

    # Convertir a RGB para mostrar en Streamlit y a JPEG para el botón de descarga
    img_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
    ok_jpg, buffer_jpg = cv2.imencode(".jpg", frame_proc, [int(cv2.IMWRITE_JPEG_QUALITY), 93])

    st.session_state.ultima_img_rgb = img_rgb
    if ok_jpg:
        st.session_state.ultimo_jpg_bytes = buffer_jpg.tobytes()

    hora_str = datetime.now().strftime("%H:%M:%S")
    st.session_state.metricas = {
        "libres": libres,
        "total": total,
        "ocupadas": ocupadas,
        "vehiculos": vehiculos,
        "observacion": en_obs,
        "hora": hora_str,
    }

    registrar_log(
        f"[{cam_id}] {obtener_nombre_camara(cam_id)} -> "
        f"Libres: {libres}/{total} | Vehículos en vía: {vehiculos}"
    )
    return True


# =====================================================================
# Interfaz Principal de Usuario
# =====================================================================
def main() -> None:
    inicializar_estado()
    analizador = obtener_analizador_cache()
    catalogo = cargar_catalogo()
    calibradas = obtener_camaras_con_plazas()

    # -----------------------------------------------------------------
    # BARRA LATERAL: Selección de cámara y parámetros en caliente
    # -----------------------------------------------------------------
    with st.sidebar:
        st.title("🚗 Parking Vision")
        st.caption("Monitorización de aparcamiento en tiempo real • Concello de Vigo")
        st.divider()

        st.subheader("📍 Cámara de Tráfico")
        filtro_texto = st.text_input(
            "Buscar calle o ubicación",
            placeholder="Ej: Castelao, Samil, Colón, 24...",
        ).strip().lower()

        opciones_ids: list[str] = []
        etiquetas_cam: dict[str, str] = {}

        for cid, ubicacion in sorted(catalogo.items(), key=lambda x: (len(x[0]), x[0])):
            if filtro_texto and filtro_texto not in f"{cid} {ubicacion}".lower():
                continue
            marca = " ★" if cid in calibradas else ""
            opciones_ids.append(cid)
            etiquetas_cam[cid] = f"[{cid}] {ubicacion}{marca}"

        if not opciones_ids:
            st.warning("No hay cámaras que coincidan con la búsqueda. Mostrando todas.")
            for cid, ubicacion in sorted(catalogo.items(), key=lambda x: (len(x[0]), x[0])):
                opciones_ids.append(cid)
                etiquetas_cam[cid] = f"[{cid}] {ubicacion}"

        idx_defecto = (
            opciones_ids.index(st.session_state.cam_id_actual)
            if st.session_state.cam_id_actual in opciones_ids
            else 0
        )

        cam_seleccionada = st.selectbox(
            "Selecciona una cámara de Vigo",
            options=opciones_ids,
            index=idx_defecto,
            format_func=lambda cid: etiquetas_cam.get(cid, cid),
        )
        st.session_state.cam_id_actual = cam_seleccionada

        st.divider()
        st.subheader("⚙️ Parámetros en Caliente")

        conf_yolo = st.slider(
            "Umbral de confianza YOLO (conf)",
            min_value=0.05,
            max_value=0.90,
            value=0.18,
            step=0.01,
            help="Confianza mínima para detectar vehículos a media distancia (por defecto 0.18).",
        )

        umbral_ocupacion = st.slider(
            "Umbral de solapamiento (UMBRAL_OCUPACION)",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Porcentaje mínimo de área de plaza cubierta por un vehículo para marcarla ocupada (0.25 = 25%).",
        )

        mostrar_cajas = st.checkbox(
            "Mostrar vehículos detectados (YOLOv8)",
            value=True,
        )
        auto_calibrar = st.checkbox(
            "✨ Auto-aprender plazas (filtro anti-semáforos)",
            value=True,
        )

        st.divider()
        st.subheader("🔄 Refresco Automático")
        auto_refresco = st.toggle("Monitorización continua en vivo", value=False)
        intervalo_seg = st.slider(
            "Intervalo de actualización (segundos)",
            min_value=3,
            max_value=30,
            value=6,
            step=1,
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            btn_actualizar = st.button("🔄 Capturar ahora", use_container_width=True, type="primary")
        with col_btn2:
            btn_recalcular = st.button("✨ Re-escanear IA", use_container_width=True)

        if btn_recalcular:
            analizador.cambiar_camara(cam_seleccionada)
            analizador.reiniciar_autocalibracion_camara()
            registrar_log(
                f"✨ Aprendizaje IA reiniciado en [{cam_seleccionada}] "
                f"{obtener_nombre_camara(cam_seleccionada)}."
            )

        # Gestión rápida de plazas desde la web (eliminar plaza por ID si alguna sobra)
        if analizador.total_plazas > 0:
            with st.expander("🛠️ Ajustar plazas detectadas"):
                ids_actuales = [p.id for p in analizador.plazas]
                plaza_a_borrar = st.selectbox(
                    "Eliminar plaza mal situada",
                    options=ids_actuales,
                    format_func=lambda pid: f"Plaza P{pid}",
                )
                if st.button("🗑️ Eliminar plaza seleccionada", use_container_width=True):
                    for p in list(analizador.plazas):
                        if p.id == plaza_a_borrar:
                            cx = int(sum(pt[0] for pt in p.puntos) / len(p.puntos))
                            cy = int(sum(pt[1] for pt in p.puntos) / len(p.puntos))
                            analizador.eliminar_plaza_en_punto(cx, cy)
                            registrar_log(f"🗑️ Plaza P{plaza_a_borrar} eliminada por el usuario.")
                            break
                    st.rerun()

    # -----------------------------------------------------------------
    # EJECUCIÓN DE CAPTURA Y ANÁLISIS
    # -----------------------------------------------------------------
    debe_capturar = (
        btn_actualizar
        or btn_recalcular
        or auto_refresco
        or st.session_state.ultima_img_rgb is None
        or analizador.cam_id != cam_seleccionada
    )

    exito_red = True
    if debe_capturar:
        with st.spinner(
            f"Conectando con cámara [{cam_seleccionada}] {obtener_nombre_camara(cam_seleccionada)}..."
        ):
            exito_red = capturar_y_analizar(
                analizador=analizador,
                cam_id=cam_seleccionada,
                conf_yolo=conf_yolo,
                umbral_ocupacion=umbral_ocupacion,
                mostrar_cajas=mostrar_cajas,
                auto_calibrar=auto_calibrar,
            )

    # -----------------------------------------------------------------
    # PANEL PRINCIPAL: Métricas, Visor de Vídeo, Snapshot y Logs
    # -----------------------------------------------------------------
    ubicacion_actual = obtener_nombre_camara(cam_seleccionada)
    st.markdown(
        f'<div class="cam-banner">📍 Cámara activa: [{cam_seleccionada}] {ubicacion_actual} '
        f'• Última actualización: {st.session_state.metricas["hora"]}</div>',
        unsafe_allow_html=True,
    )

    if not exito_red:
        st.warning(
            f"⚠️ No se pudo obtener respuesta de la cámara **[{cam_seleccionada}] {ubicacion_actual}** "
            "en este instante. La aplicación conservará el último estado y reintentará en el siguiente ciclo."
        )

    # Tarjetas de métricas superiores
    m = st.session_state.metricas
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        texto_libres = (
            f"{m['libres']} / {m['total']}" if m["total"] > 0 else "Verificando..."
        )
        st.metric("🟢 Plazas Libres / Confirmadas", texto_libres)
    with col_m2:
        st.metric("🔴 Plazas Ocupadas", f"{m['ocupadas']}")
    with col_m3:
        st.metric("🚙 Vehículos en Vía (YOLOv8)", f"{m['vehiculos']}")
    with col_m4:
        st.metric("⏱️ En Observación (Anti-Semáforo)", f"{m['observacion']}")

    # Visor de vídeo principal + Columna lateral de acciones y registro
    col_video, col_lateral = st.columns([7, 3])

    with col_video:
        if st.session_state.ultima_img_rgb is not None:
            st.image(
                st.session_state.ultima_img_rgb,
                caption=(
                    f"Cámara [{cam_seleccionada}] {ubicacion_actual} "
                    f"(conf={conf_yolo:.2f}, solapamiento={umbral_ocupacion:.2f})"
                ),
                use_container_width=True,
            )
        else:
            st.info("Pulsa «🔄 Capturar ahora» para obtener el primer fotograma de la cámara.")

    with col_lateral:
        st.subheader("📸 Captura de Pantalla")
        if st.session_state.ultimo_jpg_bytes is not None:
            marca_archivo = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_descarga = f"snapshot_cam{cam_seleccionada}_{marca_archivo}.jpg"

            st.download_button(
                label="📥 Descargar Snapshot (JPG)",
                data=st.session_state.ultimo_jpg_bytes,
                file_name=nombre_descarga,
                mime="image/jpeg",
                use_container_width=True,
            )

            if st.button("💾 Guardar en carpeta local capturas/", use_container_width=True):
                try:
                    carpeta = Path("capturas")
                    carpeta.mkdir(parents=True, exist_ok=True)
                    ruta_archivo = carpeta / nombre_descarga
                    ruta_archivo.write_bytes(st.session_state.ultimo_jpg_bytes)
                    st.success(f"Guardado en `capturas/{nombre_descarga}`")
                    registrar_log(f"📸 Snapshot guardado en capturas/{nombre_descarga}")
                except OSError as exc:
                    st.error(f"Error al guardar captura: {exc}")

        st.divider()
        st.subheader("📋 Registro de Actividad")
        logs_invertidos = "<br>".join(reversed(st.session_state.logs[-25:]))
        st.markdown(f'<div class="log-box">{logs_invertidos}</div>', unsafe_allow_html=True)

        if st.button("Limpiar registro", use_container_width=True):
            st.session_state.logs.clear()
            st.rerun()

    # Si está activo el modo de monitorización continua, esperar y refrescar
    if auto_refresco:
        time.sleep(intervalo_seg)
        st.rerun()


if __name__ == "__main__":
    main()
