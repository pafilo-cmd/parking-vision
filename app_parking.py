"""
Aplicación principal de escritorio "Parking Vision" en PyQt6.
Integra el catálogo completo de cámaras de tráfico del Concello de Vigo (`camara_vigo.py`),
el motor YOLOv8 + Shapely con filtro anti-semáforos (`detector.py`),
ajuste de umbrales en caliente, captura de snapshots y reconexión automática ante fallos de red.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QCloseEvent, QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from camara_vigo import CamaraVigo, obtener_catalogo_camaras, obtener_nombre_camara
from detector import AnalizadorParking


class HiloMonitorizacion(QThread):
    """
    Hilo de trabajo (`QThread`) resiliente que ejecuta en segundo plano:
    1. La descarga periódica de fotogramas de las cámaras de Vigo con reintento automático.
    2. El motor YOLOv8 + Shapely con actualización en caliente de `conf` y `UMBRAL_OCUPACION`.
    3. La edición en vivo de plazas sobre el visor de vídeo.
    """

    # (imagen_qt, plazas_libres, total_plazas, vehiculos_via, en_observacion, cam_id, modo_cal)
    frame_listo = pyqtSignal(QImage, int, int, int, int, str, str)
    log_mensaje = pyqtSignal(str)
    aviso_red = pyqtSignal(str, int)  # (mensaje_aviso, numero_fallos_consecutivos)
    estado_error = pyqtSignal(str)

    def __init__(
        self,
        cam_id: str = "05",
        ruta_config: str | Path = "plazas_vigo.json",
        ruta_modelo: str | Path = "yolov8m.pt",
        intervalo_segundos: float = 3.0,
        conf_yolo: float = 0.18,
        umbral_ocupacion: float = 0.25,
        mostrar_cajas_yolo: bool = True,
        auto_calibrar: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.cam_id_actual = str(cam_id)
        self._cam_id_pendiente: Optional[str] = None
        self._recargar_config_pendiente: bool = False
        self._recalcular_ia_pendiente: bool = False
        self._umbrales_modificados: bool = False
        self._plazas_manuales_pendientes: List[List[Tuple[int, int]]] = []
        self._puntos_eliminar_pendientes: List[Tuple[int, int]] = []

        self.ruta_config = Path(ruta_config)
        self.ruta_modelo = str(ruta_modelo)
        self.intervalo_segundos = max(0.5, float(intervalo_segundos))
        self.conf_yolo = float(conf_yolo)
        self.umbral_ocupacion = float(umbral_ocupacion)
        self.mostrar_cajas_yolo = mostrar_cajas_yolo
        self.auto_calibrar = auto_calibrar

        self._ejecutando = False
        self._analizador: Optional[AnalizadorParking] = None
        self._ultimo_frame_bgr: Optional[np.ndarray] = None

    def solicitar_cambio_camara(self, nuevo_cam_id: str) -> None:
        self._cam_id_pendiente = str(nuevo_cam_id)

    def solicitar_recarga_plazas(self) -> None:
        self._recargar_config_pendiente = True

    def solicitar_autocalibracion_ia(self) -> None:
        self._recalcular_ia_pendiente = True

    def solicitar_agregar_plaza(self, puntos: List[Tuple[int, int]]) -> None:
        self._plazas_manuales_pendientes.append(puntos)

    def solicitar_eliminar_plaza(self, x: int, y: int) -> None:
        self._puntos_eliminar_pendientes.append((x, y))

    def actualizar_opciones(
        self,
        intervalo_segundos: float,
        conf_yolo: float,
        umbral_ocupacion: float,
        mostrar_cajas_yolo: bool,
        auto_calibrar: bool,
    ) -> None:
        """
        Actualiza en caliente los parámetros de inferencia y visualización sin detener el hilo.
        Si cambian los umbrales o la visibilidad de cajas, despierta el hilo para refrescar al instante.
        """
        self.intervalo_segundos = max(0.5, float(intervalo_segundos))

        cambio_visual = (
            abs(self.conf_yolo - conf_yolo) > 1e-4
            or abs(self.umbral_ocupacion - umbral_ocupacion) > 1e-4
            or self.mostrar_cajas_yolo != mostrar_cajas_yolo
        )

        self.conf_yolo = float(conf_yolo)
        self.umbral_ocupacion = float(umbral_ocupacion)
        self.mostrar_cajas_yolo = mostrar_cajas_yolo
        self.auto_calibrar = auto_calibrar

        if self._analizador is not None:
            self._analizador.actualizar_umbrales(self.conf_yolo, self.umbral_ocupacion)
            self._analizador.dibujar_detecciones = mostrar_cajas_yolo
            self._analizador.auto_calibrar = auto_calibrar

        if cambio_visual:
            self._umbrales_modificados = True

    def detener(self) -> None:
        self._ejecutando = False

    def run(self) -> None:
        self._ejecutando = True

        try:
            self.log_mensaje.emit(
                f"Cargando modelo '{self.ruta_modelo}' (conf={self.conf_yolo:.2f}, "
                f"solapamiento={self.umbral_ocupacion:.2f})..."
            )
            self._analizador = AnalizadorParking(
                ruta_config=self.ruta_config,
                ruta_modelo=self.ruta_modelo,
                cam_id=self.cam_id_actual,
                conf_threshold=self.conf_yolo,
                overlap_threshold=self.umbral_ocupacion,
                dibujar_detecciones=self.mostrar_cajas_yolo,
                auto_calibrar=self.auto_calibrar,
            )
        except Exception as exc:
            self.estado_error.emit(f"Error crítico al inicializar el detector: {exc}")
            self._ejecutando = False
            return

        self._emitir_resumen_camara()

        with CamaraVigo(
            cam_id=self.cam_id_actual,
            scale_factor=self._analizador.scale_factor,
        ) as camara:
            ultimo_estado_log: Optional[tuple[str, int, int]] = None
            fallos_red_consecutivos: int = 0

            while self._ejecutando:
                # 1. ¿Cambió de cámara el usuario?
                if self._cam_id_pendiente is not None:
                    nuevo_id = self._cam_id_pendiente
                    self._cam_id_pendiente = None
                    if nuevo_id != self.cam_id_actual:
                        self.cam_id_actual = nuevo_id
                        self._ultimo_frame_bgr = None
                        fallos_red_consecutivos = 0
                        self._analizador.cambiar_camara(nuevo_id)
                        camara.cambiar_camara(nuevo_id, self._analizador.scale_factor)
                        ultimo_estado_log = None
                        self._emitir_resumen_camara()

                # 2. ¿Pulsó "Reiniciar aprendizaje IA"?
                if self._recalcular_ia_pendiente:
                    self._recalcular_ia_pendiente = False
                    self._analizador.auto_calibrar = True
                    self._analizador.reiniciar_autocalibracion_camara()
                    ultimo_estado_log = None
                    self.log_mensaje.emit(
                        f"✨ Aprendizaje IA reiniciado en [{self.cam_id_actual}] "
                        f"{obtener_nombre_camara(self.cam_id_actual)}."
                    )

                # 3. ¿Guardó cambios desde el calibrador manual externo?
                if self._recargar_config_pendiente:
                    self._recargar_config_pendiente = False
                    self._analizador.cargar_configuracion(
                        self.ruta_config, cam_id=self.cam_id_actual
                    )
                    camara.cambiar_camara(self.cam_id_actual, self._analizador.scale_factor)
                    ultimo_estado_log = None
                    self._emitir_resumen_camara()

                # 4. ¿Eliminó, añadió alguna plaza o ajustó umbrales en caliente?
                reusar_frame_cache = False
                if self._umbrales_modificados:
                    self._umbrales_modificados = False
                    reusar_frame_cache = True

                while self._puntos_eliminar_pendientes:
                    px, py = self._puntos_eliminar_pendientes.pop(0)
                    if self._analizador.eliminar_plaza_en_punto(px, py):
                        reusar_frame_cache = True
                        self.log_mensaje.emit(
                            f"🗑️ Plaza eliminada en ({px}, {py}) y marcada como calzada excluida."
                        )

                while self._plazas_manuales_pendientes:
                    pts = self._plazas_manuales_pendientes.pop(0)
                    self._analizador.agregar_plaza_manual(pts)
                    reusar_frame_cache = True
                    self.log_mensaje.emit(
                        f"➕ Nueva plaza #{self._analizador.total_plazas} añadida sobre el vídeo."
                    )

                # 5. Obtener fotograma (con manejo resiliente de caídas de red / cámara)
                if reusar_frame_cache and self._ultimo_frame_bgr is not None:
                    frame_bgr = self._ultimo_frame_bgr
                else:
                    try:
                        frame_bgr = camara.obtener_frame()
                    except Exception as exc:
                        frame_bgr = None
                        logger_msg = f"Excepción de red al conectar con cámara [{self.cam_id_actual}]: {exc}"
                        self.estado_error.emit(logger_msg)

                if not self._ejecutando:
                    break

                if frame_bgr is None:
                    fallos_red_consecutivos += 1
                    nombre_cam = obtener_nombre_camara(self.cam_id_actual)
                    msg_aviso = (
                        f"Fallo temporal al conectar con [{self.cam_id_actual}] {nombre_cam} "
                        f"(intento #{fallos_red_consecutivos}). Reintentando en {self.intervalo_segundos:.1f}s..."
                    )
                    self.aviso_red.emit(msg_aviso, fallos_red_consecutivos)
                else:
                    if fallos_red_consecutivos > 0:
                        self.log_mensaje.emit(
                            f"✅ Conexión restablecida con [{self.cam_id_actual}] "
                            f"{obtener_nombre_camara(self.cam_id_actual)} tras {fallos_red_consecutivos} intento(s)."
                        )
                        fallos_red_consecutivos = 0

                    self._ultimo_frame_bgr = frame_bgr

                    try:
                        frame_proc, libres = self._analizador.procesar_frame(frame_bgr)
                        total = self._analizador.total_plazas
                        vehiculos = self._analizador.vehiculos_detectados
                        en_obs = self._analizador.candidatos_en_observacion
                        modo_cal = self._analizador.modo_calibracion_actual
                        qimage = self._convertir_bgr_a_qimage(frame_proc)

                        self.frame_listo.emit(
                            qimage, libres, total, vehiculos, en_obs, self.cam_id_actual, modo_cal
                        )

                        estado_actual = (self.cam_id_actual, libres, total)
                        if total > 0 and estado_actual != ultimo_estado_log:
                            ocupadas = total - libres
                            self.log_mensaje.emit(
                                f"[{self.cam_id_actual}] {obtener_nombre_camara(self.cam_id_actual)}: "
                                f"{libres}/{total} plazas libres ({ocupadas} ocupadas)."
                            )
                            ultimo_estado_log = estado_actual

                    except Exception as exc:
                        self.estado_error.emit(f"Error procesando fotograma: {exc}")

                # Espera fraccionada de 100ms para respuesta inmediata a controles en caliente
                pasos_espera = int((self.intervalo_segundos * 1000) / 100)
                for _ in range(max(1, pasos_espera)):
                    if (
                        not self._ejecutando
                        or self._cam_id_pendiente is not None
                        or self._recargar_config_pendiente
                        or self._recalcular_ia_pendiente
                        or self._umbrales_modificados
                        or self._plazas_manuales_pendientes
                        or self._puntos_eliminar_pendientes
                    ):
                        break
                    self.msleep(100)

        self.log_mensaje.emit("Monitorización detenida.")

    def _emitir_resumen_camara(self) -> None:
        if self._analizador is None:
            return
        nombre = obtener_nombre_camara(self.cam_id_actual)
        total = self._analizador.total_plazas
        if total > 0:
            self.log_mensaje.emit(
                f"Cámara activa: [{self.cam_id_actual}] {nombre} ({total} plazas registradas)."
            )
        else:
            self.log_mensaje.emit(
                f"Cámara activa: [{self.cam_id_actual}] {nombre} -> "
                "Observando vehículos para distinguir estacionados reales de semáforos/carriles..."
            )

    @staticmethod
    def _convertir_bgr_a_qimage(frame_bgr: np.ndarray) -> QImage:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        alto, ancho, canales = frame_rgb.shape
        bytes_por_linea = canales * ancho
        imagen = QImage(
            frame_rgb.data,
            ancho,
            alto,
            bytes_por_linea,
            QImage.Format.Format_RGB888,
        )
        return imagen.copy()


class VisorVideoLabel(QLabel):
    """
    Visor interactivo que muestra el flujo de vídeo manteniendo la relación de aspecto,
    conserva el último fotograma procesado para capturas (`Snapshots`) y permite editar
    plazas directamente con el ratón.
    """

    plaza_dibujada = pyqtSignal(list)
    plaza_eliminar_clic = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._qimage_actual: Optional[QImage] = None
        self._pixmap_original: Optional[QPixmap] = None
        self._tamano_imagen_real: Optional[QSize] = None
        self._rect_dibujado: QRect = QRect()

        self._arrastrando: bool = False
        self._pto_inicio_widget: Optional[QPoint] = None
        self._pto_actual_widget: Optional[QPoint] = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(640, 440))
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setText(
            "Flujo de vídeo inactivo\n\n"
            "Selecciona una cámara de Vigo y pulsa «Iniciar Monitorización»."
        )
        self.setObjectName("VisorVideo")

    @property
    def qimage_actual(self) -> Optional[QImage]:
        """Devuelve la última `QImage` procesada con los polígonos dibujados."""
        return self._qimage_actual

    def actualizar_imagen(self, qimage: QImage) -> None:
        self._qimage_actual = qimage
        self._tamano_imagen_real = qimage.size()
        self._pixmap_original = QPixmap.fromImage(qimage)
        self._reescalar_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap_original is not None:
            self._reescalar_pixmap()

    def _reescalar_pixmap(self) -> None:
        if self._pixmap_original is None or self._pixmap_original.isNull():
            return

        pixmap_base = self._pixmap_original.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        x_offset = (self.width() - pixmap_base.width()) // 2
        y_offset = (self.height() - pixmap_base.height()) // 2
        self._rect_dibujado = QRect(
            x_offset, y_offset, pixmap_base.width(), pixmap_base.height()
        )

        if (
            self._arrastrando
            and self._pto_inicio_widget is not None
            and self._pto_actual_widget is not None
        ):
            copia = pixmap_base.copy()
            painter = QPainter(copia)
            pen = QPen(QColor(0, 230, 118), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 230, 118, 55))
            p1 = self._pto_inicio_widget - QPoint(x_offset, y_offset)
            p2 = self._pto_actual_widget - QPoint(x_offset, y_offset)
            painter.drawRect(QRect(p1, p2).normalized())
            painter.end()
            super().setPixmap(copia)
        else:
            super().setPixmap(pixmap_base)

    def _widget_a_imagen(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        if (
            self._tamano_imagen_real is None
            or self._rect_dibujado.width() <= 0
            or self._rect_dibujado.height() <= 0
        ):
            return None

        if not self._rect_dibujado.contains(pos):
            return None

        rel_x = (pos.x() - self._rect_dibujado.x()) / float(self._rect_dibujado.width())
        rel_y = (pos.y() - self._rect_dibujado.y()) / float(self._rect_dibujado.height())

        img_x = int(round(rel_x * self._tamano_imagen_real.width()))
        img_y = int(round(rel_y * self._tamano_imagen_real.height()))
        return (img_x, img_y)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._pixmap_original is None:
            return

        pos = event.position().toPoint()
        coord_img = self._widget_a_imagen(pos)
        if coord_img is None:
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.plaza_eliminar_clic.emit(coord_img[0], coord_img[1])
        elif event.button() == Qt.MouseButton.LeftButton:
            self._arrastrando = True
            self._pto_inicio_widget = pos
            self._pto_actual_widget = pos

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._arrastrando:
            self._pto_actual_widget = event.position().toPoint()
            self._reescalar_pixmap()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._arrastrando:
            self._arrastrando = False
            fin_pos = event.position().toPoint()
            if self._pto_inicio_widget is not None:
                c1 = self._widget_a_imagen(self._pto_inicio_widget)
                c2 = self._widget_a_imagen(fin_pos)
                if c1 is not None and c2 is not None:
                    x1, x2 = sorted((c1[0], c2[0]))
                    y1, y2 = sorted((c1[1], c2[1]))
                    if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                        poligono = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        self.plaza_dibujada.emit(poligono)
            self._pto_inicio_widget = None
            self._pto_actual_widget = None
            self._reescalar_pixmap()


class VentanaParkingVision(QMainWindow):
    """Ventana principal de la aplicación de escritorio Parking Vision."""

    RUTA_CONFIG = Path("plazas_vigo.json")
    CARPETA_CAPTURAS = Path("capturas")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Parking Vision - Monitorización Inteligente de Aparcamiento en Vigo")
        self.resize(1280, 790)

        self.hilo_worker: Optional[HiloMonitorizacion] = None
        self.catalogo_camaras: dict[str, str] = obtener_catalogo_camaras()

        self._configurar_estilos()
        self._construir_interfaz()
        self._poblar_selector_camaras()

        self._registrar_log(
            f"Catálogo listo: {len(self.catalogo_camaras)} cámaras en Vigo. "
            "Ajuste de parámetros en caliente (conf=0.18, solapamiento=0.25) y capturas disponibles."
        )

    def _obtener_camaras_calibradas(self) -> tuple[set[str], str]:
        calibradas: set[str] = set()
        cam_por_defecto = "05"

        if not self.RUTA_CONFIG.exists():
            return calibradas, cam_por_defecto

        try:
            datos = json.loads(self.RUTA_CONFIG.read_text(encoding="utf-8"))
            if "cam_id" in datos:
                cam_por_defecto = str(datos["cam_id"])
                if datos.get("plazas"):
                    calibradas.add(cam_por_defecto)

            camaras_dict = datos.get("camaras", {})
            if isinstance(camaras_dict, dict):
                for cid, info in camaras_dict.items():
                    if isinstance(info, dict) and info.get("plazas"):
                        calibradas.add(str(cid))
        except Exception:
            pass

        return calibradas, cam_por_defecto

    def _configurar_estilos(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #181a20;
                color: #e4e6eb;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QLabel#VisorVideo {
                background-color: #0f1115;
                border: 2px solid #2b2f3a;
                border-radius: 10px;
                color: #7d8597;
                font-size: 15px;
                padding: 8px;
            }
            QLabel#BannerCamara {
                background-color: #20242d;
                border: 1px solid #2f3545;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: bold;
                color: #f0f4f8;
            }
            QLabel#AyudaDirecta {
                background-color: #161922;
                border: 1px solid #283044;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
                color: #9ec5fe;
            }
            QGroupBox {
                background-color: #20232b;
                border: 1px solid #2f3441;
                border-radius: 8px;
                margin-top: 8px;
                padding: 12px 10px 10px 10px;
                font-weight: bold;
                color: #cfd4e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
            }
            QComboBox, QLineEdit {
                background-color: #121419;
                border: 1px solid #3c4252;
                border-radius: 5px;
                padding: 5px 10px;
                color: #ffffff;
                font-size: 13px;
            }
            QComboBox:hover, QLineEdit:focus {
                border: 1px solid #4d84ff;
            }
            QComboBox QAbstractItemView {
                background-color: #181b22;
                color: #e4e6eb;
                selection-background-color: #2b59c3;
                selection-color: #ffffff;
                border: 1px solid #3c4252;
            }
            QPushButton#BotonAccion {
                background-color: #198754;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 14px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#BotonAccion:hover {
                background-color: #157347;
            }
            QPushButton#BotonAccion[activo="true"] {
                background-color: #dc3545;
            }
            QPushButton#BotonAccion[activo="true"]:hover {
                background-color: #bb2d3b;
            }
            QPushButton#BotonSnapshot {
                background-color: #0d6efd;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 9px 14px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton#BotonSnapshot:hover {
                background-color: #0b5ed7;
            }
            QPushButton#BotonSnapshot:disabled {
                background-color: #263042;
                color: #6c757d;
            }
            QPushButton#BotonAutoIA {
                background-color: #6f42c1;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 7px 12px;
                font-weight: bold;
            }
            QPushButton#BotonAutoIA:hover {
                background-color: #59339d;
            }
            QPushButton#BotonSecundario {
                background-color: #2b303c;
                color: #dce0e8;
                border: 1px solid #3c4252;
                border-radius: 5px;
                padding: 6px 10px;
            }
            QPushButton#BotonSecundario:hover {
                background-color: #363c4a;
            }
            QTextEdit {
                background-color: #121419;
                border: 1px solid #2b2f3a;
                border-radius: 6px;
                color: #b8c0d0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 6px;
            }
            QDoubleSpinBox {
                background-color: #121419;
                border: 1px solid #3c4252;
                border-radius: 4px;
                padding: 3px 6px;
                color: #ffffff;
                min-width: 68px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #121419;
                border: 1px solid #2e3442;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #e4e6eb;
                border: 1px solid #3b82f6;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QFrame#TarjetaContador {
                background-color: #14171d;
                border: 1px solid #2e3442;
                border-radius: 8px;
                padding: 8px;
            }
            """
        )

    def _construir_interfaz(self) -> None:
        widget_central = QWidget(self)
        self.setCentralWidget(widget_central)

        layout_principal = QHBoxLayout(widget_central)
        layout_principal.setContentsMargins(14, 12, 14, 12)
        layout_principal.setSpacing(14)

        # =====================================================================
        # PANEL IZQUIERDO: Cabecera + Flujo de Vídeo Interactivo + Guía Ratón
        # =====================================================================
        contenedor_izquierdo = QWidget(self)
        layout_izquierdo = QVBoxLayout(contenedor_izquierdo)
        layout_izquierdo.setContentsMargins(0, 0, 0, 0)
        layout_izquierdo.setSpacing(6)

        self.lbl_banner_camara = QLabel("📍 Cámara seleccionada: [05] Castelao - Grove")
        self.lbl_banner_camara.setObjectName("BannerCamara")
        layout_izquierdo.addWidget(self.lbl_banner_camara)

        self.visor_video = VisorVideoLabel(self)
        self.visor_video.plaza_dibujada.connect(self._al_dibujar_plaza_en_visor)
        self.visor_video.plaza_eliminar_clic.connect(self._al_eliminar_plaza_en_visor)
        layout_izquierdo.addWidget(self.visor_video, stretch=1)

        lbl_ayuda_directa = QLabel(
            "💡 Edición directa sobre el vídeo:  "
            "[Clic Izquierdo y Arrastrar] Añadir plaza de aparcamiento   •   "
            "[Clic Derecho sobre una plaza] Borrar plaza mal situada"
        )
        lbl_ayuda_directa.setObjectName("AyudaDirecta")
        lbl_ayuda_directa.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_izquierdo.addWidget(lbl_ayuda_directa)

        layout_principal.addWidget(contenedor_izquierdo, stretch=7)

        # =====================================================================
        # PANEL DERECHO: Cámara, Parámetros en Caliente, Disponibilidad y Logs
        # =====================================================================
        panel_derecho = QWidget(self)
        panel_derecho.setFixedWidth(405)
        layout_derecho = QVBoxLayout(panel_derecho)
        layout_derecho.setContentsMargins(0, 0, 0, 0)
        layout_derecho.setSpacing(8)

        # 1. Grupo de Selección de Cámara y Calibración
        grupo_camara = QGroupBox("Cámara de Tráfico (Concello de Vigo)")
        layout_camara = QVBoxLayout(grupo_camara)
        layout_camara.setSpacing(6)

        self.txt_filtro_camara = QLineEdit()
        self.txt_filtro_camara.setPlaceholderText("🔍 Buscar calle, plaza o ID (ej. Castelao, Samil, 24)...")
        self.txt_filtro_camara.setClearButtonEnabled(True)
        self.txt_filtro_camara.textChanged.connect(self._filtrar_selector_camaras)
        layout_camara.addWidget(self.txt_filtro_camara)

        self.combo_camaras = QComboBox()
        self.combo_camaras.setMaxVisibleItems(18)
        self.combo_camaras.currentIndexChanged.connect(self._al_cambiar_camara_combo)
        layout_camara.addWidget(self.combo_camaras)

        self.chk_auto_calibrar = QCheckBox("✨ Auto-aprender plazas (con filtro anti-semáforos)")
        self.chk_auto_calibrar.setChecked(True)
        self.chk_auto_calibrar.stateChanged.connect(self._actualizar_ajustes_worker)
        layout_camara.addWidget(self.chk_auto_calibrar)

        fila_botones_cal = QHBoxLayout()
        btn_recalcular_ia = QPushButton("✨ Limpiar y Re-escanear IA")
        btn_recalcular_ia.setObjectName("BotonAutoIA")
        btn_recalcular_ia.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_recalcular_ia.clicked.connect(self._recalibrar_camara_por_ia)

        btn_calibrar_manual = QPushButton("✏️ Editor 4 Puntos")
        btn_calibrar_manual.setObjectName("BotonSecundario")
        btn_calibrar_manual.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_calibrar_manual.clicked.connect(self._abrir_calibrador_camara_actual)

        fila_botones_cal.addWidget(btn_recalcular_ia, stretch=2)
        fila_botones_cal.addWidget(btn_calibrar_manual, stretch=1)
        layout_camara.addLayout(fila_botones_cal)

        layout_derecho.addWidget(grupo_camara)

        # 2. Grupo de Control y Parámetros en Caliente
        grupo_control = QGroupBox("Control y Parámetros en Caliente")
        layout_control = QVBoxLayout(grupo_control)
        layout_control.setSpacing(7)

        # Botones principales: Iniciar/Detener + Guardar Snapshot
        fila_acciones = QHBoxLayout()
        self.btn_iniciar_detener = QPushButton("Iniciar Monitorización")
        self.btn_iniciar_detener.setObjectName("BotonAccion")
        self.btn_iniciar_detener.setProperty("activo", False)
        self.btn_iniciar_detener.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_iniciar_detener.clicked.connect(self._alternar_monitorizacion)

        self.btn_snapshot = QPushButton("📸 Guardar Snapshot")
        self.btn_snapshot.setObjectName("BotonSnapshot")
        self.btn_snapshot.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_snapshot.setEnabled(False)
        self.btn_snapshot.setToolTip("Guarda el fotograma actual procesado en la carpeta 'capturas/'")
        self.btn_snapshot.clicked.connect(self._guardar_snapshot)

        fila_acciones.addWidget(self.btn_iniciar_detener, stretch=3)
        fila_acciones.addWidget(self.btn_snapshot, stretch=2)
        layout_control.addLayout(fila_acciones)

        # Control 1: Umbral de confianza de YOLO (conf, por defecto 0.18)
        grid_params = QGridLayout()
        grid_params.setHorizontalSpacing(8)
        grid_params.setVerticalSpacing(6)

        lbl_conf = QLabel("Confianza YOLO (conf):")
        lbl_conf.setToolTip("Umbral mínimo de confianza para detectar vehículos (por defecto 0.18)")
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(5, 90)
        self.slider_conf.setValue(18)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.05, 0.90)
        self.spin_conf.setSingleStep(0.01)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(0.18)

        self.slider_conf.valueChanged.connect(
            lambda val: self.spin_conf.setValue(round(val / 100.0, 2))
        )
        self.spin_conf.valueChanged.connect(self._al_cambiar_spin_conf)

        grid_params.addWidget(lbl_conf, 0, 0)
        grid_params.addWidget(self.slider_conf, 0, 1)
        grid_params.addWidget(self.spin_conf, 0, 2)

        # Control 2: Umbral de solapamiento de ocupación (UMBRAL_OCUPACION, por defecto 0.25)
        lbl_overlap = QLabel("Solapamiento ocupación:")
        lbl_overlap.setToolTip(
            "Fracción mínima de área de plaza cubierta por un vehículo para marcarla ocupada (por defecto 0.25)"
        )
        self.slider_overlap = QSlider(Qt.Orientation.Horizontal)
        self.slider_overlap.setRange(5, 90)
        self.slider_overlap.setValue(25)

        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.05, 0.90)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_overlap.setDecimals(2)
        self.spin_overlap.setValue(0.25)

        self.slider_overlap.valueChanged.connect(
            lambda val: self.spin_overlap.setValue(round(val / 100.0, 2))
        )
        self.spin_overlap.valueChanged.connect(self._al_cambiar_spin_overlap)

        grid_params.addWidget(lbl_overlap, 1, 0)
        grid_params.addWidget(self.slider_overlap, 1, 1)
        grid_params.addWidget(self.spin_overlap, 1, 2)

        # Control 3: Intervalo de refresco en segundos
        lbl_intervalo = QLabel("Intervalo de refresco:")
        self.spin_intervalo = QDoubleSpinBox()
        self.spin_intervalo.setRange(1.0, 30.0)
        self.spin_intervalo.setSingleStep(0.5)
        self.spin_intervalo.setValue(3.0)
        self.spin_intervalo.setSuffix(" s")
        self.spin_intervalo.valueChanged.connect(self._actualizar_ajustes_worker)

        grid_params.addWidget(lbl_intervalo, 2, 0, 1, 2)
        grid_params.addWidget(self.spin_intervalo, 2, 2)
        layout_control.addLayout(grid_params)

        self.chk_mostrar_yolo = QCheckBox("Mostrar vehículos en tránsito y observación (YOLOv8)")
        self.chk_mostrar_yolo.setChecked(True)
        self.chk_mostrar_yolo.stateChanged.connect(self._actualizar_ajustes_worker)
        layout_control.addWidget(self.chk_mostrar_yolo)

        layout_derecho.addWidget(grupo_control)

        # 3. Grupo de Estado de Plazas en Tiempo Real
        grupo_estado = QGroupBox("Disponibilidad en Tiempo Real")
        layout_estado = QVBoxLayout(grupo_estado)

        tarjeta_contador = QFrame()
        tarjeta_contador.setObjectName("TarjetaContador")
        grid_contador = QGridLayout(tarjeta_contador)
        grid_contador.setContentsMargins(8, 4, 8, 4)

        lbl_titulo_libres = QLabel("PLAZAS LIBRES / CONFIRMADAS")
        lbl_titulo_libres.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_titulo_libres.setStyleSheet("color: #9aa4b8; font-size: 11px; letter-spacing: 1px;")

        self.lbl_contador_plazas = QLabel("-- / --")
        fuente_contador = QFont("Segoe UI", 24, QFont.Weight.Bold)
        self.lbl_contador_plazas.setFont(fuente_contador)
        self.lbl_contador_plazas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_contador_plazas.setStyleSheet("color: #2ecc71;")

        self.lbl_vehiculos_via = QLabel("Vehículos en vía: --  |  En observación: --")
        self.lbl_vehiculos_via.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_vehiculos_via.setStyleSheet("color: #63b3ed; font-size: 12px; font-weight: bold;")

        self.lbl_subestado = QLabel("En espera de conexión...")
        self.lbl_subestado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_subestado.setStyleSheet("color: #8c95a6; font-size: 12px;")

        grid_contador.addWidget(lbl_titulo_libres, 0, 0)
        grid_contador.addWidget(self.lbl_contador_plazas, 1, 0)
        grid_contador.addWidget(self.lbl_vehiculos_via, 2, 0)
        grid_contador.addWidget(self.lbl_subestado, 3, 0)

        layout_estado.addWidget(tarjeta_contador)
        layout_derecho.addWidget(grupo_estado)

        # 4. Grupo de Registro de Actividad
        grupo_logs = QGroupBox("Registro de Actividad")
        layout_logs = QVBoxLayout(grupo_logs)
        layout_logs.setSpacing(6)

        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        layout_logs.addWidget(self.txt_logs, stretch=1)

        btn_limpiar_logs = QPushButton("Limpiar registro")
        btn_limpiar_logs.setObjectName("BotonSecundario")
        btn_limpiar_logs.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_limpiar_logs.clicked.connect(self.txt_logs.clear)
        layout_logs.addWidget(btn_limpiar_logs)

        layout_derecho.addWidget(grupo_logs, stretch=1)
        layout_principal.addWidget(panel_derecho)

    def _al_cambiar_spin_conf(self, valor: float) -> None:
        self.slider_conf.blockSignals(True)
        self.slider_conf.setValue(int(round(valor * 100)))
        self.slider_conf.blockSignals(False)
        self._actualizar_ajustes_worker()

    def _al_cambiar_spin_overlap(self, valor: float) -> None:
        self.slider_overlap.blockSignals(True)
        self.slider_overlap.setValue(int(round(valor * 100)))
        self.slider_overlap.blockSignals(False)
        self._actualizar_ajustes_worker()

    def _guardar_snapshot(self) -> None:
        """
        Guarda en la carpeta local `capturas/` el fotograma actual procesado
        (con los polígonos verde/rojo y cajas de detección) añadiendo marca de tiempo.
        """
        qimg = self.visor_video.qimage_actual
        if qimg is None or qimg.isNull():
            self._registrar_log("⚠️ No hay ningún fotograma activo para guardar como Snapshot.")
            return

        try:
            self.CARPETA_CAPTURAS.mkdir(parents=True, exist_ok=True)
            cid = self.cam_id_seleccionado()
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"snapshot_cam{cid}_{marca_tiempo}.jpg"
            ruta_destino = self.CARPETA_CAPTURAS / nombre_archivo

            if qimg.save(str(ruta_destino), "JPG", 95):
                self._registrar_log(
                    f"📸 Snapshot guardado correctamente en: {ruta_destino.resolve()}"
                )
            else:
                self._al_recibir_error(
                    f"No se pudo escribir el archivo de imagen en '{ruta_destino}'."
                )
        except OSError as exc:
            self._al_recibir_error(f"Error de disco al guardar Snapshot: {exc}")

    def _al_dibujar_plaza_en_visor(self, puntos: list) -> None:
        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self.hilo_worker.solicitar_agregar_plaza(puntos)

    def _al_eliminar_plaza_en_visor(self, x: int, y: int) -> None:
        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self.hilo_worker.solicitar_eliminar_plaza(x, y)

    def _poblar_selector_camaras(self, filtro: str = "", cam_id_seleccionar: Optional[str] = None) -> None:
        calibradas, cam_defecto = self._obtener_camaras_calibradas()
        id_previo = cam_id_seleccionar or self.cam_id_seleccionado() or cam_defecto

        texto_filtro = filtro.strip().lower()
        self.combo_camaras.blockSignals(True)
        self.combo_camaras.clear()

        indice_a_seleccionar = 0
        items_agregados = 0

        for cid, ubicacion in sorted(self.catalogo_camaras.items(), key=lambda x: (len(x[0]), x[0])):
            etiqueta_busqueda = f"{cid} {ubicacion}".lower()
            if texto_filtro and texto_filtro not in etiqueta_busqueda:
                continue

            marca = " ★" if cid in calibradas else ""
            texto_item = f"[{cid}] {ubicacion}{marca}"
            self.combo_camaras.addItem(texto_item, userData=cid)

            if cid == id_previo:
                indice_a_seleccionar = items_agregados
            items_agregados += 1

        if items_agregados > 0:
            self.combo_camaras.setCurrentIndex(indice_a_seleccionar)

        self.combo_camaras.blockSignals(False)
        self._actualizar_banner_ubicacion()

    def _filtrar_selector_camaras(self, texto: str) -> None:
        id_antes = self.cam_id_seleccionado()
        self._poblar_selector_camaras(filtro=texto, cam_id_seleccionar=id_antes)
        id_despues = self.cam_id_seleccionado()
        if id_despues and id_despues != id_antes:
            self._al_cambiar_camara_combo()

    def cam_id_seleccionado(self) -> str:
        data = self.combo_camaras.currentData()
        return str(data) if data is not None else "05"

    def _actualizar_banner_ubicacion(self, modo: Optional[str] = None) -> None:
        cid = self.cam_id_seleccionado()
        ubicacion = obtener_nombre_camara(cid)
        if modo == "auto":
            etiqueta_modo = "  •  ✨ Aprendizaje IA (Anti-Semáforos)"
        elif modo == "manual":
            etiqueta_modo = "  •  ✏️ Plazas Personalizadas"
        else:
            etiqueta_modo = "  •  ✨ Filtro inteligente de calzada activo"
        self.lbl_banner_camara.setText(f"📍 [{cid}] {ubicacion}{etiqueta_modo}")

    def _al_cambiar_camara_combo(self) -> None:
        cid = self.cam_id_seleccionado()
        if not cid:
            return

        self._actualizar_banner_ubicacion()
        ubicacion = obtener_nombre_camara(cid)
        self._registrar_log(f"Seleccionada cámara [{cid}] {ubicacion}.")

        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self.lbl_subestado.setText(f"Cambiando a [{cid}] {ubicacion}...")
            self.hilo_worker.solicitar_cambio_camara(cid)

    def _recalibrar_camara_por_ia(self) -> None:
        self.chk_auto_calibrar.setChecked(True)
        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self.hilo_worker.solicitar_autocalibracion_ia()
        else:
            self._iniciar_monitorizacion()
            if self.hilo_worker is not None:
                self.hilo_worker.solicitar_autocalibracion_ia()

    def _abrir_calibrador_camara_actual(self) -> None:
        cid = self.cam_id_seleccionado()
        ubicacion = obtener_nombre_camara(cid)
        self._registrar_log(f"Abriendo editor 4 puntos para [{cid}] {ubicacion}...")

        try:
            subprocess.run(
                [sys.executable, "calibrar_plazas.py", "--cam-id", cid],
                check=False,
            )
        except Exception as exc:
            self._al_recibir_error(f"No se pudo lanzar calibrar_plazas.py: {exc}")
            return

        self._poblar_selector_camaras(
            filtro=self.txt_filtro_camara.text(),
            cam_id_seleccionar=cid,
        )
        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self.hilo_worker.solicitar_recarga_plazas()

    def _alternar_monitorizacion(self) -> None:
        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self._detener_monitorizacion()
        else:
            self._iniciar_monitorizacion()

    def _iniciar_monitorizacion(self) -> None:
        cid = self.cam_id_seleccionado()

        self.btn_iniciar_detener.setText("Detener Monitorización")
        self.btn_iniciar_detener.setProperty("activo", True)
        self.btn_iniciar_detener.style().unpolish(self.btn_iniciar_detener)
        self.btn_iniciar_detener.style().polish(self.btn_iniciar_detener)

        self.lbl_subestado.setStyleSheet("color: #8c95a6; font-size: 12px;")
        self.lbl_subestado.setText("Conectando e inicializando YOLOv8...")

        self.hilo_worker = HiloMonitorizacion(
            cam_id=cid,
            ruta_config=self.RUTA_CONFIG,
            ruta_modelo="yolov8m.pt",
            intervalo_segundos=self.spin_intervalo.value(),
            conf_yolo=self.spin_conf.value(),
            umbral_ocupacion=self.spin_overlap.value(),
            mostrar_cajas_yolo=self.chk_mostrar_yolo.isChecked(),
            auto_calibrar=self.chk_auto_calibrar.isChecked(),
            parent=self,
        )
        self.hilo_worker.frame_listo.connect(self._al_recibir_frame)
        self.hilo_worker.log_mensaje.connect(self._registrar_log)
        self.hilo_worker.aviso_red.connect(self._al_recibir_aviso_red)
        self.hilo_worker.estado_error.connect(self._al_recibir_error)
        self.hilo_worker.finished.connect(self._al_finalizar_hilo)
        self.hilo_worker.start()

    def _detener_monitorizacion(self) -> None:
        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self.btn_iniciar_detener.setEnabled(False)
            self.btn_iniciar_detener.setText("Deteniendo...")
            self._registrar_log("Solicitando parada del hilo de monitorización...")
            self.hilo_worker.detener()

    def _al_finalizar_hilo(self) -> None:
        self.btn_iniciar_detener.setEnabled(True)
        self.btn_iniciar_detener.setText("Iniciar Monitorización")
        self.btn_iniciar_detener.setProperty("activo", False)
        self.btn_iniciar_detener.style().unpolish(self.btn_iniciar_detener)
        self.btn_iniciar_detener.style().polish(self.btn_iniciar_detener)
        self.lbl_subestado.setStyleSheet("color: #8c95a6; font-size: 12px;")
        self.lbl_subestado.setText("Monitorización en pausa")

    def _actualizar_ajustes_worker(self) -> None:
        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self.hilo_worker.actualizar_opciones(
                intervalo_segundos=self.spin_intervalo.value(),
                conf_yolo=self.spin_conf.value(),
                umbral_ocupacion=self.spin_overlap.value(),
                mostrar_cajas_yolo=self.chk_mostrar_yolo.isChecked(),
                auto_calibrar=self.chk_auto_calibrar.isChecked(),
            )

    def _al_recibir_frame(
        self,
        qimage: QImage,
        libres: int,
        total: int,
        vehiculos: int,
        en_observacion: int,
        cam_id: str,
        modo_cal: str,
    ) -> None:
        self.visor_video.actualizar_imagen(qimage)
        self.btn_snapshot.setEnabled(True)
        self._actualizar_banner_ubicacion(modo=modo_cal)
        self.lbl_vehiculos_via.setText(
            f"En vía: {vehiculos}  |  Verificando estacionamiento: {en_observacion}"
        )

        hora_str = datetime.now().strftime("%H:%M:%S")
        self.lbl_subestado.setStyleSheet("color: #8c95a6; font-size: 12px;")

        if total == 0:
            self.lbl_contador_plazas.setText("Verificando...")
            self.lbl_contador_plazas.setStyleSheet("color: #f39c12; font-size: 20px;")
            self.lbl_subestado.setText(
                f"Filtrando tráfico/semáforos (o arrastra en el vídeo) • {hora_str}"
            )
            return

        self.lbl_contador_plazas.setText(f"{libres} / {total}")
        ocupadas = max(0, total - libres)
        self.lbl_subestado.setText(
            f"Ocupadas: {ocupadas}  •  Última act.: {hora_str}"
        )

        if libres == 0:
            color = "#e74c3c"
        elif libres <= max(1, total // 4):
            color = "#f39c12"
        else:
            color = "#2ecc71"

        self.lbl_contador_plazas.setStyleSheet(f"color: {color}; font-size: 24px;")

    def _al_recibir_aviso_red(self, mensaje: str, fallos: int) -> None:
        """
        Muestra avisos de red en el registro y actualiza el indicador visual sin
        bloquear ni detener el hilo de monitorización.
        """
        self._registrar_log(f"⚠️ [RED] {mensaje}")
        self.lbl_subestado.setStyleSheet("color: #f39c12; font-size: 12px; font-weight: bold;")
        self.lbl_subestado.setText(f"⚠️ Sin respuesta de cámara (reintento #{fallos})...")

    def _al_recibir_error(self, mensaje_error: str) -> None:
        self._registrar_log(f"❌ [ERROR] {mensaje_error}")

    def _registrar_log(self, mensaje: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_logs.append(f"[{timestamp}] {mensaje}")
        barra = self.txt_logs.verticalScrollBar()
        if barra is not None:
            barra.setValue(barra.maximum())

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.hilo_worker is not None and self.hilo_worker.isRunning():
            self._registrar_log("Cerrando aplicación: esperando parada segura del hilo...")
            self.hilo_worker.detener()
            if not self.hilo_worker.wait(5000):
                self.hilo_worker.terminate()
                self.hilo_worker.wait(1000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Parking Vision")

    ventana = VentanaParkingVision()
    ventana.showNormal()
    ventana.raise_()
    ventana.activateWindow()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
