"""
Módulo de detección de ocupación y aprendizaje inteligente de plazas de aparcamiento
mediante YOLOv8 y Shapely.
Incluye filtrado de carriles de circulación, detección de colas de semáforo,
verificación de cambio real de fotograma en las cámaras de Vigo y soporte de edición directa.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, box
from ultralytics import YOLO

from camara_vigo import obtener_nombre_camara

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@dataclass
class PlazaParking:
    """Representa una plaza de aparcamiento y su estado actual."""

    id: int
    puntos: List[Tuple[int, int]]
    poligono: Polygon
    ocupada: bool = False
    max_solapamiento: float = 0.0
    origen: str = "auto"          # "manual" o "auto"
    capturas_estatico: int = 1    # Número de capturas REALES distintas que lleva inmóvil


@dataclass
class CandidatoEstacionamiento:
    """
    Vehículo en observación temporal antes de ser aceptado como plaza de aparcamiento.
    Evita confundir coches circulando o esperando en un semáforo con plazas reales.
    """

    poligono: Polygon
    capturas_quieto: int = 1
    confianza: float = 0.0


class AnalizadorParking:
    """
    Motor de análisis de ocupación de plazas de aparcamiento basado en YOLOv8 y Shapely.

    Mejoras clave contra falsos positivos en calzada y semáforos:
    1. Detecta cuándo el servidor de Vigo realmente actualiza la imagen JPEG (cada ~10-15s)
       para no contar fotogramas repetidos como "tiempo estacionado".
    2. Registra una máscara de carriles de tráfico y semáforos (`_zonas_calzada`): cualquier
       vehículo que se mueva o desaparezca tras una parada breve marca su huella como
       carril de circulación, impidiendo crear plazas en mitad de la calzada.
    3. Detecta aperturas de semáforo (cuando 2 o más vehículos detenidos arrancan a la vez)
       y purga automáticamente esas posiciones en lugar de marcarlas como "plazas libres".
    4. Permite añadir o eliminar plazas con un clic directamente desde el visor de PyQt6.
    """

    CLASES_VEHICULOS: List[int] = [2, 3, 5, 7]
    NOMBRES_CLASES: dict[int, str] = {
        2: "Coche",
        3: "Moto",
        5: "Bus",
        7: "Camion",
    }

    # Número mínimo de capturas reales distintas de cámara (~30-40s) sin moverse
    # para que un vehículo aislado se consolide como estacionado (superando un semáforo)
    CAPTURAS_MIN_CONSOLIDACION = 3

    # Colores en formato BGR (OpenCV)
    COLOR_LIBRE: Tuple[int, int, int] = (0, 210, 0)         # Verde (Plaza Libre)
    COLOR_OCUPADA: Tuple[int, int, int] = (0, 0, 220)       # Rojo (Plaza Ocupada)
    COLOR_OBSERVACION: Tuple[int, int, int] = (0, 190, 255) # Ámbar (Evaluando si está aparcado o en semáforo)
    COLOR_DETECCION: Tuple[int, int, int] = (255, 180, 0)   # Cian (Vehículo en tránsito)
    COLOR_TEXTO: Tuple[int, int, int] = (255, 255, 255)     # Blanco

    def __init__(
        self,
        ruta_config: str | Path = "plazas_vigo.json",
        ruta_modelo: str | Path = "yolov8m.pt",
        cam_id: Optional[str] = None,
        conf_threshold: float = 0.18,
        overlap_threshold: float = 0.25,
        dibujar_detecciones: bool = True,
        auto_calibrar: bool = True,
    ) -> None:
        self.ruta_config = Path(ruta_config)
        self.ruta_modelo = str(ruta_modelo)
        self.conf_threshold = float(conf_threshold)
        self.overlap_threshold = float(overlap_threshold)
        self.dibujar_detecciones = dibujar_detecciones
        self.auto_calibrar = auto_calibrar

        self.cam_id: str = str(cam_id) if cam_id is not None else "05"
        self.scale_factor: float = 2.0
        self.plazas: List[PlazaParking] = []
        self.vehiculos_detectados: int = 0
        self.candidatos_en_observacion: int = 0
        self.modo_calibracion_actual: str = "auto"

        # Estado temporal para distinguir tráfico/semáforos de aparcamiento real
        self._miniatura_previa: Optional[np.ndarray] = None
        self._candidatos: List[CandidatoEstacionamiento] = []
        self._zonas_calzada: List[Polygon] = []  # Huellas de carriles y semáforos
        self._capturas_reales_camara: int = 0

        # 1. Cargar configuración de plazas si existe
        self.cargar_configuracion(self.ruta_config, cam_id=cam_id)

        # 2. Cargar modelo YOLOv8
        logger.info("Cargando modelo YOLOv8 (%s)...", self.ruta_modelo)
        self.modelo = YOLO(self.ruta_modelo)
        logger.info("Modelo YOLOv8 cargado correctamente.")

    @property
    def ubicacion(self) -> str:
        """Devuelve la descripción de la ubicación de la cámara actual."""
        return obtener_nombre_camara(self.cam_id)

    def actualizar_umbrales(
        self,
        conf_threshold: float,
        overlap_threshold: float,
    ) -> None:
        """Actualiza en caliente los umbrales de confianza YOLO y solapamiento de ocupación."""
        self.conf_threshold = max(0.01, min(0.99, float(conf_threshold)))
        self.overlap_threshold = max(0.01, min(0.99, float(overlap_threshold)))

    def cambiar_camara(self, nuevo_cam_id: str) -> bool:
        """Cambia la cámara activa y reinicia la memoria temporal de tráfico."""
        self.cam_id = str(nuevo_cam_id)
        self._miniatura_previa = None
        self._candidatos.clear()
        self._zonas_calzada.clear()
        self._capturas_reales_camara = 0
        return self.cargar_configuracion(self.ruta_config, cam_id=self.cam_id)

    def reiniciar_autocalibracion_camara(self) -> None:
        """Limpia las plazas automáticas de la cámara actual para re-aprender desde cero."""
        self.plazas = [p for p in self.plazas if p.origen == "manual"]
        self._miniatura_previa = None
        self._candidatos.clear()
        self._capturas_reales_camara = 0
        self.modo_calibracion_actual = "manual" if self.plazas else "auto"
        self.guardar_plazas_actuales()

    def agregar_plaza_manual(self, puntos: List[Tuple[int, int]]) -> None:
        """Añade directamente una plaza definida por el usuario sobre el visor de vídeo."""
        if len(puntos) < 3:
            return
        poly = Polygon(puntos)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area <= 50:
            return

        nuevo_id = len(self.plazas) + 1
        self.plazas.append(
            PlazaParking(
                id=nuevo_id,
                puntos=[(int(x), int(y)) for x, y in puntos],
                poligono=poly,
                origen="manual",
                capturas_estatico=99,
            )
        )
        self._renumerar_plazas()
        self.guardar_plazas_actuales()

    def eliminar_plaza_en_punto(self, x: int, y: int) -> bool:
        """
        Elimina la plaza que contenga el punto `(x, y)` (clic derecho del usuario)
        y registra su área como zona de calzada excluida para que la IA no vuelva a crearla.
        """
        punto = Point(x, y)
        for idx, plaza in enumerate(self.plazas):
            if plaza.poligono.contains(punto):
                eliminada = self.plazas.pop(idx)
                self._zonas_calzada.append(eliminada.poligono)
                self._renumerar_plazas()
                self.guardar_plazas_actuales()
                return True
        return False

    def _renumerar_plazas(self) -> None:
        for idx, p in enumerate(self.plazas, start=1):
            p.id = idx

    def cargar_configuracion(
        self,
        ruta_config: Optional[str | Path] = None,
        cam_id: Optional[str] = None,
    ) -> bool:
        """Carga los polígonos de `plazas_vigo.json` para la cámara indicada."""
        if ruta_config is not None:
            self.ruta_config = Path(ruta_config)

        if cam_id is not None:
            self.cam_id = str(cam_id)

        if not self.ruta_config.exists():
            self.plazas.clear()
            return False

        try:
            contenido = json.loads(self.ruta_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Error leyendo '%s': %s", self.ruta_config, exc)
            self.plazas.clear()
            return False

        plazas_raw = []
        zonas_excluidas_raw = []
        camaras_dict = contenido.get("camaras", {})
        origen_guardado = "manual"

        if cam_id is None and "cam_id" in contenido:
            self.cam_id = str(contenido.get("cam_id", "05"))

        if isinstance(camaras_dict, dict) and self.cam_id in camaras_dict:
            bloque = camaras_dict[self.cam_id]
            self.scale_factor = float(bloque.get("scale_factor", 2.0))
            plazas_raw = bloque.get("plazas", [])
            zonas_excluidas_raw = bloque.get("zonas_calzada", [])
            origen_guardado = str(bloque.get("origen", "manual"))
        elif str(contenido.get("cam_id")) == self.cam_id:
            self.scale_factor = float(contenido.get("scale_factor", 2.0))
            plazas_raw = contenido.get("plazas", [])
            zonas_excluidas_raw = contenido.get("zonas_calzada", [])
            origen_guardado = str(contenido.get("origen", "manual"))
        else:
            self.scale_factor = 2.0
            self.plazas.clear()
            return False

        # Cargar zonas de calzada excluidas si las hubiera
        self._zonas_calzada.clear()
        for coords_z in zonas_excluidas_raw:
            if len(coords_z) >= 3:
                pz = Polygon([(int(pt[0]), int(pt[1])) for pt in coords_z])
                if pz.is_valid and pz.area > 0:
                    self._zonas_calzada.append(pz)

        plazas_cargadas: List[PlazaParking] = []
        for idx, item in enumerate(plazas_raw, start=1):
            if isinstance(item, dict):
                plaza_id = int(item.get("id", idx))
                coords = item.get("poligono", [])
                origen_plaza = str(item.get("origen", origen_guardado))
            else:
                plaza_id = idx
                coords = item
                origen_plaza = origen_guardado

            if len(coords) < 3:
                continue

            puntos: List[Tuple[int, int]] = [(int(pt[0]), int(pt[1])) for pt in coords]
            poligono = Polygon(puntos)
            if not poligono.is_valid:
                poligono = poligono.buffer(0)
            if poligono.area <= 0:
                continue

            plazas_cargadas.append(
                PlazaParking(
                    id=plaza_id,
                    puntos=puntos,
                    poligono=poligono,
                    origen=origen_plaza,
                    capturas_estatico=self.CAPTURAS_MIN_CONSOLIDACION,
                )
            )

        self.plazas = plazas_cargadas
        tiene_manuales = any(p.origen == "manual" for p in self.plazas)
        self.modo_calibracion_actual = "manual" if tiene_manuales else "auto"
        return len(self.plazas) > 0

    def guardar_plazas_actuales(self, alto_img: int = 720, ancho_img: int = 1280) -> None:
        """Guarda en `plazas_vigo.json` las plazas y las zonas de calzada excluidas."""
        plazas_serializadas = [
            {
                "id": idx,
                "poligono": [[int(x), int(y)] for x, y in p.puntos],
                "origen": p.origen,
            }
            for idx, p in enumerate(self.plazas, start=1)
        ]

        zonas_serializadas = [
            [[int(x), int(y)] for x, y in list(z.exterior.coords)[:-1]]
            for z in self._zonas_calzada[-40:]
            if isinstance(z, Polygon) and not z.is_empty
        ]

        camaras_guardadas = {}
        if self.ruta_config.exists():
            try:
                previo = json.loads(self.ruta_config.read_text(encoding="utf-8"))
                if isinstance(previo.get("camaras"), dict):
                    for k, v in previo["camaras"].items():
                        if isinstance(v, dict) and "plazas" in v:
                            camaras_guardadas[str(k)] = v
            except Exception:
                pass

        info_actual = {
            "cam_id": self.cam_id,
            "ubicacion": self.ubicacion,
            "scale_factor": self.scale_factor,
            "origen": self.modo_calibracion_actual,
            "resolucion": {"ancho": ancho_img, "alto": alto_img},
            "total_plazas": len(plazas_serializadas),
            "plazas": plazas_serializadas,
            "zonas_calzada": zonas_serializadas,
        }
        camaras_guardadas[self.cam_id] = info_actual

        datos = {
            "cam_id": self.cam_id,
            "ubicacion": self.ubicacion,
            "scale_factor": self.scale_factor,
            "origen": self.modo_calibracion_actual,
            "resolucion": {"ancho": ancho_img, "alto": alto_img},
            "total_plazas": len(plazas_serializadas),
            "plazas": plazas_serializadas,
            "zonas_calzada": zonas_serializadas,
            "camaras": camaras_guardadas,
        }

        try:
            self.ruta_config.write_text(
                json.dumps(datos, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("No se pudo guardar en disco: %s", exc)

    def _es_captura_nueva(self, frame: np.ndarray) -> bool:
        """
        Comprueba si el fotograma descargado del servidor de Vigo es realmente nuevo
        o si es la misma imagen en caché del servidor (que refresca cada ~10-15 s).
        """
        alto = frame.shape[0]
        recorte = frame[int(alto * 0.12) :, :]
        gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gris, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)

        if self._miniatura_previa is None:
            self._miniatura_previa = mini
            return True

        diff_media = float(np.mean(np.abs(mini - self._miniatura_previa)))
        if diff_media >= 1.2:
            self._miniatura_previa = mini
            return True
        return False

    @staticmethod
    def _es_falso_positivo_estatico(frame: np.ndarray, caja: Polygon) -> bool:
        """
        Descarta objetos estáticos de infraestructura vial que YOLO a veces confunde con coches
        (por ejemplo, amortiguadores rojos de impacto en bocas de túnel o carteles).
        """
        x1, y1, x2, y2 = map(int, caja.bounds)
        alto_f, ancho_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ancho_f, x2), min(alto_f, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return True

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Detectar rojo/granate uniforme de amortiguadores de impacto de túnel / hitos viales
        mask_rojo1 = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([12, 255, 255]))
        mask_rojo2 = cv2.inRange(hsv, np.array([165, 80, 60]), np.array([180, 255, 255]))
        ratio_rojo = float(np.count_nonzero(mask_rojo1 | mask_rojo2)) / float(roi.shape[0] * roi.shape[1])
        return ratio_rojo > 0.25

    def _cae_en_calzada_o_carril(self, caja: Polygon) -> bool:
        """Indica si una caja solapa significativamente con un carril de tráfico o semáforo conocido."""
        area_c = max(1.0, caja.area)
        for z in self._zonas_calzada:
            if caja.intersects(z):
                if (caja.intersection(z).area / area_c) > 0.30:
                    return True
        return False

    def _actualizar_aprendizaje_temporal(
        self,
        frame: np.ndarray,
        detecciones: List[Tuple[Polygon, float, int]],
    ) -> None:
        """
        Actualiza el aprendizaje de plazas únicamente cuando la cámara de Vigo entrega
        un fotograma nuevo:
        - Registra como calzada los autobuses/camiones en tránsito y los coches que se mueven.
        - Si un candidato desaparece antes de consolidarse, era tráfico o espera de semáforo ->
          su huella pasa a `_zonas_calzada`.
        - Si 2 o más plazas auto-detectadas se vacían de golpe en el mismo fotograma ->
          era una cola de semáforo arrancando en verde -> se eliminan esas falsas plazas.
        """
        alto_img, ancho_img = frame.shape[:2]
        area_frame = float(alto_img * ancho_img)
        self._capturas_reales_camara += 1

        cajas_turismos: List[Tuple[Polygon, float]] = []

        for poly, conf, cls_id in detecciones:
            x1, y1, x2, y2 = poly.bounds
            area_p = (x2 - x1) * (y2 - y1)

            # Ignorar banda del reloj superior o cajas desproporcionadas
            if y1 < alto_img * 0.08 or area_p < area_frame * 0.0018 or area_p > area_frame * 0.14:
                continue

            # Autobuses (5) y camiones grandes (7) circulan por carriles: marcan calzada
            if cls_id in (5, 7) and area_p > area_frame * 0.025:
                self._zonas_calzada.append(poly)
                continue

            if self._es_falso_positivo_estatico(frame, poly):
                continue

            cajas_turismos.append((poly, conf))

        # 1. Detectar si una cola de semáforo entre las plazas auto-detectadas acaba de arrancar:
        # Si >= 2 plazas auto-detectadas que estaban ocupadas se quedan vacías en el mismo refresco,
        # no son aparcamientos que se hayan liberado a la vez: ¡es un semáforo que se puso en verde!
        plazas_auto_recien_vaciadas: List[PlazaParking] = []
        for p in self.plazas:
            if p.origen == "auto" and p.ocupada:
                sigue_ocupada = any(
                    (p.poligono.intersection(c).area / max(1.0, p.poligono.area)) > self.overlap_threshold
                    for c, _ in cajas_turismos
                )
                if not sigue_ocupada:
                    plazas_auto_recien_vaciadas.append(p)

        if len(plazas_auto_recien_vaciadas) >= 2:
            # Efecto pelotón de semáforo: eliminar esas cajas y marcarlas como carril de semáforo
            for p_sem in plazas_auto_recien_vaciadas:
                self._zonas_calzada.append(p_sem.poligono)
                if p_sem in self.plazas:
                    self.plazas.remove(p_sem)
            self._renumerar_plazas()

        # 2. Actualizar candidatos en observación
        nuevos_candidatos: List[CandidatoEstacionamiento] = []
        usados_actuales = [False] * len(cajas_turismos)

        for cand in self._candidatos:
            mejor_iou = 0.0
            mejor_idx = -1
            for idx, (c_act, _) in enumerate(cajas_turismos):
                inter = cand.poligono.intersection(c_act).area
                if inter <= 0:
                    continue
                union = cand.poligono.union(c_act).area
                iou = inter / union if union > 0 else 0.0
                if iou > mejor_iou:
                    mejor_iou = iou
                    mejor_idx = idx

            if mejor_iou >= 0.65 and mejor_idx >= 0:
                # El vehículo sigue exactamente en el mismo sitio en una captura nueva
                usados_actuales[mejor_idx] = True
                cand.capturas_quieto += 1
                cand.poligono = cajas_turismos[mejor_idx][0]

                if not self._cae_en_calzada_o_carril(cand.poligono):
                    if cand.capturas_quieto >= self.CAPTURAS_MIN_CONSOLIDACION:
                        self._promover_candidato_a_plaza(cand.poligono)
                    else:
                        nuevos_candidatos.append(cand)
            else:
                # El vehículo se movió o desapareció antes de consolidar estacionamiento:
                # Era un coche en circulación o parado en un semáforo -> registrar como calzada
                self._zonas_calzada.append(cand.poligono)

        # 3. Añadir nuevos vehículos detectados a observación (si no están ya en una plaza o en un carril)
        for idx, (c_act, conf) in enumerate(cajas_turismos):
            if usados_actuales[idx]:
                continue
            if self._cae_en_calzada_o_carril(c_act):
                continue
            ya_en_plaza = any(
                (p.poligono.intersection(c_act).area / max(1.0, p.poligono.area)) > 0.25
                or (p.poligono.intersection(c_act).area / max(1.0, c_act.area)) > 0.35
                for p in self.plazas
            )
            if not ya_en_plaza:
                nuevos_candidatos.append(
                    CandidatoEstacionamiento(poligono=c_act, capturas_quieto=1, confianza=conf)
                )

        self._candidatos = nuevos_candidatos
        self.candidatos_en_observacion = len(self._candidatos)

        # 4. En la primera captura de una cámara, si hay grupos claros de coches aparcados en cordón
        # o batería en los márgenes (fuera del centro de la calzada), pre-registrar solo esos grupos
        # sin inventar huecos en mitad de la calle
        if self._capturas_reales_camara == 1 and not self.plazas and len(self._candidatos) >= 2:
            self._detectar_grupos_aparcados_en_margen(alto_img, ancho_img)

        # 5. Buscar si hay un hueco libre de 1 coche justo en medio de un cordón de acera consolidado
        if len(self.plazas) >= 2:
            self._inferir_hueco_acera_estricto(frame)
            self.guardar_plazas_actuales(alto_img, ancho_img)

    def _detectar_grupos_aparcados_en_margen(self, alto_img: int, ancho_img: int) -> None:
        """
        Identifica en la primera captura únicamente vehículos que forman una fila compacta
        de estacionamiento pegada entre sí o alineada en los márgenes, descartando vehículos
        aislados en los carriles centrales.
        """
        cands = [c.poligono for c in self._candidatos]
        for i, c1 in enumerate(cands):
            x1_a, y1_a, x2_a, y2_a = c1.bounds
            cx_a = (x1_a + x2_a) / 2.0
            cy_a = (y1_a + y2_a) / 2.0
            w_a = x2_a - x1_a
            h_a = y2_a - y1_a
            diag_a = math.hypot(w_a, h_a)

            # Contar cuántos otros vehículos tiene inmediatamente pegados (en batería o cordón)
            vecinos_pegados = 0
            for j, c2 in enumerate(cands):
                if i == j:
                    continue
                x1_b, y1_b, x2_b, y2_b = c2.bounds
                cx_b = (x1_b + x2_b) / 2.0
                cy_b = (y1_b + y2_b) / 2.0
                w_b = x2_b - x1_b
                h_b = y2_b - y1_b

                # Deben tener un tamaño de perspectiva coherente
                if max(w_a, w_b) / max(1.0, min(w_a, w_b)) > 1.55:
                    continue

                dist_centros = math.hypot(cx_a - cx_b, cy_a - cy_b)
                if dist_centros <= diag_a * 1.35:
                    vecinos_pegados += 1

            # Si pertenece a un bloque compacto de aparcamiento o está en el margen lateral
            en_margen_lateral = (cx_a < ancho_img * 0.22) or (cx_a > ancho_img * 0.78)
            if vecinos_pegados >= 1 and (en_margen_lateral or cy_a < alto_img * 0.45):
                self._promover_candidato_a_plaza(c1, capturas_iniciales=2)

    def _promover_candidato_a_plaza(self, caja: Polygon, capturas_iniciales: int = 3) -> None:
        """Convierte un candidato estacionado verificado en una plaza registrada."""
        if self._cae_en_calzada_o_carril(caja):
            return
        ya_existe = any(
            (p.poligono.intersection(caja).area / max(1.0, p.poligono.area)) > 0.25
            or (p.poligono.intersection(caja).area / max(1.0, caja.area)) > 0.35
            for p in self.plazas
        )
        if ya_existe:
            return

        pts = self._caja_a_poligono_plaza(caja)
        nuevo_id = len(self.plazas) + 1
        self.plazas.append(
            PlazaParking(
                id=nuevo_id,
                puntos=pts,
                poligono=Polygon(pts),
                origen="auto",
                capturas_estatico=capturas_iniciales,
            )
        )
        self._renumerar_plazas()

    def _inferir_hueco_acera_estricto(self, frame: np.ndarray) -> None:
        """
        Infiere como máximo 1 plaza libre únicamente cuando dos vehículos estacionados
        consolidados están alineados a lo largo del mismo bordillo lateral (acera)
        dejando exactamente el espacio de 1 coche entre ellos, y nunca cruzando la calzada.
        """
        alto_img, ancho_img = frame.shape[:2]
        plazas_base = [p for p in self.plazas if p.ocupada]
        if len(plazas_base) < 2:
            return

        for i in range(len(plazas_base)):
            for j in range(i + 1, len(plazas_base)):
                p1 = plazas_base[i].poligono
                p2 = plazas_base[j].poligono

                x1_a, y1_a, x2_a, y2_a = p1.bounds
                x1_b, y1_b, x2_b, y2_b = p2.bounds
                w_a, h_a = x2_a - x1_a, y2_a - y1_a
                w_b, h_b = x2_b - x1_b, y2_b - y1_b

                # Ambos coches deben tener dimensiones parecidas (misma línea de perspectiva)
                if max(w_a, w_b) / max(1.0, min(w_a, w_b)) > 1.45:
                    continue
                if max(h_a, h_b) / max(1.0, min(h_a, h_b)) > 1.45:
                    continue

                cx_a, cy_a = (x1_a + x2_a) / 2.0, (y1_a + y2_a) / 2.0
                cx_b, cy_b = (x1_b + x2_b) / 2.0, (y1_b + y2_b) / 2.0

                # PROHIBIDO interpolar horizontalmente a través del centro de la calzada:
                # Solo permitir si ambos coches están en el mismo margen lateral (acera izq o der)
                # y alineados en cordón (desplazamiento vertical mayor o igual que el horizontal)
                mismo_margen_derecho = cx_a > ancho_img * 0.72 and cx_b > ancho_img * 0.72
                mismo_margen_izquierdo = cx_a < ancho_img * 0.28 and cx_b < ancho_img * 0.28
                if not (mismo_margen_derecho or mismo_margen_izquierdo):
                    continue

                dx = abs(cx_b - cx_a)
                dy = abs(cy_b - cy_a)
                h_medio = (h_a + h_b) / 2.0
                w_medio = (w_a + w_b) / 2.0

                # En cordón junto a la acera, el hueco de 1 plaza está entre 1.6x y 2.3x la altura entre centros
                if dy > dx and (h_medio * 1.65 <= dy <= h_medio * 2.35) and dx <= w_medio * 0.85:
                    hx_c = (cx_a + cx_b) / 2.0
                    hy_c = (cy_a + cy_b) / 2.0
                    hw = w_medio * 0.92
                    hh = h_medio * 0.92
                    candidato = box(hx_c - hw / 2, hy_c - hh / 2, hx_c + hw / 2, hy_c + hh / 2)

                    if self._cae_en_calzada_o_carril(candidato):
                        continue
                    solapa = any(
                        candidato.intersection(p.poligono).area / max(1.0, candidato.area) > 0.20
                        for p in self.plazas
                    )
                    if not solapa:
                        self._promover_candidato_a_plaza(candidato, capturas_iniciales=2)

    @staticmethod
    def _caja_a_poligono_plaza(caja: Polygon) -> List[Tuple[int, int]]:
        x1, y1, x2, y2 = caja.bounds
        w = x2 - x1
        inclinacion = max(2, int(w * 0.04))
        return [
            (int(x1 + inclinacion), int(y1)),
            (int(x2 - inclinacion), int(y1)),
            (int(x2), int(y2)),
            (int(x1), int(y2)),
        ]

    def procesar_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Analiza un fotograma BGR para detectar vehículos y evaluar la ocupación de cada plaza.
        """
        if frame is None or frame.size == 0:
            raise ValueError("El fotograma proporcionado está vacío o es inválido.")

        es_nuevo = self._es_captura_nueva(frame)

        # 1. Ejecutar inferencia con YOLOv8
        resultados = self.modelo.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=self.CLASES_VEHICULOS,
            verbose=False,
        )

        cajas_vehiculos: List[Tuple[Polygon, float, int]] = []
        if resultados:
            resultado = resultados[0]
            if resultado.boxes is not None and len(resultado.boxes) > 0:
                xyxy = resultado.boxes.xyxy.cpu().numpy()
                confs = resultado.boxes.conf.cpu().numpy()
                clss = resultado.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                    caja_shapely = box(float(x1), float(y1), float(x2), float(y2))
                    if not self._es_falso_positivo_estatico(frame, caja_shapely):
                        cajas_vehiculos.append((caja_shapely, float(conf), int(cls_id)))

        self.vehiculos_detectados = len(cajas_vehiculos)

        # 2. Actualizar aprendizaje temporal solo cuando hay una captura nueva del servidor de Vigo
        if self.auto_calibrar and es_nuevo:
            self._actualizar_aprendizaje_temporal(frame, cajas_vehiculos)

        # 3. Evaluar el solapamiento entre cada plaza y los vehículos detectados
        plazas_libres = 0
        for plaza in self.plazas:
            plaza.ocupada = False
            plaza.max_solapamiento = 0.0
            area_plaza = plaza.poligono.area

            if area_plaza <= 0:
                continue

            for caja_vehiculo, _, _ in cajas_vehiculos:
                if not plaza.poligono.intersects(caja_vehiculo):
                    continue

                area_interseccion = plaza.poligono.intersection(caja_vehiculo).area
                ratio_solapamiento = area_interseccion / area_plaza

                if ratio_solapamiento > plaza.max_solapamiento:
                    plaza.max_solapamiento = ratio_solapamiento

                if ratio_solapamiento > self.overlap_threshold:
                    plaza.ocupada = True

            if not plaza.ocupada:
                plazas_libres += 1

        # 4. Renderizar resultado visual
        frame_procesado = self._dibujar_resultados(
            frame=frame,
            cajas_vehiculos=cajas_vehiculos,
        )

        return frame_procesado, plazas_libres

    def _dibujar_resultados(
        self,
        frame: np.ndarray,
        cajas_vehiculos: List[Tuple[Polygon, float, int]],
    ) -> np.ndarray:
        """
        Dibuja:
        - En VERDE semitransparente las plazas LIBRES.
        - En ROJO semitransparente las plazas OCUPADAS.
        - En ÁMBAR punteado los candidatos en observación (verificando si están aparcados o en semáforo).
        - En CIAN los vehículos en tránsito.
        """
        lienzo = frame.copy()
        overlay = lienzo.copy()

        # 1. Dibujar detecciones de vehículos y candidatos en observación
        if self.dibujar_detecciones:
            for caja_shapely, conf, cls_id in cajas_vehiculos:
                x1, y1, x2, y2 = map(int, caja_shapely.bounds)
                cv2.rectangle(
                    lienzo,
                    (x1, y1),
                    (x2, y2),
                    self.COLOR_DETECCION,
                    1,
                    cv2.LINE_AA,
                )
                nombre_cls = self.NOMBRES_CLASES.get(cls_id, str(cls_id))
                cv2.putText(
                    lienzo,
                    f"{nombre_cls} {conf:.2f}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    self.COLOR_DETECCION,
                    1,
                    cv2.LINE_AA,
                )

            # Mostrar candidatos que están siendo verificados antes de confirmar si son aparcamiento
            for cand in self._candidatos:
                x1, y1, x2, y2 = map(int, cand.poligono.bounds)
                cv2.rectangle(
                    lienzo,
                    (x1, y1),
                    (x2, y2),
                    self.COLOR_OBSERVACION,
                    1,
                    cv2.LINE_AA,
                )

        # 2. Dibujar polígonos de las plazas confirmadas (verde = libre, rojo = ocupada)
        for plaza in self.plazas:
            color = self.COLOR_OCUPADA if plaza.ocupada else self.COLOR_LIBRE
            pts = np.array(plaza.puntos, dtype=np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(
                lienzo,
                [pts],
                isClosed=True,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.35, lienzo, 0.65, 0, lienzo)

        # 3. Dibujar etiquetas de cada plaza
        for plaza in self.plazas:
            cx = int(sum(p[0] for p in plaza.puntos) / len(plaza.puntos))
            cy = int(sum(p[1] for p in plaza.puntos) / len(plaza.puntos))
            estado_txt = f"P{plaza.id}" if plaza.ocupada else f"P{plaza.id} LIBRE"

            cv2.putText(
                lienzo,
                estado_txt,
                (cx - 18, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                lienzo,
                estado_txt,
                (cx - 18, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                self.COLOR_TEXTO,
                1,
                cv2.LINE_AA,
            )

        return lienzo

    @property
    def total_plazas(self) -> int:
        return len(self.plazas)

    @property
    def plazas_ocupadas(self) -> int:
        return sum(1 for p in self.plazas if p.ocupada)
