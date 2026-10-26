"""
Herramienta interactiva con OpenCV para calibrar y delimitar las plazas de aparcamiento
sobre un fotograma de las cámaras de tráfico del Concello de Vigo.

Incluye barra superior externa con botones clicables con el ratón (además de atajos de teclado)
para que nunca tape la imagen de la cámara ni quede oculta por la barra de tareas de Windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import List, Tuple
import sys

import cv2
import numpy as np

from camara_vigo import CamaraVigo, obtener_nombre_camara, CAMARAS_VIGO

# Tipos auxiliares
Punto = Tuple[int, int]
Poligono = List[Punto]


class CalibradorPlazas:
    """
    Gestiona la interfaz interactiva para dibujar polígonos de 4 puntos
    que representan plazas de aparcamiento y exportarlos a `plazas_vigo.json`.
    """

    WINDOW_NAME = "Parking Vision - Calibrador de Plazas"
    VERTICES_POR_PLAZA = 4
    ALTO_BARRA_SUPERIOR = 92  # Franja externa superior (no tapa la imagen de la cámara)

    # Colores en formato BGR
    COLOR_PLAZA_COMPLETA = (0, 200, 0)       # Verde
    COLOR_PLAZA_ACTUAL = (0, 215, 255)       # Amarillo/Ámbar
    COLOR_VERTICE = (0, 0, 255)              # Rojo
    COLOR_TEXTO = (255, 255, 255)            # Blanco
    COLOR_FONDO_BARRA = (24, 26, 32)         # Gris oscuro

    def __init__(
        self,
        cam_id: str = "05",
        scale_factor: float = 2.0,
        archivo_salida: str | Path = "plazas_vigo.json",
    ) -> None:
        self.cam_id = str(cam_id)
        self.ubicacion = obtener_nombre_camara(self.cam_id)
        self.scale_factor = float(scale_factor)
        self.archivo_salida = Path(archivo_salida)

        self.camara = CamaraVigo(cam_id=self.cam_id, scale_factor=self.scale_factor)
        self.frame_base: np.ndarray | None = None

        # Estado del dibujo
        self.plazas: List[Poligono] = []
        self.puntos_actuales: Poligono = []
        self.pos_raton: Punto | None = None
        self.solicitar_salida: bool = False
        self.mensaje_estado_temp: str = ""
        self.tiempo_mensaje_temp: float = 0.0

        # Definición de botones clicables en la barra superior: (x1, y1, x2, y2, accion, texto, color_bgr)
        self.botones: list[tuple[int, int, int, int, str, str, tuple[int, int, int]]] = [
            (16, 42, 255, 82, "guardar", "GUARDAR PLAZAS [Tecla S]", (45, 150, 45)),
            (268, 42, 505, 82, "deshacer", "DESHACER PUNTO [Clic Der]", (50, 110, 180)),
            (518, 42, 745, 82, "refrescar", "NUEVA FOTO [Tecla R]", (140, 95, 35)),
            (758, 42, 965, 82, "limpiar", "BORRAR TODO [Tecla C]", (50, 50, 175)),
            (978, 42, 1190, 82, "salir", "SALIR [Tecla Q / ESC]", (65, 65, 75)),
        ]

        self._cargar_plazas_existentes()

    def _mostrar_aviso_temporal(self, texto: str, duracion: float = 3.5) -> None:
        self.mensaje_estado_temp = texto
        self.tiempo_mensaje_temp = time.time() + duracion

    def _cargar_plazas_existentes(self) -> None:
        """
        Si ya existen plazas guardadas para esta cámara en el JSON, las carga inicialmente
        y adapta sus coordenadas si cambió el factor de escala.
        """
        if not self.archivo_salida.exists():
            return
        try:
            datos = json.loads(self.archivo_salida.read_text(encoding="utf-8"))
            plazas_raw = []
            escala_previa = self.scale_factor

            camaras_dict = datos.get("camaras", {})
            if isinstance(camaras_dict, dict) and self.cam_id in camaras_dict:
                info_cam = camaras_dict[self.cam_id]
                plazas_raw = info_cam.get("plazas", [])
                escala_previa = float(info_cam.get("scale_factor", self.scale_factor))
            elif str(datos.get("cam_id")) == self.cam_id:
                plazas_raw = datos.get("plazas", [])
                escala_previa = float(datos.get("scale_factor", self.scale_factor))

            ratio = self.scale_factor / escala_previa if escala_previa > 0 else 1.0

            for item in plazas_raw:
                coords = item.get("poligono", []) if isinstance(item, dict) else item
                if len(coords) == self.VERTICES_POR_PLAZA:
                    self.plazas.append(
                        [(int(round(x * ratio)), int(round(y * ratio))) for x, y in coords]
                    )

            if self.plazas:
                print(
                    f"[i] Se cargaron {len(self.plazas)} plaza(s) previas para la cámara "
                    f"[{self.cam_id}] {self.ubicacion}."
                )
        except Exception:
            pass

    def descargar_fotograma(self) -> bool:
        """Descarga un fotograma actualizado desde la cámara de Vigo."""
        print(
            f"[*] Descargando fotograma de prueba (Cámara [{self.cam_id}] {self.ubicacion}, "
            f"Escala: {self.scale_factor}x)..."
        )
        frame = self.camara.obtener_frame()
        if frame is None:
            print(
                f"[!] Error: No se pudo obtener el fotograma de la cámara '{self.cam_id}'."
            )
            return False

        self.frame_base = frame
        alto, ancho = frame.shape[:2]
        print(f"[+] Fotograma descargado correctamente ({ancho}x{alto} px).")
        self._mostrar_aviso_temporal("Fotograma actualizado", 2.0)
        return True

    def _deshacer_ultimo_punto(self) -> None:
        if self.puntos_actuales:
            punto_eliminado = self.puntos_actuales.pop()
            print(
                f"  - Punto {punto_eliminado} deshecho. "
                f"Quedan {len(self.puntos_actuales)}/{self.VERTICES_POR_PLAZA} puntos."
            )
        elif self.plazas:
            plaza_eliminada = self.plazas.pop()
            self.puntos_actuales = plaza_eliminada[:-1]
            ultimo_pt = plaza_eliminada[-1]
            print(
                f"  - Reabierta la última plaza (eliminado vértice {ultimo_pt}). "
                f"Quedan {len(self.puntos_actuales)}/{self.VERTICES_POR_PLAZA} puntos."
            )
        else:
            print("  [i] No hay puntos ni plazas para deshacer.")

    def _callback_raton(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        """
        Manejador de eventos de ratón:
        - Si el clic es en la barra superior (`y < ALTO_BARRA_SUPERIOR`), pulsa el botón correspondiente.
        - Si el clic es sobre la imagen (`y >= ALTO_BARRA_SUPERIOR`), traduce `y_img = y - ALTO_BARRA_SUPERIOR`
          y añade el vértice de la plaza con total precisión.
        """
        if self.frame_base is None:
            return

        alto_img, ancho_img = self.frame_base.shape[:2]

        if event == cv2.EVENT_MOUSEMOVE:
            if y >= self.ALTO_BARRA_SUPERIOR:
                x_img = max(0, min(ancho_img - 1, x))
                y_img = max(0, min(alto_img - 1, y - self.ALTO_BARRA_SUPERIOR))
                self.pos_raton = (x_img, y_img)
            else:
                self.pos_raton = None

        elif event == cv2.EVENT_LBUTTONDOWN:
            # 1. ¿Hizo clic en alguno de los botones de la barra superior?
            if y < self.ALTO_BARRA_SUPERIOR:
                for x1, y1, x2, y2, accion, _, _ in self.botones:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        if accion == "guardar":
                            self.guardar_json()
                        elif accion == "deshacer":
                            self._deshacer_ultimo_punto()
                        elif accion == "refrescar":
                            self.descargar_fotograma()
                        elif accion == "limpiar":
                            self.plazas.clear()
                            self.puntos_actuales.clear()
                            self._mostrar_aviso_temporal("Todas las plazas borradas", 2.5)
                        elif accion == "salir":
                            self.solicitar_salida = True
                        return
                return

            # 2. Hizo clic dentro de la imagen de la cámara: añadir esquina de plaza
            x_img = max(0, min(ancho_img - 1, x))
            y_img = max(0, min(alto_img - 1, y - self.ALTO_BARRA_SUPERIOR))

            self.puntos_actuales.append((x_img, y_img))
            num_pt = len(self.puntos_actuales)
            print(f"  + Punto {num_pt}/{self.VERTICES_POR_PLAZA} añadido en ({x_img}, {y_img})")

            if num_pt == self.VERTICES_POR_PLAZA:
                self.plazas.append(self.puntos_actuales.copy())
                self.puntos_actuales.clear()
                print(
                    f"[+] Plaza #{len(self.plazas)} completada. "
                    f"(Total plazas: {len(self.plazas)})"
                )

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._deshacer_ultimo_punto()

    def _renderizar_vista(self) -> np.ndarray:
        """
        Dibuja los polígonos sobre la imagen y añade en la parte superior una barra externa
        con botones grandes clicables (sin tapar ni un solo píxel de la cámara).
        """
        if self.frame_base is None:
            raise RuntimeError("No hay ningún fotograma cargado para renderizar.")

        lienzo = self.frame_base.copy()
        overlay = lienzo.copy()

        # 1. Dibujar polígonos de plazas ya completadas
        for idx, poligono in enumerate(self.plazas, start=1):
            pts = np.array(poligono, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], self.COLOR_PLAZA_COMPLETA)
            cv2.polylines(
                lienzo,
                [pts],
                isClosed=True,
                color=self.COLOR_PLAZA_COMPLETA,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            cx = int(sum(p[0] for p in poligono) / len(poligono))
            cy = int(sum(p[1] for p in poligono) / len(poligono))
            cv2.putText(
                lienzo,
                f"P{idx}",
                (cx - 14, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                lienzo,
                f"P{idx}",
                (cx - 14, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                self.COLOR_TEXTO,
                2,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.28, lienzo, 0.72, 0, lienzo)

        # 2. Dibujar puntos y líneas de la plaza actual en construcción
        if self.puntos_actuales:
            for i, pt in enumerate(self.puntos_actuales):
                cv2.circle(
                    lienzo,
                    pt,
                    radius=5,
                    color=self.COLOR_VERTICE,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
                if i > 0:
                    cv2.line(
                        lienzo,
                        self.puntos_actuales[i - 1],
                        pt,
                        self.COLOR_PLAZA_ACTUAL,
                        2,
                        cv2.LINE_AA,
                    )

            if self.pos_raton is not None:
                cv2.line(
                    lienzo,
                    self.puntos_actuales[-1],
                    self.pos_raton,
                    self.COLOR_PLAZA_ACTUAL,
                    1,
                    cv2.LINE_AA,
                )

        # 3. Añadir barra superior EXTERNA con botones clicables
        return self._agregar_barra_superior(lienzo)

    def _agregar_barra_superior(self, imagen: np.ndarray) -> np.ndarray:
        """
        Añade un panel superior fuera del área de la cámara con el estado y 5 botones grandes
        que se pueden pulsar tanto con el ratón como con el teclado.
        """
        _, ancho_img = imagen.shape[:2]

        vista = cv2.copyMakeBorder(
            imagen,
            top=self.ALTO_BARRA_SUPERIOR,
            bottom=0,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=self.COLOR_FONDO_BARRA,
        )

        # Línea separadora entre la barra y la imagen de la cámara
        cv2.line(
            vista,
            (0, self.ALTO_BARRA_SUPERIOR - 1),
            (ancho_img, self.ALTO_BARRA_SUPERIOR - 1),
            (80, 90, 115),
            2,
        )

        # Texto de estado en la primera línea de la barra superior
        linea_info = (
            f"Camara [{self.cam_id}] {self.ubicacion}   |   "
            f"Plazas completadas: {len(self.plazas)}   |   "
            f"Esquinas plaza actual: {len(self.puntos_actuales)}/{self.VERTICES_POR_PLAZA} "
            "(haz clic izquierdo en 4 esquinas de cada plaza)"
        )
        cv2.putText(
            vista,
            linea_info,
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (120, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Mostrar aviso de confirmación si acaba de guardar
        if self.mensaje_estado_temp and time.time() < self.tiempo_mensaje_temp:
            cv2.putText(
                vista,
                self.mensaje_estado_temp,
                (ancho_img - 360, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (80, 255, 120),
                2,
                cv2.LINE_AA,
            )

        # Dibujar los botones clicables en la segunda fila de la barra superior
        for x1, y1, x2, y2, _, texto, color_fondo in self.botones:
            cv2.rectangle(vista, (x1, y1), (x2, y2), color_fondo, thickness=-1)
            cv2.rectangle(vista, (x1, y1), (x2, y2), (200, 205, 215), thickness=1)
            cv2.putText(
                vista,
                texto,
                (x1 + 12, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return vista

    def guardar_json(self) -> bool:
        """
        Guarda la configuración calibrada en el archivo JSON preservando las demás cámaras.
        """
        if self.puntos_actuales:
            print(
                f"[!] Aviso: Hay {len(self.puntos_actuales)} punto(s) sin completar "
                "que no se incluirán como plaza."
            )

        alto, ancho = (
            self.frame_base.shape[:2] if self.frame_base is not None else (0, 0)
        )

        plazas_serializadas = [
            {
                "id": idx,
                "poligono": [[int(x), int(y)] for x, y in poligono],
            }
            for idx, poligono in enumerate(self.plazas, start=1)
        ]

        camaras_guardadas = {}
        if self.archivo_salida.exists():
            try:
                previo = json.loads(self.archivo_salida.read_text(encoding="utf-8"))
                if isinstance(previo.get("camaras"), dict):
                    camaras_guardadas = previo["camaras"]
                elif previo.get("cam_id") and previo.get("plazas"):
                    cid_prev = str(previo["cam_id"])
                    camaras_guardadas[cid_prev] = {
                        "cam_id": cid_prev,
                        "ubicacion": obtener_nombre_camara(cid_prev),
                        "scale_factor": previo.get("scale_factor", 1.0),
                        "resolucion": previo.get("resolucion", {}),
                        "total_plazas": len(previo.get("plazas", [])),
                        "plazas": previo.get("plazas", []),
                    }
            except Exception:
                pass

        info_actual = {
            "cam_id": self.cam_id,
            "ubicacion": self.ubicacion,
            "scale_factor": self.scale_factor,
            "resolucion": {"ancho": ancho, "alto": alto},
            "total_plazas": len(plazas_serializadas),
            "plazas": plazas_serializadas,
        }
        camaras_guardadas[self.cam_id] = info_actual

        datos = {
            "cam_id": self.cam_id,
            "ubicacion": self.ubicacion,
            "scale_factor": self.scale_factor,
            "resolucion": {"ancho": ancho, "alto": alto},
            "total_plazas": len(plazas_serializadas),
            "plazas": plazas_serializadas,
            "camaras": camaras_guardadas,
        }

        try:
            self.archivo_salida.write_text(
                json.dumps(datos, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            self._mostrar_aviso_temporal(
                f"GUARDADO OK ({len(self.plazas)} plazas)", 4.0
            )
            print(
                f"\n[OK] Se han guardado {len(self.plazas)} plaza(s) para [{self.cam_id}] "
                f"{self.ubicacion} en '{self.archivo_salida.resolve()}'."
            )
            return True
        except OSError as exc:
            print(f"\n[!] Error al guardar el archivo JSON: {exc}")
            return False

    def ejecutar(self) -> None:
        """Inicia el bucle principal de la interfaz de calibración OpenCV."""
        if not self.descargar_fotograma():
            self.camara.cerrar()
            return

        titulo_ventana = f"{self.WINDOW_NAME} - [{self.cam_id}] {self.ubicacion}"

        # Tamaño inicial cómodo (1050x640) para que quepa entero en cualquier portátil
        cv2.namedWindow(titulo_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(titulo_ventana, 1050, 640)
        cv2.setMouseCallback(titulo_ventana, self._callback_raton)

        try:
            while not self.solicitar_salida:
                vista = self._renderizar_vista()
                cv2.imshow(titulo_ventana, vista)

                key = cv2.waitKey(20) & 0xFF

                if cv2.getWindowProperty(titulo_ventana, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("s"), ord("S")):
                    self.guardar_json()
                elif key in (ord("r"), ord("R")):
                    self.descargar_fotograma()
                elif key in (ord("c"), ord("C")):
                    self.plazas.clear()
                    self.puntos_actuales.clear()
                    self._mostrar_aviso_temporal("Todas las plazas borradas", 2.5)

        finally:
            self.camara.cerrar()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrador interactivo de plazas de aparcamiento para las cámaras de Vigo."
    )
    parser.add_argument("--cam-id", type=str, default="05")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="plazas_vigo.json")
    parser.add_argument("--listar", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.listar:
        for cid, ubicacion in sorted(CAMARAS_VIGO.items()):
            print(f"  ID {cid:<4} -> {ubicacion}")
        sys.exit(0)

    calibrador = CalibradorPlazas(
        cam_id=args.cam_id,
        scale_factor=args.scale,
        archivo_salida=args.output,
    )
    calibrador.ejecutar()
