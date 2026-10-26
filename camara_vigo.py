"""
Módulo para la captura y decodificación de imágenes desde las cámaras de tráfico del Concello de Vigo.
Incluye el catálogo oficial de cámaras, cabeceras completas de navegador de escritorio y
sistema de reintentos cortos (retries) optimizado tanto para ejecución local como en Streamlit Cloud.
"""

from typing import Optional
import json
import logging
import time

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    SSLError,
    Timeout,
)
import urllib3
from urllib3.util.retry import Retry

# Evitar warnings en consola si en algún entorno cloud se requiere fallback sin verificación SSL estricta
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuración básica de logging para trazabilidad
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# Catálogo oficial de cámaras de tráfico del Concello de Vigo (datos.vigo.org)
CAMARAS_VIGO: dict[str, str] = {
    "01": "Praza de América 1",
    "02": "Praza de España",
    "03": "Urzáiz - Gran Vía - Lepanto",
    "04": "Colón - Policarpo Sanz - García Barbón",
    "05": "Castelao - Grove",
    "06": "Aragón - Jenaro de la Fuente",
    "07": "Travesía de Vigo - Gregorio Espino",
    "08": "Gran Vía - Tarragona",
    "09": "Av. Florida - Caramuxo",
    "10": "Paseo de Alfonso XII",
    "11": "Venezuela - Granada",
    "12": "Av. Madrid",
    "13": "Pizarro - Vázquez Varela",
    "14": "Sanjurjo Badía - Buenos Aires",
    "15": "Castrelos - Antonio Palacios",
    "16": "Camelias - Ourense",
    "17": "García Barbón - Isaac Peral",
    "18": "Plaza de la Industria",
    "19": "Travesía de Vigo - Aragón",
    "20": "Av. Portanet - Val Miñor",
    "21": "Baixada Pontenova",
    "22": "Carral - Montero Ríos",
    "23": "Martínez Garrido - Ramón Nieto",
    "24": "Policarpo Sanz - Reconquista",
    "25": "Aeropuerto",
    "26": "Tomás Paredes - Av. Atlántida",
    "27": "Av. Europa - Samil",
    "28": "Martínez Garrido - Dr. Carracido",
    "29": "Florida - Manuel de Castro",
    "30": "Alfonso XIII - Cervantes",
    "31": "López Mora - Pintor Lugrís",
    "32": "Praza Isabel la Católica",
    "33": "Pi y Margall - Tomás Alonso",
    "34": "Areal - Miragaia",
    "35": "Avenida de Madrid - Seminario",
    "36": "Baixada ó Pontillón",
    "37": "Balaídos - Fragoso",
    "38": "Aragón - Padre Celso",
    "39": "Praza América 2",
    "40": "Hospital A. Cunqueiro",
    "41": "Gran Vía - Valle Inclán hacia Pr. América",
    "42": "Gran Vía - Valle Inclán hacia Pr. España",
    "43": "Colón - Urzáiz",
    "44": "Camelias - Romil",
    "45": "Emilia Pardo Bazán - A. Palacios",
    "46": "Gregorio Espino - Reiseñor",
    "47": "García Barbón - Serafín Avendaño",
    "48": "Camelias - Dr. Marañón",
    "49": "Vía Norte - Escultor Gregorio Fdez",
    "50": "Vía Norte - Urzáiz",
    "51": "Trav. de Vigo - Dr. Canoa",
    "52": "Av. Beiramar - Bouzas",
    "53": "Av. Beiramar - Berbés",
    "54": "Trav. de Vigo - Mestre Montes",
    "55": "Gran Vía - Venezuela",
    "001": "Túnel Orillamar - Berbés",
    "003": "Túnel Orillamar - Cánovas del Castillo",
    "011": "Túnel Orillamar - Areal",
    "616": "Túnel Areal - Sanjurjo Badía",
}

OPEN_DATA_CAMARAS_URL: str = "https://datos.vigo.org/data/trafico/camaras.json"


def obtener_catalogo_camaras(actualizar_online: bool = False, timeout: float = 5.0) -> dict[str, str]:
    """
    Devuelve el diccionario `{cam_id: nombre_ubicacion}` de todas las cámaras de Vigo.
    """
    catalogo = dict(CAMARAS_VIGO)
    if actualizar_online:
        try:
            resp = requests.get(
                OPEN_DATA_CAMARAS_URL,
                headers=CamaraVigo.DEFAULT_HEADERS,
                timeout=timeout,
            )
            resp.raise_for_status()
            try:
                datos = json.loads(resp.content.decode("cp1252"))
            except UnicodeDecodeError:
                datos = resp.json()

            for item in datos:
                cid = str(item.get("id", "")).strip()
                nombre = str(item.get("nombre", "")).strip()
                if cid and nombre:
                    catalogo[cid] = nombre
        except Exception as exc:
            logger.debug("Usando catálogo local de cámaras (Open Data no disponible: %s)", exc)

    return catalogo


def obtener_nombre_camara(cam_id: int | str) -> str:
    """Devuelve el nombre de la ubicación de una cámara a partir de su ID."""
    cid = str(cam_id).strip()
    return CAMARAS_VIGO.get(cid, f"Cámara {cid}")


class CamaraVigo:
    """
    Cliente robusto para obtener fotogramas en tiempo real de las cámaras de tráfico de Vigo.
    Incluye cabeceras completas de navegador de escritorio y reintentos automáticos ante
    latencia o rechazos puntuales desde servidores cloud (como Streamlit Cloud).
    """

    # Usamos directamente HTTPS para evitar el salto 301 Redirect adicional en cada petición
    BASE_URL: str = "https://camaras.vigo.org/webcam/camv2.php"
    FALLBACK_HTTP_URL: str = "http://camaras.vigo.org/webcam/camv2.php"

    # Diccionario completo de cabeceras simulando un navegador Chrome de escritorio real
    DEFAULT_HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://camaras.vigo.org/",
        "Origin": "https://camaras.vigo.org",
        "Connection": "keep-alive",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-origin",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    def __init__(
        self,
        cam_id: int | str = "05",
        scale_factor: float = 1.0,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        """
        Inicializa el conector de la cámara con sesión persistente y política de reintentos.

        Args:
            cam_id: Identificador de la cámara del Concello de Vigo.
            scale_factor: Factor de escalado opcional para la imagen resultante (por defecto 1.0).
            timeout: Tiempo máximo de espera por petición HTTP en segundos (por defecto 10.0).
            max_retries: Número máximo de reintentos cortos ante fallos transitorios de red.
        """
        if scale_factor <= 0:
            raise ValueError("El factor de escala (scale_factor) debe ser mayor que 0.")

        self.cam_id = str(cam_id)
        self.scale_factor = scale_factor
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))

        self._session = self._crear_sesion_con_reintentos()

    def _crear_sesion_con_reintentos(self) -> requests.Session:
        """
        Configura una sesión `requests.Session` con cabeceras de navegador y un `HTTPAdapter`
        con reintentos automáticos a nivel de transporte TCP/HTTP.
        """
        sesion = requests.Session()
        sesion.headers.update(self.DEFAULT_HEADERS)

        estrategia_retry = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=2,
            status=2,
            backoff_factor=0.35,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adaptador = HTTPAdapter(
            max_retries=estrategia_retry,
            pool_connections=4,
            pool_maxsize=8,
        )
        sesion.mount("https://", adaptador)
        sesion.mount("http://", adaptador)
        return sesion

    @property
    def ubicacion(self) -> str:
        """Devuelve la descripción de la ubicación de la cámara actual."""
        return obtener_nombre_camara(self.cam_id)

    def cambiar_camara(self, nuevo_cam_id: int | str, scale_factor: Optional[float] = None) -> None:
        """Cambia el identificador de la cámara activa reutilizando la misma sesión HTTP."""
        self.cam_id = str(nuevo_cam_id)
        if scale_factor is not None and scale_factor > 0:
            self.scale_factor = float(scale_factor)

    def obtener_frame(self, scale_factor: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Descarga el fotograma actual de la cámara con reintentos cortos y lo decodifica
        en formato BGR de OpenCV.

        Args:
            scale_factor: Factor de escala puntual (si no se especifica, usa el de la instancia).

        Returns:
            Un array de NumPy (H, W, 3) en formato BGR si la descarga fue exitosa,
            o None si tras los reintentos ocurrió algún error de red o decodificación.
        """
        escala = scale_factor if scale_factor is not None else self.scale_factor
        if escala <= 0:
            logger.error("Factor de escala inválido: %s", escala)
            return None

        # Tupla (connect_timeout, read_timeout) para no quedar bloqueado en handshakes lentos
        timeout_tupla = (min(5.0, self.timeout), self.timeout)

        for intento in range(1, self.max_retries + 1):
            # Timestamp en milisegundos como parámetro anti-caché renovado en cada intento
            timestamp_ms = int(time.time() * 1000)
            params = {
                "id": self.cam_id,
                "t": timestamp_ms,
            }

            try:
                try:
                    response = self._session.get(
                        self.BASE_URL,
                        params=params,
                        timeout=timeout_tupla,
                    )
                except SSLError:
                    # Fallback sin verificación estricta si el contenedor cloud carece de CA intermedia
                    response = self._session.get(
                        self.BASE_URL,
                        params=params,
                        timeout=timeout_tupla,
                        verify=False,
                    )

                response.raise_for_status()

                if not response.content or len(response.content) < 256:
                    logger.warning(
                        "Respuesta vacía o incompleta desde cámara ID %s (intento %d/%d).",
                        self.cam_id,
                        intento,
                        self.max_retries,
                    )
                    if intento < self.max_retries:
                        time.sleep(0.4 * intento)
                        continue
                    return None

                # Convertir los bytes recibidos en un buffer de NumPy y decodificar con OpenCV
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if frame is None:
                    logger.warning(
                        "No se pudo decodificar la imagen de la cámara ID %s (intento %d/%d).",
                        self.cam_id,
                        intento,
                        self.max_retries,
                    )
                    if intento < self.max_retries:
                        time.sleep(0.4 * intento)
                        continue
                    return None

                # Aplicar redimensionado si el factor de escala es distinto de 1.0
                if escala != 1.0:
                    interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
                    frame = cv2.resize(
                        frame,
                        dsize=None,
                        fx=escala,
                        fy=escala,
                        interpolation=interp,
                    )

                return frame

            except (Timeout, ConnectionError) as exc:
                logger.warning(
                    "Intento %d/%d fallido por red/timeout en cámara ID %s (%s): %s",
                    intento,
                    self.max_retries,
                    self.cam_id,
                    self.ubicacion,
                    exc,
                )
            except HTTPError as exc:
                logger.error(
                    "Error HTTP %s en cámara ID %s (intento %d/%d): %s",
                    exc.response.status_code if exc.response is not None else "desconocido",
                    self.cam_id,
                    intento,
                    self.max_retries,
                    exc,
                )
            except RequestException as exc:
                logger.error(
                    "Error en la petición para la cámara ID %s (intento %d/%d): %s",
                    self.cam_id,
                    intento,
                    self.max_retries,
                    exc,
                )
            except cv2.error as exc:
                logger.error(
                    "Error de OpenCV procesando el frame de la cámara ID %s: %s",
                    self.cam_id,
                    exc,
                )
                return None

            # Pequeña espera progresiva antes del siguiente reintento
            if intento < self.max_retries:
                time.sleep(0.45 * intento)

        return None

    def cerrar(self) -> None:
        """Cierra la sesión HTTP subyacente."""
        self._session.close()

    def __enter__(self) -> "CamaraVigo":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cerrar()


def descargar_frame_vigo(
    cam_id: int | str,
    scale_factor: float = 1.0,
    timeout: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Función auxiliar de conveniencia para realizar una captura puntual sin instanciar la clase.
    """
    with CamaraVigo(cam_id=cam_id, scale_factor=scale_factor, timeout=timeout) as cam:
        return cam.obtener_frame()


if __name__ == "__main__":
    CAM_ID_PRUEBA = "05"
    print(f"Probando captura de la cámara {CAM_ID_PRUEBA} ({obtener_nombre_camara(CAM_ID_PRUEBA)})...")

    with CamaraVigo(cam_id=CAM_ID_PRUEBA, scale_factor=1.0) as camara:
        imagen = camara.obtener_frame()
        if imagen is not None:
            alto, ancho, canales = imagen.shape
            print(f"Frame capturado con éxito: {ancho}x{alto} ({canales} canales)")
        else:
            print("No se pudo obtener el frame. Verifica tu conexión o el ID de la cámara.")
