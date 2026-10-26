"""
Módulo para la captura y decodificación de imágenes desde las cámaras de tráfico del Concello de Vigo.
Incluye:
- Catálogo oficial de las 59 cámaras de tráfico de Vigo.
- Conexión directa con cabeceras de navegador de escritorio.
- Enrutamiento automático por Proxies HTTP con nodo en España (country=ES) y pasarelas web
  alternativas cuando el servidor `camaras.vigo.org` bloquea IPs extranjeras (ej. Streamlit Cloud).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import time
from typing import Optional
import urllib.parse

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

# Suprimir advertencias TLS cuando se conecta a través de proxies/pasarelas sin CA intermedia
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    """Devuelve el diccionario `{cam_id: nombre_ubicacion}` de todas las cámaras de Vigo."""
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


class GestorProxiesEspana:
    """
    Descubridor y gestor en caché de proxies HTTP públicos ubicados en España (country=ES)
    y pasarelas web alternativas para sortear bloqueos geográficos de IPs cloud extranjeras.
    """

    FUENTES_PROXIES_ES: list[str] = [
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=ES&ssl=all&anonymity=all",
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=4000&country=PT,FR,ES&ssl=all&anonymity=all",
    ]
    GEONODE_ES_URL: str = (
        "https://proxylist.geonode.com/api/proxy-list?limit=30&page=1&sort_by=lastChecked&sort_type=desc&country=ES"
    )

    _proxy_ganador: Optional[str] = None
    _lista_candidatos: list[str] = []
    _ultima_actualizacion_lista: float = 0.0
    TTL_LISTA_SEGUNDOS: float = 300.0  # Refrescar lista de candidatos cada 5 minutos

    @classmethod
    def obtener_candidatos_es(cls) -> list[str]:
        """Obtiene una lista actualizada de proxies públicos en España / Península Ibérica."""
        ahora = time.time()
        if cls._lista_candidatos and (ahora - cls._ultima_actualizacion_lista) < cls.TTL_LISTA_SEGUNDOS:
            return cls._lista_candidatos

        candidatos: list[str] = []

        # 1. Consultar GeoNode (nodos en España)
        try:
            r_geo = requests.get(cls.GEONODE_ES_URL, timeout=4.0)
            if r_geo.status_code == 200:
                for item in r_geo.json().get("data", []):
                    ip = item.get("ip")
                    port = item.get("port")
                    protos = item.get("protocols", [])
                    if ip and port and ("http" in protos or "https" in protos):
                        candidatos.append(f"http://{ip}:{port}")
        except Exception:
            pass

        # 2. Consultar ProxyScrape (nodos ES y respaldo ibérico/europeo)
        for url_fuente in cls.FUENTES_PROXIES_ES:
            try:
                r_ps = requests.get(url_fuente, timeout=4.0)
                if r_ps.status_code == 200:
                    for linea in r_ps.text.splitlines():
                        limpia = linea.strip()
                        if limpia and ":" in limpia:
                            proxy_url = limpia if limpia.startswith("http") else f"http://{limpia}"
                            if proxy_url not in candidatos:
                                candidatos.append(proxy_url)
            except Exception:
                pass

        if candidatos:
            cls._lista_candidatos = candidatos[:25]
            cls._ultima_actualizacion_lista = ahora

        return cls._lista_candidatos

    @classmethod
    def invalidar_proxy_ganador(cls) -> None:
        cls._proxy_ganador = None


class CamaraVigo:
    """
    Cliente robusto para obtener fotogramas en tiempo real de las cámaras de tráfico de Vigo.

    Estrategia de conexión multinivel (ideal para Streamlit Cloud y escritorio):
    1. Si se proporciona un `proxy_url` explícito (o variable de entorno `VIGO_PROXY_URL`),
       enruta la petición a través de ese proxy.
    2. Intenta conexión directa rápida con cabeceras de navegador (óptimo cuando se ejecuta en España).
    3. Si la conexión directa es bloqueada por IP extranjera (Streamlit Cloud), enruta automáticamente
       a través de pasarelas web y un pool en paralelo de proxies HTTP con nodo en España (`GestorProxiesEspana`),
       recordando en caché el nodo que responda más rápido.
    """

    BASE_URL: str = "https://camaras.vigo.org/webcam/camv2.php"
    HTTP_URL: str = "http://camaras.vigo.org/webcam/camv2.php"

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
        timeout: float = 8.0,
        max_retries: int = 2,
        usar_proxy_fallback: bool = True,
        proxy_url: Optional[str] = None,
    ) -> None:
        if scale_factor <= 0:
            raise ValueError("El factor de escala (scale_factor) debe ser mayor que 0.")

        self.cam_id = str(cam_id)
        self.scale_factor = scale_factor
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.usar_proxy_fallback = usar_proxy_fallback
        self.proxy_url = (proxy_url or os.environ.get("VIGO_PROXY_URL", "")).strip() or None

        # Si ya detectamos que la IP local está bloqueada (ej. Streamlit Cloud),
        # saltamos directamente al enrutamiento por pasarela/proxy en las siguientes llamadas
        self._conexion_directa_bloqueada: bool = False
        self.ultimo_metodo_conexion: str = "Directo"

        self._session = self._crear_sesion()

    def _crear_sesion(self) -> requests.Session:
        sesion = requests.Session()
        sesion.headers.update(self.DEFAULT_HEADERS)

        estrategia_retry = Retry(
            total=self.max_retries,
            connect=1,
            read=1,
            status=1,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adaptador = HTTPAdapter(
            max_retries=estrategia_retry,
            pool_connections=6,
            pool_maxsize=12,
        )
        sesion.mount("https://", adaptador)
        sesion.mount("http://", adaptador)
        return sesion

    @property
    def ubicacion(self) -> str:
        return obtener_nombre_camara(self.cam_id)

    def cambiar_camara(self, nuevo_cam_id: int | str, scale_factor: Optional[float] = None) -> None:
        self.cam_id = str(nuevo_cam_id)
        if scale_factor is not None and scale_factor > 0:
            self.scale_factor = float(scale_factor)

    def configurar_proxy_manual(self, proxy_url: Optional[str]) -> None:
        """Permite establecer o limpiar un proxy HTTP personalizado en caliente."""
        limpio = (proxy_url or "").strip()
        if limpio and not limpio.startswith("http"):
            limpio = f"http://{limpio}"
        self.proxy_url = limpio or None

    @staticmethod
    def _es_contenido_jpeg_valido(content: bytes) -> bool:
        """Verifica que el buffer recibido sea una imagen válida (mínimo 512 bytes y cabecera JPEG/PNG)."""
        if not content or len(content) < 512:
            return False
        # Cabecera JPEG (FF D8) o PNG (89 50 4E 47)
        return content[:2] == b"\xff\xd8" or content[:4] == b"\x89PNG"

    def _decodificar_y_escalar(self, content: bytes, escala: float) -> Optional[np.ndarray]:
        """Convierte los bytes de imagen en un array BGR de OpenCV y aplica el escalado."""
        try:
            image_array = np.frombuffer(content, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is None:
                return None

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
        except cv2.error as exc:
            logger.error("Error de OpenCV decodificando imagen de cámara %s: %s", self.cam_id, exc)
            return None

    def _descargar_directo(self, timestamp_ms: int) -> Optional[bytes]:
        """Intenta descargar el fotograma directamente desde el servidor del Concello de Vigo."""
        params = {"id": self.cam_id, "t": timestamp_ms}
        timeout_directo = (3.5, min(6.0, self.timeout))

        try:
            try:
                resp = self._session.get(self.BASE_URL, params=params, timeout=timeout_directo)
            except SSLError:
                resp = self._session.get(
                    self.BASE_URL, params=params, timeout=timeout_directo, verify=False
                )

            if resp.status_code == 200 and self._es_contenido_jpeg_valido(resp.content):
                self.ultimo_metodo_conexion = "Directo (ES)"
                return resp.content
        except RequestException as exc:
            logger.debug("Conexión directa no disponible para cámara %s: %s", self.cam_id, exc)

        return None

    def _probar_un_proxy(self, proxy_url: str, url_destino: str, timeout_s: float = 4.5) -> Optional[bytes]:
        """Realiza una petición a la cámara a través de un único proxy HTTP."""
        proxies = {"http": proxy_url, "https": proxy_url}
        try:
            resp = requests.get(
                url_destino,
                headers=self.DEFAULT_HEADERS,
                proxies=proxies,
                timeout=timeout_s,
                verify=False,
            )
            if resp.status_code == 200 and self._es_contenido_jpeg_valido(resp.content):
                return resp.content
        except Exception:
            pass
        return None

    def _descargar_via_pasarela_o_proxy_es(self, timestamp_ms: int) -> Optional[bytes]:
        """
        Enruta la petición cuando la IP de origen (ej. Streamlit Cloud) está bloqueada:
        1. Prueba primero el proxy manual si el usuario configuró uno.
        2. Prueba pasarelas web de retransmisión rápida.
        3. Prueba el último proxy español ganador en caché (`GestorProxiesEspana._proxy_ganador`).
        4. Si no hay ganador o caducó, lanza una sonda concurrente sobre el pool de proxies españoles (`country=ES`).
        """
        url_http_target = f"{self.HTTP_URL}?id={self.cam_id}&t={timestamp_ms}"
        url_https_target = f"{self.BASE_URL}?id={self.cam_id}&t={timestamp_ms}"

        # 1. Proxy manual configurado por el usuario o variable de entorno
        if self.proxy_url:
            contenido = self._probar_un_proxy(self.proxy_url, url_http_target, timeout_s=6.0)
            if contenido is not None:
                self.ultimo_metodo_conexion = f"Proxy Manual ({self.proxy_url})"
                return contenido

        # 2. Proxy español ganador ya verificado en caché
        if GestorProxiesEspana._proxy_ganador:
            proxy_cache = GestorProxiesEspana._proxy_ganador
            contenido = self._probar_un_proxy(proxy_cache, url_http_target, timeout_s=4.5)
            if contenido is not None:
                self.ultimo_metodo_conexion = f"Proxy España ({proxy_cache})"
                return contenido
            GestorProxiesEspana.invalidar_proxy_ganador()

        # 3. Pasarelas públicas de retransmisión HTTP/HTTPS
        url_codificada = urllib.parse.quote(url_https_target, safe="")
        url_http_codificada = urllib.parse.quote(url_http_target, safe="")
        pasarelas = [
            ("Pasarela CodeTabs", f"https://api.codetabs.com/v1/proxy?quest={url_http_codificada}"),
            ("Pasarela CorsProxy", f"https://corsproxy.io/?{url_codificada}"),
            ("Pasarela AllOrigins", f"https://api.allorigins.win/raw?url={url_http_codificada}"),
        ]

        for nombre_pasarela, url_pasarela in pasarelas:
            try:
                resp = self._session.get(url_pasarela, timeout=(3.5, 6.0), verify=False)
                if resp.status_code == 200 and self._es_contenido_jpeg_valido(resp.content):
                    self.ultimo_metodo_conexion = nombre_pasarela
                    return resp.content
            except Exception:
                continue

        # 4. Sonda en paralelo sobre el pool de proxies públicos en España (country=ES)
        candidatos_es = GestorProxiesEspana.obtener_candidatos_es()
        if candidatos_es:
            logger.info(
                "Buscando nodo proxy activo en España entre %d candidatos...",
                len(candidatos_es),
            )
            with ThreadPoolExecutor(max_workers=8) as executor:
                futuros = {
                    executor.submit(self._probar_un_proxy, p_url, url_http_target, 4.0): p_url
                    for p_url in candidatos_es[:16]
                }
                for futuro in as_completed(futuros):
                    p_url = futuros[futuro]
                    try:
                        datos_img = futuro.result()
                        if datos_img is not None:
                            GestorProxiesEspana._proxy_ganador = p_url
                            self.ultimo_metodo_conexion = f"Proxy España ({p_url})"
                            logger.info("Nodo proxy español conectado con éxito: %s", p_url)
                            return datos_img
                    except Exception:
                        continue

        return None

    def obtener_frame(self, scale_factor: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Descarga el fotograma actual de la cámara (directamente o vía proxy/pasarela si hay
        bloqueo de IP en la nube) y lo decodifica en un array BGR de OpenCV.
        """
        escala = scale_factor if scale_factor is not None else self.scale_factor
        if escala <= 0:
            logger.error("Factor de escala inválido: %s", escala)
            return None

        timestamp_ms = int(time.time() * 1000)
        contenido_imagen: Optional[bytes] = None

        # Paso 1: Si el usuario no forzó un proxy manual y la conexión directa no está marcada como bloqueada
        if not self.proxy_url and not self._conexion_directa_bloqueada:
            contenido_imagen = self._descargar_directo(timestamp_ms)
            if contenido_imagen is None and self.usar_proxy_fallback:
                logger.info(
                    "Conexión directa a camaras.vigo.org sin respuesta (posible bloqueo de IP cloud). "
                    "Activando enrutamiento por pasarela / proxy en España..."
                )
                self._conexion_directa_bloqueada = True

        # Paso 2: Enrutamiento mediante Proxy en España o Pasarela alternativa
        if contenido_imagen is None and (self.usar_proxy_fallback or self.proxy_url):
            contenido_imagen = self._descargar_via_pasarela_o_proxy_es(timestamp_ms)
            # Si falló el proxy pero antes no habíamos probado directo en este ciclo, reintentar directo una vez
            if contenido_imagen is None and self._conexion_directa_bloqueada:
                contenido_imagen = self._descargar_directo(timestamp_ms)
                if contenido_imagen is not None:
                    self._conexion_directa_bloqueada = False

        if contenido_imagen is None:
            logger.error(
                "No se pudo obtener imagen de la cámara ID %s (%s) ni por vía directa ni por proxy.",
                self.cam_id,
                self.ubicacion,
            )
            return None

        return self._decodificar_y_escalar(contenido_imagen, escala)

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
    timeout: float = 8.0,
    proxy_url: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Función auxiliar de conveniencia para realizar una captura puntual."""
    with CamaraVigo(
        cam_id=cam_id,
        scale_factor=scale_factor,
        timeout=timeout,
        proxy_url=proxy_url,
    ) as cam:
        return cam.obtener_frame()


if __name__ == "__main__":
    CAM_ID_PRUEBA = "05"
    print(f"Probando captura de la cámara {CAM_ID_PRUEBA} ({obtener_nombre_camara(CAM_ID_PRUEBA)})...")

    with CamaraVigo(cam_id=CAM_ID_PRUEBA, scale_factor=1.0) as camara:
        imagen = camara.obtener_frame()
        if imagen is not None:
            alto, ancho, canales = imagen.shape
            print(
                f"Frame capturado con éxito ({camara.ultimo_metodo_conexion}): "
                f"{ancho}x{alto} ({canales} canales)"
            )
        else:
            print("No se pudo obtener el frame.")
