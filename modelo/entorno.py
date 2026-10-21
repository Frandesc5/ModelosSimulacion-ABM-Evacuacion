"""Recinto, humo dinámico, accesibilidad y mapa de distancias BFS.

Las matrices se indexan siempre como matriz[y, x]. La vecindad es de cuatro
celdas (Von Neumann); el BFS ignora temporalmente dónde están las personas.
"""
from collections import deque
import numpy as np

VECINOS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Entorno:
    """Recinto en grilla con geometría fija y humo como estado dinámico.

    Las paredes y las salidas se mantienen fijas. Las concentraciones H[y, x]
    se actualizan cada paso. La accesibilidad y las distancias BFS son
    informaciones DERIVADAS del humo y de la geometría: no son nuevas
    variables físicas independientes.
    """

    def __init__(self, configuracion):
        """Construye el recinto y su estado ambiental inicial.

        Parámetros:
            configuracion (Configuracion): Dimensiones del recinto,
                ubicación del foco y parámetros del humo.

        Devuelve:
            None: Inicializa paredes, salidas, fuentes y una matriz
            de humo con valores cero.

        Notas:
            El recinto incluye dos salidas de dos celdas y obstáculos
            interiores. La geometría permanece fija.
        """
        self.configuracion = configuracion
        self.ancho = configuracion.ancho
        self.alto = configuracion.alto
        self.paredes = np.zeros((self.alto, self.ancho), dtype=bool)
        self.paredes[0, :] = True
        self.paredes[-1, :] = True
        self.paredes[:, 0] = True
        self.paredes[:, -1] = True
        self.salidas = [(0, 7), (0, 8), (self.ancho - 1, 16), (self.ancho - 1, 17)]
        for x, y in self.salidas:
            self.paredes[y, x] = False
        self.paredes[9:12, 16:18] = True
        self.paredes[14:17, 23:25] = True
        self.humo = np.zeros((self.alto, self.ancho), dtype=float)
        self.fuentes = self._crear_fuentes()

    def _crear_fuentes(self):
        """Obtiene las celdas que emiten humo en el escenario.

        Devuelve:
            list[tuple[int, int]]: Coordenadas (x, y) de las celdas
            emisoras no ocupadas por paredes.

        Notas:
            posicion_foco selecciona el foco izquierdo, central
            o derecho. Cada foco se representa con cuatro celdas.
        """
        ubicaciones = {
            "izquierda": [(3, 7), (3, 8), (4, 7), (4, 8)],
            "centro": [(19, 11), (20, 11), (19, 12), (20, 12)],
            "derecha": [(34, 16), (34, 17), (35, 16), (35, 17)],
        }
        fuentes = ubicaciones[self.configuracion.posicion_foco]
        return [posicion for posicion in fuentes if not self.paredes[posicion[1], posicion[0]]]

    def dentro(self, x, y):
        """Comprueba si una posición pertenece a la grilla.

        Parámetros:
            x (int): Columna que se desea comprobar.
            y (int): Fila que se desea comprobar.

        Devuelve:
            bool: True si la posición está dentro de los límites;
            False en caso contrario.

        Notas:
            Una celda dentro del recinto puede ser una pared.
        """
        return 0 <= x < self.ancho and 0 <= y < self.alto

    def es_transitable(self, x, y):
        """Determina si se permite ingresar a una celda.

        Parámetros:
            x (int): Columna de la celda.
            y (int): Fila de la celda.

        Devuelve:
            bool: True si la celda no es pared ni foco de incendio,
            y no está bloqueada por humo; False en caso contrario.

        Notas:
            Las fuentes de humo representan un incendio fijo y no
            permiten el ingreso, incluso antes de alcanzar el umbral.
            Las salidas nunca se bloquean por humo. No comprueba
            si la celda está ocupada por una persona.
            Un agente que ya esté en una celda bloqueada puede
            intentar salir de ella.
        """
        if not self.dentro(x, y) or self.paredes[y, x]:
            return False
        if (x, y) in self.fuentes:
            return False  # El foco del incendio no permite el ingreso.
        if (x, y) in self.salidas:
            return True  # Simplificación: las salidas no se cierran por humo.
        if self.configuracion.usar_humo and self.humo[y, x] >= self.configuracion.umbral_bloqueo:
            return False
        return True

    def actualizar_humo(self):
        """Actualiza simultáneamente el campo de humo un paso.

        Devuelve:
            None: Actualiza el atributo humo; si usar_humo es False,
            no realiza cambios.

        Notas:
            Cada celda no ocupada por una pared se actualiza con
            H_nueva = H + D * (promedio_vecinos - H) - lambda * H,
            más el aporte de fuente cuando corresponde.
            Las cuatro concentraciones vecinas se leen del campo
            anterior; si un vecino es pared se usa el valor propio,
            representando ausencia de flujo. El resultado se limita
            al intervalo [0, 1]. No se modela un incendio físico real.
        """

        
        actual_humo = np.copy(self.humo)
        if self.configuracion.usar_humo:
            for y in range(self.alto):
                for x in range(self.ancho):
                    if not self.paredes[y, x]:
                        vecinos = []
                        for dy, dx in VECINOS:
                            ny, nx = y + dy, x + dx
                            if self.dentro(nx, ny) and not self.paredes[ny, nx]:
                                vecinos.append(actual_humo[ny, nx])
                            else:
                                vecinos.append(actual_humo[y, x])  # Sin flujo a través de paredes
                        promedio_vecinos = np.mean(vecinos)
                        H_nueva = actual_humo[y, x] + self.configuracion.difusion_humo * (promedio_vecinos - actual_humo[y, x]) - self.configuracion.decaimiento_humo * actual_humo[y, x]
                        if (x, y) in self.fuentes:
                            H_nueva += 1.0  # Aporte de fuente
                        self.humo[y, x] = np.clip(H_nueva, 0.0, 1.0)
        # TODO: Cuando usar_humo es False, no se modifica el entorno.
        # Usar una matriz nueva; todas las celdas leen el humo anterior.
        # Para cada celda que no sea pared:
        #   - Consultar sus cuatro vecinos (sin flujo a través de paredes).
        #   - Calcular difusión y decaimiento.
        # Incorporar el aporte de las cuatro fuentes.
        # Limitar los valores finales al intervalo [0, 1].

    def calcular_distancias(self):
        """Calcula mediante BFS las distancias mínimas a las salidas.

        Devuelve:
            numpy.ndarray: Matriz de enteros [y, x]. Cada valor
            no negativo indica los movimientos ortogonales mínimos
            hasta alguna salida. -1 significa celda bloqueada o
            sin camino transitable a ninguna salida.

        Notas:
            Se inicia una cola FIFO con todas las salidas a distancia
            cero. Se consideran paredes, foco y bloqueos por humo,
            pero no la ocupación de los agentes. El método no modifica
            el entorno y se ejecuta nuevamente en cada paso.
        """
        distancias = np.full((self.alto, self.ancho), -1, dtype=int)
        cola = deque()
        for x, y in self.salidas:
            distancias[y, x] = 0
            cola.append((x, y))
        while cola:
            x, y = cola.popleft()
            for dx, dy in VECINOS:
                vx, vy = x + dx, y + dy
                # Solo se expanden celdas transitables todavía no visitadas.
                if self.es_transitable(vx, vy) and distancias[vy, vx] == -1:
                    distancias[vy, vx] = distancias[y, x] + 1
                    cola.append((vx, vy))
        return distancias

