"""
homtransf.py — Transformaciones homogéneas para robótica 3D
------------------------------------------------------------

Este módulo implementa funciones básicas de álgebra matricial utilizadas en robótica
y cinemática para representar posiciones y orientaciones de un cuerpo rígido 
mediante **matrices de transformación homogénea (4x4)**.

Autor: Dr. Juan Manuel Ahuactzin Larios
Versión: 1.0
Licencia: GPL
Correo: juan_manuel_ahuactzin@msn.com
Año: 2023

------------------------------------------------------------
Funciones principales:
- Point(x, y, z): Crea un punto en coordenadas homogéneas.
- Trans(p): Devuelve la matriz de traslación correspondiente al vector p.
- Rotx(alpha), Roty(phi), Rotz(theta): Rotaciones sobre los ejes X, Y y Z.
- Invert(T): Calcula la matriz inversa de una transformación homogénea T.
------------------------------------------------------------
"""

__author__ = "Juan Manuel Ahuactzin Larios"
__copyright__ = "Copyright 2023, Juan Manuel Ahuactzin Larios"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "juan_manuel_ahuactzin@msn.com"
__status__ = "Production"

import numpy as np
import math

# ===========================================================
# Creación de puntos y transformaciones homogéneas
# ===========================================================

def Point(x: float, y: float, z: float) -> np.ndarray:
    """
    Crea un punto 3D en coordenadas homogéneas.

    Args:
        x (float): Coordenada en X.
        y (float): Coordenada en Y.
        z (float): Coordenada en Z.

    Returns:
        np.ndarray: Vector columna de 4x1 representando el punto [x, y, z, 1]^T.
    """
    return np.array([[x], [y], [z], [1.0]])


def Trans(p: np.ndarray) -> np.ndarray:
    """
    Crea una matriz de traslación homogénea 4x4 a partir de un vector de posición.

    Args:
        p (np.ndarray): Vector columna [px, py, pz, 1]^T o [px, py, pz]^T.

    Returns:
        np.ndarray: Matriz de traslación homogénea.
    """
    px, py, pz = p[0][0], p[1][0], p[2][0]
    mat = np.array([
        [1.0, 0.0, 0.0, px],
        [0.0, 1.0, 0.0, py],
        [0.0, 0.0, 1.0, pz],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return mat


# ===========================================================
# Rotaciones homogéneas alrededor de los ejes cartesianos
# ===========================================================

def Rotx(alpha: float) -> np.ndarray:
    """
    Genera una matriz de rotación homogénea alrededor del eje X.

    Args:
        alpha (float): Ángulo de rotación en radianes.

    Returns:
        np.ndarray: Matriz 4x4 de rotación sobre X.
    """
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, math.cos(alpha), -math.sin(alpha), 0.0],
        [0.0, math.sin(alpha),  math.cos(alpha), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def Roty(phi: float) -> np.ndarray:
    """
    Genera una matriz de rotación homogénea alrededor del eje Y.

    Args:
        phi (float): Ángulo de rotación en radianes.

    Returns:
        np.ndarray: Matriz 4x4 de rotación sobre Y.
    """
    return np.array([
        [math.cos(phi),  0.0, math.sin(phi), 0.0],
        [0.0,            1.0, 0.0,          0.0],
        [-math.sin(phi), 0.0, math.cos(phi), 0.0],
        [0.0,            0.0, 0.0,          1.0]
    ])


def Rotz(theta: float) -> np.ndarray:
    """
    Genera una matriz de rotación homogénea alrededor del eje Z.

    Args:
        theta (float): Ángulo de rotación en radianes.

    Returns:
        np.ndarray: Matriz 4x4 de rotación sobre Z.
    """
    return np.array([
        [math.cos(theta), -math.sin(theta), 0.0, 0.0],
        [math.sin(theta),  math.cos(theta), 0.0, 0.0],
        [0.0,              0.0,             1.0, 0.0],
        [0.0,              0.0,             0.0, 1.0]
    ])


# ===========================================================
# Inversión de matrices homogéneas
# ===========================================================

def Invert(T: np.ndarray) -> np.ndarray:
    """
    Calcula la inversa de una matriz de transformación homogénea 4x4.

    Args:
        T (np.ndarray): Matriz de transformación homogénea 4x4.

    Returns:
        np.ndarray: Matriz inversa de T, que invierte tanto la rotación
        como la traslación.
    """
    # Extraer ejes del marco original
    n_t = np.array([T[0][0], T[1][0], T[2][0]])  # eje x
    o_t = np.array([T[0][1], T[1][1], T[2][1]])  # eje y
    a_t = np.array([T[0][2], T[1][2], T[2][2]])  # eje z
    p = np.array([T[0][3], T[1][3], T[2][3]])    # vector de traslación

    # Calcular inversa usando propiedades de transformaciones homogéneas
    T_inv = np.array([
        [n_t[0], n_t[1], n_t[2], -np.dot(n_t, p)],
        [o_t[0], o_t[1], o_t[2], -np.dot(o_t, p)],
        [a_t[0], a_t[1], a_t[2], -np.dot(a_t, p)],
        [0.0,    0.0,    0.0,    1.0]
    ])

    return T_inv
