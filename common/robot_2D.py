"""
planar_robot.py — Robot planar 2R con cinemática directa e inversa
-------------------------------------------------------------------

Este módulo implementa un modelo geométrico de un **robot planar de dos
grados de libertad (2R)** utilizando transformaciones homogéneas 3D.

El robot está compuesto por:
- Dos eslabones rectangulares (Link)
- Un espacio de trabajo cuadrado (WorkSpace)
- Transformaciones homogéneas para describir la configuración

Se incluyen métodos para:
- Configurar la postura del robot
- Calcular la cinemática directa
- Resolver la cinemática inversa (elbow up / elbow down)
- Dibujar el robot en el plano

Autor: Dr. Juan Manuel Ahuactzin Larios
Versión: 1.0
Licencia: GPL
Correo: juan_manuel_ahuactzin@msn.com
Año: 2023

-------------------------------------------------------------------
Clases principales:
- rectangle: Clase base para figuras rectangulares.
- Link: Representa un eslabón del robot.
- WorkSpace: Representa el área de trabajo del robot.
- PlanarRobot: Implementa el modelo completo del robot 2R.

Métodos relevantes en PlanarRobot:
- set_conf(theta): Define la configuración articular.
- direct_kinematics(): Calcula la posición del efector final.
- inverse_kinematics(px, py, elbow_type): Calcula las soluciones articulares.
- draw(ax=None): Dibuja el robot en un eje Matplotlib opcional.
-------------------------------------------------------------------
"""

__author__ = "Juan Manuel Ahuactzin Larios"
__copyright__ = "Copyright 2026, Juan Manuel Ahuactzin Larios"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "juan_manuel_ahuactzin@msn.com"
__status__ = "Production"




import matplotlib.pyplot as plt
from math import sin, cos, atan2, sqrt, pi
import sys
import copy

from .homtransf import *

import copy
import matplotlib.pyplot as plt
import numpy as np

class rectangle:
    def __init__(self, l, w):
        self.edges = []
        self.edges.append(Point(l, w, 0))
        self.edges.append(Point(-l, w, 0))
        self.edges.append(Point(-l, -w, 0))
        self.edges.append(Point(l, -w, 0))
        
    def draw(self, ax=None):
        # ax opcional: si no te pasan, usa el eje actual
        if ax is None:
            ax = plt.gca()

        for i in range(0, len(self.edges)):
            x = [float(self.edges[i][0][0]), float(self.edges[i-1][0][0])]
            y = [float(self.edges[i][1][0]), float(self.edges[i-1][1][0])]
            ax.plot(x, y, color='blue')
            
    def transform(self, T):
        new_points = []
        for p in self.edges:
            tranf_p = T @ p
            new_points.append(tranf_p)
        
        new_link = copy.copy(self)
        new_link.edges = new_points
        
        return new_link    

class Link(rectangle):
    def __init__(self, l, w):
        self.edges = []
        self.edges.append(Point(w/2.0, w/2.0, 0))
        self.edges.append(Point(-l+w/2.0, w/2.0, 0))
        self.edges.append(Point(-l+w/2.0, -w/2.0, 0))
        self.edges.append(Point(w/2.0, -w/2.0, 0))
        
class WorkSpace(rectangle):
    def __init__(self, l):
        super().__init__(l, l)
        
class PlanarRobot:
    def __init__(self, l, w):
        
        self.l = l
        self.w = w
        
        self.workspace = WorkSpace(2*l)
        self.art = Link(l, w)

        self.theta = [0.0, 0.0]
        
        self.T1_w = None
        self.T2_1 = None
        self.T2_w = None
        
        self.set_conf(self.theta)
        
    def T_i(self, i):
        return Rotz(self.theta[i]) @ Trans(Point(self.l-self.w, 0, 0))
    
    def set_conf(self, theta):
        self.theta = theta
        
        self.T1_w = self.T_i(0)
        self.T2_1 = self.T_i(1)
        
        self.T2_w = self.T1_w @ self.T2_1
        
    def draw(self, ax=None):
        # ax opcional (para subplots)
        if ax is None:
            ax = plt.gca()

        art1_w = self.art.transform(self.T1_w)
        art2_w = self.art.transform(self.T2_w)

        ax.set_aspect('equal', adjustable='box')

        # dibujar en el mismo eje
        self.workspace.draw(ax=ax)
        art1_w.draw(ax=ax)
        art2_w.draw(ax=ax)

        x1 = float(self.T1_w[0][3])
        y1 = float(self.T1_w[1][3])
        x2, y2 = self.direct_kinematics()

        x = [0.0, x1, float(x2)]
        y = [0.0, y1, float(y2)]

        ax.scatter(x, y, marker="o", color='red')
        
    def direct_kinematics(self):
        # asegura escalares
        x = float(self.T2_w[0][3])
        y = float(self.T2_w[1][3])
        return x, y
    
    def inverse_kinematics(self, px, py, elbow_type):
        if (elbow_type == "elbow down"):
            sign = 1
        elif (elbow_type == "elbow up"):
            sign = -1
        else:
            return [] # Invalid option

        cos_q2 = (px**2 + py**2 - 2*(self.l-self.w)**2)/(2 * (self.l-self.w)**2)
        
        if (cos_q2 > 1):
            return []  # No solution

        q2 = atan2(sign*sqrt(1-cos_q2**2), cos_q2)
        betha = atan2(py, px)
        alpha = atan2(self.l*sin(q2), self.l+self.l*cos_q2)
        q1 = betha - alpha
        
        return [q1, q2]

        