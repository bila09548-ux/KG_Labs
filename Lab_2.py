import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi
import math
import random
import sys

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)

def barycentric_coordinates (x0,y0,x1,y1,x2,y2,x,y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]

def drawing_triangle (x0,y0,z0,x1,y1,z1,x2,y2,z2):
    xmin = int(min(x0, x1, x2))
    xmax = int(max(x0, x1, x2))+ 1
    ymin = int(min(y0, y1, y2))
    ymax = int(max(y0, y1, y2))+ 1
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > 2000): xmax = 2000
    if (ymax > 2000): ymax = 2000

    color = (random.randrange(0,255), random.randrange(0,255), random.randrange(0,255))

    cos=cos_angle(n(x0, y0, x1, y1, x2, y2, z0, z1, z2))
    if (cos>=0):
        return
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            temporary = barycentric_coordinates(x0, y0, x1, y1, x2, y2, x, y)
            z = temporary[0] * z0 + temporary[1] * z1 + temporary[2] * z2
            if temporary[0] >= 0 and temporary[1] >= 0 and temporary[2] >= 0 and z<z_buffer[y, x]:
                img_mat[y][x] = (cos * -89, cos * 0, cos * -52)
                # img_mat[y][x] = color
                z_buffer[y, x] = z

# Вычисление векторного произведения
def n(x0, y0, x1, y1, x2, y2, z0, z1, z2):
    a = np.array([x1-x2, y1-y2, z1-z2])
    b = np.array([x1-x0, y1-y0, z1-z0])
    return np.cross(a, b)


def cos_angle(n):
    l = [0,0,1]
    nFirst = np.dot(n,l) # Скалярное произведение вектора
    nSecond = np.sqrt(np.dot(n, n)) # Вычисление длины вектора
    return nFirst / nSecond

z_buffer = np.zeros((2000,2000,1))
for i in range(2000):
    for j in range(2000):
        z_buffer[i][j] = sys.maxsize


file=open('model_1.obj')
v=[]
f=[]
for s in file:
    sp=s.split()
    if(sp[0]=='v'):
        v.append([float(sp[1]), float(sp[2]),float(sp[3])])
# print(v)
    if(sp[0]=='f'):
        f.append([int(sp[1].split('/')[0]), int(sp[2].split('/')[0]), int(sp[3].split('/')[0])])

for k in range (len(f)):
    x0=9000*v[f[k][0]-1][0]+1000
    y0=9000*v[f[k][0]-1][1]+1000
    x1=9000*v[f[k][1]-1][0]+1000
    y1=9000*v[f[k][1]-1][1]+1000
    x2=9000*v[f[k][2]-1][0]+1000
    y2=9000*v[f[k][2]-1][1]+1000
    z0 = 9000 * v[f[k][0] - 1][2] + 1000
    z1 = 9000 * v[f[k][1] - 1][2] + 1000
    z2 = 9000 * v[f[k][2] - 1][2] + 1000
    drawing_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)

img=Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')
img.show('img.png')
