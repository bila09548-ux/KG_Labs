import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi
import math

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)


def draw_line1(image,x0,y0,x1,y1, count, color):
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color


def draw_line2(image, x0, y0, x1, y1, count, color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y,x] = color

def draw_line3(image, x0, y0, x1, y1, count, color):
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line4(image, x0, y0, x1, y1, count, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line5(image, x0, y0, x1, y1, count, color):

    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def draw_line6(image, x0, y0, x1, y1, count, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0)/(x1-x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(int(x0), int(x1)):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update


def draw_line7(image, x0, y0, x1, y1, count, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2.0 * (x1-x0) * abs(y1 - y0)/(x1-x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(int(x0), int(x1)):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 2.0*(x1 - x0)*0.5):
            derror -= 2.0 * (x1 - x0)*1.0
            y += y_update

def draw_line8(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(int(x0), int(x1)):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update


for k in range(13):
    x0, y0 = 100, 100
    x1 = int(100 + 95 * cos(2 * pi / 13 * k))
    y1 = int(100 + 95 * sin(2 * pi / 13 * k))
#     # draw_line1(img_mat, x0, y0, x1, y1, 100, (255, 0, 255))
#     # draw_line2(img_mat, x0, y0, x1, y1, 100, (255, 0, 255))
#     draw_line3(img_mat, x0, y0, x1, y1, 100, (255, 0, 255))
#     # draw_line4(img_mat, x0, y0, x1, y1, 100, (255, 0, 255))
#     # draw_line5(img_mat, x0, y0, x1, y1, 100, (255, 0, 255))
#     #draw_line6(img_mat, x0, y0, x1, y1, 100, (255, 0, 255))
    #draw_line7(img_mat, x0, y0, x1, y1, 100, (255, 0, 255))
#     draw_line8(img_mat, x0, y0, x1, y1, (255, 0, 255))
#
# img = Image.fromarray(img_mat, mode='RGB')
# img.save('img.png')
# img.show('img.png')

file=open('model_1.obj')
v=[]
f=[]
for s in file:
    sp=s.split()
    if(sp[0]=='v'):
        v.append([float(sp[1]), float(sp[2]),float(sp[3])])
    #print(v)
    if(sp[0]=='f'):
        f.append([int(sp[1].split('/')[0]), int(sp[2].split('/')[0]), int(sp[3].split('/')[0])])
for k in range (len(f)):
    x0=int(9000*v[f[k][0]-1][0]+1000)
    y0=int(9000*v[f[k][0]-1][1]+1000)
    x1=int(9000*v[f[k][1]-1][0]+1000)
    y1=int(9000*v[f[k][1]-1][1]+1000)
    x2=int(9000*v[f[k][2]-1][0]+1000)
    y2=int(9000*v[f[k][2]-1][1]+1000)
    draw_line8(img_mat, x0, y0, x1, y1, (255, 0, 255))
    draw_line8(img_mat, x0, y0, x2, y2, (255, 0, 255))
    draw_line8(img_mat, x2, y2, x1, y1, (255, 0, 255))

img=Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')
img.show('img.png')
