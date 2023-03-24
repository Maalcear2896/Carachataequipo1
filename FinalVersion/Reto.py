import skimage
import matplotlib.pyplot as plt
import numpy
from PIL import Image
from scipy import ndimage

# Abrir imagen
Is = Image.open('Sample.png');
Iz = skimage.io.imread("Sample.png")
Izoom = Iz[0:100,150:250]
I = Is.convert('L');
I = numpy.asarray(I);
Ic = I[0:100,150:250]

I = I / 255.0;


#Kernels
enfoque = numpy.array([[0,0,0,0,0],[0,0,-1,0,0],[0,-1,5,-1,0],[0,0,-1,0,0],[0,0,0,0,0]])
desenfoque = numpy.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]])
rbordes = numpy.array([[0,0,0,0,0],[0,0,0,0,0],[0,-1,1,-0,0],[0,0,0,0,0],[0,0,0,0,0]])
dbordes = numpy.array([[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],[0,0,1,0,0],[0,0,0,0,0]])
repujado = numpy.array([[0,0,0,0,0],[0,-2,-1,0,0],[0,-1,1,1,0],[0,0,1,2,0],[0,0,0,0,0]])


J0 = ndimage.convolve(Ic, enfoque, mode='constant', cval=0.0)
J1 = ndimage.convolve(Ic, desenfoque, mode='constant', cval=0.0)
J2 = ndimage.convolve(Ic, rbordes, mode='constant', cval=0.0)
J3 = ndimage.convolve(Ic, dbordes, mode='constant', cval=0.0)
J4 = ndimage.convolve(Ic, repujado, mode='constant', cval=0.0)


plt.figure(figsize = (10,10))

plt.subplot(2,4,1)
plt.imshow(Is)
plt.xlabel('Imagen Normal')

plt.subplot(2,4,2)
plt.imshow(Izoom)
plt.xlabel('Imagen Zoom Normal')

plt.subplot(2,4,3)
plt.imshow(Ic)
plt.xlabel('Input Zoom Filtro')

plt.subplot(2,4,4)
plt.imshow(J0)
plt.xlabel('Enfoque')

plt.subplot(2,4,5)
plt.imshow(J1)
plt.xlabel('Desenfoque')

plt.subplot(2,4,6)
plt.imshow(J2)
plt.xlabel('Realzar Bordes')

plt.subplot(2,4,7)
plt.imshow(J3)
plt.xlabel('Detectar Bordes')

plt.subplot(2,4,8)
plt.imshow(J4)
plt.xlabel('Repujado')


plt.grid(False)
plt.show()
