from typing import Counter
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
import scipy.io as sc
import ctypes  # An included library with Python install.
import os
import imageio
import nibabel as nib

def posXY(entrada):
        xpos = []
        ypos = []
        final = len(entrada)
        #print(final)
        #print(entrada)
        for i in range(0,final):
            xpos.append(entrada[i][0])
            #print(xpos)
            ypos.append(entrada[i][1])
        return xpos, ypos

def buscarCarpeta():
    root = Tk()
    root.withdraw() 
    root.update()
    filename = askopenfilename()
    root.quit()     # stops mainloop
    root.destroy()
    return filename

def guardarCarpeta():
    root = Tk()
    root.withdraw() 
    root.update()
    filename = asksaveasfilename()
    root.quit()     # stops mainloop
    root.destroy()
    return filename

def indices(entrada):
    if entrada == 512:
        indice = 0
    elif entrada == 256:
        indice = 1
    elif entrada == 128:
        indice = 2
    elif entrada == 64:
        indice = 3
    elif entrada == 32:
        indice = 4
    elif entrada == 16:
        indice = 5
    elif entrada == 8:
        indice = 6
    elif entrada == 4:
        indice = 7
    elif entrada == 2:
        indice = 8
    #print("El indice es ", indice)
    return indice

def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def auto_canny(image, sigma=0.66):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def preprocesamientoff(image):
    gray_img=cv2.bitwise_not(image)
    img1_gray = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
    imageedge = auto_canny(img1_gray)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(imageedge,kernel,iterations = 1)
    # Copy the thresholded image.
    im_th = dilation
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (156,0), 255)
    # # Invert floodfilled image
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # im_out = im_th | im_floodfill_inv
    # diamondkernel = np.array([[0, 1, 0], [1, 1, 1],[0, 1, 0]],np.uint8)
    # erode1 = cv2.erode(im_out,diamondkernel,iterations = 1)#Iclear si si pone lo de
    # return erode1
    return im_floodfill

def preprocesamiento(image):
    img1_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imageedge = auto_canny(img1_gray)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(imageedge,kernel,iterations = 1)
    # Copy the thresholded image.
    im_th = dilation
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (156,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    diamondkernel = np.array([[0, 1, 0], [1, 1, 1],[0, 1, 0]],np.uint8)
    erode1 = cv2.erode(im_out,diamondkernel,iterations = 1)#Iclear si si pone lo de
    return erode1

def counting(image):
    rows,cols = image.shape
    refinamientos = [2,4,8,16,32]
    almacen = []
    boxpointalm = []
    boxpoint = []
    box = 2
    aux = False
    print(rows,cols)
    for box in refinamientos:
        M = rows//box
        N = cols//box
        puntos = 0
        for y in range(0,rows,M):
            for x in range(0,cols, N):
                tiles = image[y:y+M,x:x+N]
                # cv2.imshow('image',tiles)
                # cv2.waitKey(0) 
                # plt.imshow(tiles, cmap='gray')
                r,c = tiles.shape
                for i in range(r):
                    for j in range(1,c):
                        if tiles[i][j] != tiles[i][j-1]:
                            puntos = puntos +1
                            aux = True
                            break
                    if aux ==  True:
                        boxpoint.append([(x+x+N)/20.48,(y+y+M)/20.48]) 
                        aux =False
                        break
        if boxpoint != []:
            boxpointalm.append(boxpoint)
        almacen.append(puntos)
        boxpoint = []
        puntos = 0
    # print(almacen, boxpointalm)
    return almacen, boxpointalm

def dibujarContorno(image,erode1):
    contours,_ = cv2.findContours(erode1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    final = image.copy()
    # Iterate over all contours
    for i,c in enumerate(contours):
        # Find mean colour inside this contour by doing a masked mean
        mask = np.zeros(erode1.shape, np.uint8)
        cv2.drawContours(mask,[c],-1,255, -1)
        # DEBUG: cv2.imwrite(f"mask-{i}.png",mask)
        mean,_,_,_ = cv2.mean(erode1, mask=mask)
        # DEBUG: print(f"i: {i}, mean: {mean}")

        # # Get appropriate colour for this label
        # label = 2 if mean > 1.0 else 1
        # colour = RGBforLabel.get(label)
        # # DEBUG: print(f"Colour: {colour}")

        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(final,[c],-1,(0,255,0),1)
    return final

def read_niifile(niifile):           #读取niifile文件
    img = nib.load(niifile)          #下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()      #获取niifile数据
    return img_fdata

def save_fig(file, savepicdir):                  #保存为图片
    fdata = read_niifile(file)       #调用上面的函数，获得数据
    (x,y,z) = fdata.shape            #获得数据shape信息：（长，宽，维度-切片数量）
    for k in range(z):
        silce = fdata[k,:,:]         #三个位置表示三个不同角度的切片
        imageio.imwrite(os.path.join(savepicdir,'{}.png'.format(k)),silce)

class Index(object):
    im = np.zeros( (100, 100), np.uint8 );
    im_resized = 0
    alto = 0
    ancho = 0
    mat = 0
    numgrid = 512
    x = [0,0,0,0,0,0,0,0,0]
    y = [0,0,0,0,0,0,0,0,0]
    vector = []
    todosv1 = []
    todos =[0,0,0,0,0,0,0,0,0]
    j=0
    fillTable = [[512,0],
                [256,0],
                [128,0],
                [64,0],
                [32,0],
                [16,0],
                [8,0],
                [4,0],
                [2,0]]
    preset = [1/512,1/256,1/128,1/64,1/32,1/16,1/8,1/4,1/2]

    def gridNumsobre2(self, secundaria):
        #scatter.remove()
        if self.numgrid == 32:
            self.numgrid = 512
            legend.set_text("Cuadricula = %.0f" %(self.numgrid))
        else:
            self.numgrid = int(self.numgrid/2)
            legend.set_text("Cuadricula = %.0f" %(self.numgrid))
        #plt.show()
        #---------------------------------------------------
        scatter.set_visible(False)
        dimensiones =self.im.shape
        alto = dimensiones[0]
        ancho = dimensiones [1]
        escAlt =  1024 / alto
        escAnc = 1024 / ancho
        #print (ancho, alto, type(ancho))
        self.im_resized = cv2.resize(self.im, None,fx=escAnc, fy=escAlt, interpolation=cv2.INTER_LINEAR)
        #print (im_resized.shape)
        for i in range(0,self.im_resized.shape[0],self.numgrid):
                cv2.line(self.im_resized, (i, 0), (i, self.im_resized.shape[0]), color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=1)

        for i in range(0,self.im_resized.shape[1],self.numgrid):
                cv2.line(self.im_resized, (0, i), (self.im_resized.shape[1], i), color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=1)
        myobj.set_data(cv2.cvtColor(self.im_resized, cv2.COLOR_BGR2RGB))
        # Grafico de puntos ya obtenidos
        temp1 = self.todosv1[indices(self.numgrid)]
        scatter.set_offsets(temp1)
        scatter.set_visible(True)

        plt.show()

    def gridNumpor2(self, secundaria):
        #scatter.remove()
        if self.numgrid == 512:
            self.numgrid = 32
            legend.set_text("Cuadricula = %.0f" %(self.numgrid))
        else:
            self.numgrid = int(self.numgrid*2)
            legend.set_text("Cuadricula = %.0f" %(self.numgrid))
        #plt.show()
        #-----------------------------------------------
        scatter.set_visible(False)
        dimensiones =self.im.shape
        alto = dimensiones[0]
        ancho = dimensiones [1]
        escAlt =  1024 / alto
        escAnc = 1024 / ancho
        #print (ancho, alto, type(ancho))
        self.im_resized = cv2.resize(self.im, None,fx=escAnc, fy=escAlt, interpolation=cv2.INTER_LINEAR)
        #print (im_resized.shape)
        for i in range(0,self.im_resized.shape[0],self.numgrid):
                cv2.line(self.im_resized, (i, 0), (i, self.im_resized.shape[0]), color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=1)

        for i in range(0,self.im_resized.shape[1],self.numgrid):
                cv2.line(self.im_resized, (0, i), (self.im_resized.shape[1], i), color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=1)
        myobj.set_data(cv2.cvtColor(self.im_resized, cv2.COLOR_BGR2RGB))
        # Grafico de puntos ya obtenidos
        temp1 = self.todosv1[indices(self.numgrid)]
        scatter.set_offsets(temp1)
        scatter.set_visible(True)

        plt.show()

        
    
    def Puntos(self,secundaria):
        
        #try:
        puntosI = plt.ginput(-1,0,True)

        #cantidad de clics
        numclic = len(puntosI)
        xpos, ypos = posXY(puntosI)
        # Proceso para almacenar puntos cuando ya se han tomado algunos
        if self.todos[indices(self.numgrid)] != 0:
            Findex = indices(self.numgrid)
            self.x[Findex] = 1/self.numgrid
            self.todos[Findex][1]+=xpos
            self.todos[Findex][2]+=ypos
            #print("este es x puntos",self.x)
            self.y[indices(self.numgrid)] += numclic
            # self.vector = [numclic, xpos, ypos, self.numgrid]
            self.todos[indices(self.numgrid)][0] += numclic
        # Almacenamiento de los datos por primera vez
        else:
            self.x[indices(self.numgrid)] = 1/self.numgrid
            #print("este es x puntos",self.x)
            self.y[indices(self.numgrid)] = numclic
            self.vector = [numclic, xpos, ypos, self.numgrid,self.im]
            self.todos[indices(self.numgrid)] = self.vector
            #print(self.todos)
        #print(self.todos)

        if self.x[indices(self.numgrid)] != 0:
            concatnuevo= []
            temp1 = self.todos[indices(self.numgrid)][1]
            temp2 = self.todos[indices(self.numgrid)][2]
            
            for i in range(0,int(self.todos[indices(self.numgrid)][0])):
                nuevo = [temp1[i],temp2[i]]
                concatnuevo.append(nuevo)
            scatter.set_offsets(concatnuevo)
            scatter.set_visible(True)
        #Llenado de tabla al lado derecho
        self.fillTable[indices(self.numgrid)][1]= self.todos[indices(self.numgrid)][0]
        tabla.get_celld()[(indices(self.numgrid), 1)].get_text().set_text(self.todos[indices(self.numgrid)][0])
        # self.j=self.j+1
        plt.show()
        #print(puntosI)
        # print("Matriz-------------------")
        # print(vector)
        # except:
        #         print("No se ejecuto el ginput")
        #         x = []

    def Quitar(self,second):
        QuitarPuntos = plt.ginput(1,0,True)
        Findex = indices(self.numgrid) 
        qpx, qpy = posXY(QuitarPuntos)
        procesx = np.absolute(np.array(self.todos[Findex][1])- qpx)
        procesy = np.absolute(np.array(self.todos[Findex][2])- qpy)
        # print("Puntos almacenados en x= ",self.todos[indices(self.numgrid)][1])
        # print("Puntos almacenados en y= ",self.todos[indices(self.numgrid)][2])
        # print("Puntos tomados para remover = ",qpx,qpy)
        # print("Almacenados menos tomados en x= ", procesx)
        # print("Almacenados menos tomados en y= ", procesy)
        convx = procesx.tolist().index(min(procesx.tolist()))
        convy = procesy.tolist().index(min(procesy.tolist()))
        # print("Indice del menor x",convx)
        # print("Indice del menor y",convy)

        if convx == convy:
            self.todos[Findex][1].pop(convx)
            self.todos[Findex][2].pop(convy)
            self.y[indices(self.numgrid)] = len(self.todos[Findex][1])
            self.todos[indices(self.numgrid)][0] = len(self.todos[Findex][1])

            if self.x[indices(self.numgrid)] != 0:
                concatnuevo= []
                temp1 = self.todos[indices(self.numgrid)][1]
                temp2 = self.todos[indices(self.numgrid)][2]
            
                for i in range(0,int(self.todos[indices(self.numgrid)][0])):
                    nuevo = [temp1[i],temp2[i]]
                    concatnuevo.append(nuevo)
                scatter.set_offsets(concatnuevo)
                scatter.set_visible(True)
            #Llenado de tabla al lado derecho
            self.fillTable[indices(self.numgrid)][1]= self.todos[indices(self.numgrid)][0]
            tabla.get_celld()[(indices(self.numgrid), 1)].get_text().set_text(self.todos[indices(self.numgrid)][0])
            plt.show()
        
        # print("Asi queda x = ",self.todos[indices(self.numgrid)][1])
        # print("Asi queda y = ",self.todos[indices(self.numgrid)][2])

        
    def calculoAutomatico(self, second):
        scatter.set_visible(True)
        self.im_resized = self.im
        # erode1 = preprocesamientoff(self.im_resized)
        # erode1 = preprocesamiento(erode1)
        erode1 = preprocesamiento(self.im_resized)
        final = dibujarContorno(self.im_resized,erode1)

        myobj.set_data(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

        reg = [1/512,1/256,1/128,1/64,1/32]

        count, boxpointscatter = counting(erode1)
        self.todosv1 = boxpointscatter
        z = np.polyfit(np.log(reg),np.log(count),1)
        print("y=%.6fx+(%.2f)"%(z[0],z[1]))
        # r = np.corrcoef(np.log(self.x[0:indices(self.numgrid)+1]),np.log(self.y[0:indices(self.numgrid)+1]))[0,1]
        r = np.corrcoef(np.log(reg),np.log(count))[0,1]
        print(r)
        legend2.set_text("y=%.6fx+(%.2f)"%(z[0],z[1]))
        legend4.set_text("R^2=%.3f"%r)

        for ints in range(len(reg)):
            self.fillTable[ints][1]= count[ints]
            tabla.get_celld()[(ints, 1)].get_text().set_text(count[ints])
        
        # Grafico de puntos ya obtenidos
        temp1 = self.todosv1[indices(self.numgrid)]
        print(temp1)
        scatter.set_offsets(temp1)
        scatter.set_visible(True)
        # self.j=self.j+1
        plt.show()


    def calculo(self, second):
        z=[]
        r=0
        print("este es x",self.x)
       
        for i in range(0,len(self.x)):
            if self.x[i] > 0:
                indcal = i+1
        print(indcal)
        # z = np.polyfit(np.log(self.x[0:indices(self.numgrid)+1]),np.log(self.y[0:indices(self.numgrid)+1]),1)
        z = np.polyfit(np.log(self.x[0:indcal]),np.log(self.y[0:indcal]),1)
        print("y=%.6fx+(%.2f)"%(z[0],z[1]))
        # r = np.corrcoef(np.log(self.x[0:indices(self.numgrid)+1]),np.log(self.y[0:indices(self.numgrid)+1]))[0,1]
        r = np.corrcoef(np.log(self.x[0:indcal]),np.log(self.y[0:indcal]))[0,1]
        print(r)
        legend2.set_text("y=%.6fx+(%.2f)"%(z[0],z[1]))
        legend4.set_text("R^2=%.3f"%r)
        # sc.savemat('Datos.mat',{'datos':self.todos})
        # sc.savemat('Points.mat',{'npoints':self.y})
        
    def cargar3d(self, second):
        dsize = (1024, 1024)
        savepicdir = 'conversion1/sujeto'  
        try:                    #保存png的路径
            paths = buscarCarpeta()# filename#r'D:\MAESTRIA\AVANCES_TESIS\MICHAEL_INTERFAZ\curva-de-koch.jpg'
            rutasplit= paths.split('/')
            palabra = rutasplit[len(rutasplit)-1].split(".")
            palabra = palabra[0]
            savepicdir = 'conversion1/' + palabra 
            if not os.path.isdir(savepicdir):
                os.mkdir(savepicdir)                 #创建文件夹e
            save_fig(paths,savepicdir)

            acumuladoz0 = []
            acumuladoz1 = []
            acumulador = []
            listaimagenes = os.listdir('conversion1/' + palabra)
            for i in listaimagenes:
                scatter.set_visible(False)
                self.im = cv2.resize(cv2.imread('conversion1/'+palabra+'/'+i,1),dsize)
                self.im_resized = self.im

                #Mismo codigo de cargaAutomatica
                erode1 = preprocesamientoff(self.im_resized)
                erode1 = preprocesamiento(self.im_resized)
                final = dibujarContorno(self.im_resized,erode1)

                myobj.set_data(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

                reg = [1/512,1/256,1/128,1/64,1/32]

                count, boxpointscatter = counting(erode1)
                self.todosv1 = boxpointscatter
                z = np.polyfit(np.log(reg),np.log(count),1)
                print("y=%.6fx+(%.2f)"%(z[0],z[1]))
                # r = np.corrcoef(np.log(self.x[0:indices(self.numgrid)+1]),np.log(self.y[0:indices(self.numgrid)+1]))[0,1]
                r = np.corrcoef(np.log(reg),np.log(count))[0,1]
                print(r)
                legend2.set_text("y=%.6fx+(%.2f)"%(z[0],z[1]))
                legend4.set_text("R^2=%.3f"%r)

                for ints in range(len(reg)):
                    self.fillTable[ints][1]= count[ints]
                    tabla.get_celld()[(ints, 1)].get_text().set_text(count[ints])
                
                # Grafico de puntos ya obtenidos
                temp1 = self.todosv1[indices(self.numgrid)]
                print(temp1)
                scatter.set_offsets(temp1)
                scatter.set_visible(True)
                # self.j=self.j+1
                acumuladoz0.append(z[0])
                acumuladoz1.append(z[1])
                acumulador.append(r)
                # plt.show()
            legend2.set_text("y=%.6fx+(%.2f)"%(np.mean(acumuladoz0),np.mean(acumuladoz0)))
            legend4.set_text("R^2=%.3f"%np.mean(acumulador))
            plt.show()
        # except FileExistsError:
        #     print("carpeta ya creada, revise el navegador")
        except ValueError:
            print("Ese archivo nii es diferente, seleccione uno correcto")
            Mbox('Error', 'Ese archivo nii es diferente, seleccione uno correcto', 1)
        except:
            print("Error en la carga de archivo")
            Mbox('Error', 'Archivo incorrecto o no seleccionado', 1)
            print("Error en la carga de archivoss")

        


    def cargarImagen(self,second):
        # root = Tk()
        # root.withdraw() 
        # root.update()
        # filename = askopenfilename()
        # root.quit()     # stops mainloop
        # root.destroy()
        dsize = (1024, 1024)
        try:
            paths = buscarCarpeta()# filename#r'D:\MAESTRIA\AVANCES_TESIS\MICHAEL_INTERFAZ\curva-de-koch.jpg'
            self.im = cv2.resize(cv2.imread(paths,1),dsize)

            #print(type(self.im))
            myobj.set_data(cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB))
            #plt.imshow(im_resized)
            #plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
            plt.show()
        except:
            print("Error en la carga de archivo")
            Mbox('Error', 'Archivo incorrecto o no seleccionado', 1)
            print("Error en la carga de archivoss")
        
    
    def cuadricula(self,second):
        scatter.set_visible(False)
        dimensiones =self.im.shape
        alto = dimensiones[0]
        ancho = dimensiones [1]
        escAlt =  1024 / alto
        escAnc = 1024 / ancho
        #print (ancho, alto, type(ancho))
        self.im_resized = cv2.resize(self.im, None,fx=escAnc, fy=escAlt, interpolation=cv2.INTER_LINEAR)
        #print (im_resized.shape)
        for i in range(0,self.im_resized.shape[0],self.numgrid):
                cv2.line(self.im_resized, (i, 0), (i, self.im_resized.shape[0]), color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=1)

        for i in range(0,self.im_resized.shape[1],self.numgrid):
                cv2.line(self.im_resized, (0, i), (self.im_resized.shape[1], i), color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=1)
        myobj.set_data(cv2.cvtColor(self.im_resized, cv2.COLOR_BGR2RGB))
        # Grafico de puntos ya obtenidos
        if self.x[indices(self.numgrid)] != 0:
            concatnuevo= []
            temp1 = self.todos[indices(self.numgrid)][1]
            temp2 = self.todos[indices(self.numgrid)][2]
            
            for i in range(0,int(self.todos[indices(self.numgrid)][0])):
                nuevo = [temp1[i],temp2[i]]
                concatnuevo.append(nuevo)
            scatter.set_offsets(concatnuevo)
            scatter.set_visible(True)

        plt.show()


        
    def guardar(self,second):
        # arr1 = np.array(self.todos)
        # arr2 = np.array(self.y)
        # guardar1 = np.vstack(arr1.ravel()).T
        # guardar2 = np.vstack(arr2.ravel()).T
        mdict = guardarCarpeta()
        sc.savemat(os.path.join(mdict),{'datos':self.todos})
        # sc.savemat('Points.mat',{'npoints':self.y})

    def cargarmat(self,second):
        # i = 0
        aux = 0
        arraytt = np.zeros( (100, 100), np.uint8 )
        # cargas = []
        # cargas1 = []
        direccion = buscarCarpeta()
        unit33 =sc.loadmat(direccion)#'matlab.mat')
        for key in unit33 :
            #print( key, ":",unit33[key])
            ultima = key
        array = unit33[ultima]

        for i in range(0,int(len(array[0]))):
            try :
                numpoint = array[0,i][0,0][0,0]
                grid = int(array[0,i][0,3][0,0])
                self.numgrid = grid
                corx= array[0,i][0,1]
                cory = array[0,i][0,2]
                try : 
                    img = im = array[0,0][0,4]
                except:
                    img = arraytt
                    print("No tiene imagen")
                if numpoint != 0:
                    tabla.get_celld()[(i, 1)].get_text().set_text(numpoint)
                    self.x[indices(self.numgrid)] = 1/grid
                    self.y[indices(self.numgrid)] = int(numpoint)
                    # cargas.append(int(numpoint))
                    # cargas1.append(grid)
                    aux = aux + 1
                    self.vector = [int(numpoint),corx.tolist()[0],cory.tolist()[0],grid,img]
                    # print("este es vector .........",self.vector)
                    print("")
                    self.todos[indices(self.numgrid)] = self.vector
                    # print(self.todos)
                    # self.todos[1] = corx.tolist()[0]
                    # self.todos[2] = cory.tolist()[0]
                    # self.todos[3] = grid
            except:
                print("solo tiene ", aux,"simbolos")
        
        #print("imprime todo el vector",self.todos)
        # print(grid,cargas1)
        # self.x =self.preset[0:aux]
        # print ("nuevo x", self.x)
        # self.y = cargas
        # print ("nuevo y", self.y)
        self.im = img
        myobj.set_data(cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB))
        legend.set_text("Cuadricula = %.0f" %(self.numgrid))
        plt.show()
        
    def help(self,second):
        os.startfile('INSTRUCTIVO.pdf')
        # Mbox('Help', 'Your text.....dfasfas\n aslkdjflaksfjsalf\n sjfklasfjlaskf\n asljhfaskjfhskjafhf', 1)
        


def save(text):
        ydata = eval(text)
        print(str(ydata))
        sc.savemat('Datos.mat',{'datos':callback.todos})
        # sc.savemat('Points.mat',{'npoints':callback.y})


        


datos= [[512,0],
        [256,0],
        [128,0],
        [64,0],
        [32,0],
        [16,0],
        [8,0],
        [4,0],
        [2,0]]
textstr = "Cuadricula = 512"
titulodd= "Fractal Dimension Box Counting Method"
textstr2 = "Y = m * x + b"
textstr3 = "Resultados"
textstr4 = "R^2 = ..."
textDevel = "Development by:\n        Javier Villamizar "
textUnder = "Under the supervision by:\n        Michael Alvarez \n        Duwang Prada"
b = np.random.randint(7, size=(5,5))

fig = plt.figure()


fig.canvas.set_window_title('DFBC_python_V1')
legend = fig.text(0.05, 0.95, textstr, color='black')#, fontweight='bold')
legend5 = fig.text(0.5, 0.9, titulodd, color='black', fontweight='bold',fontsize = 14,ha = 'center')
legend3 = fig.text(0.05, 0.44, textstr3, color='black')
legend2 = fig.text(0.04, 0.40, textstr2, color='black')
legend4 = fig.text(0.05, 0.36, textstr4, color='black')
devel = fig.text(0.32, 0.25, textDevel, color='black', style='italic',fontsize = 9)
under = fig.text(0.32, 0.15, textUnder, color='black', style='italic', fontsize = 9)
"right-top",
callback = Index()


initial_text =" "


#Cargar imagen
axcargar = plt.axes([0.05, 0.8, 0.12, 0.05])#0.11
bcargar = Button(axcargar, 'Load')
bcargar.on_clicked(callback.cargarImagen)

#SUPERIOR IZQUIERDA
axcuadSI = plt.axes([0.05, 0.74, 0.12, 0.05])
bcuadSI = Button(axcuadSI, 'Grid')
bcuadSI.on_clicked(callback.cuadricula)

#GRID POR DOS
ax16 = plt.axes([0.05, 0.68, 0.12, 0.05])
b16 = Button(ax16, 'Grid Up')
b16.on_clicked(callback.gridNumpor2)

#GRID DIVIDIDO EN DOS
ax100 = plt.axes([0.05, 0.62, 0.12, 0.05])
b100= Button(ax100, 'Grid Down')
b100.on_clicked(callback.gridNumsobre2)

#PONER PUNTOS
axpuntos = plt.axes([0.05, 0.56, 0.12, 0.05])
bpuntos = Button(axpuntos, 'Load 3D')#anterior era el de puntos
bpuntos.on_clicked(callback.cargar3d)

# #QUITAR PUNTOS.......................
# axquitar = plt.axes([0.05, 0.50, 0.12, 0.05])
# bquitar = Button(axquitar, 'Remove')
# bquitar.on_clicked(callback.Quitar)

#CALCULO DIMENSION FRACTAL
calculo = plt.axes([0.05, 0.30, 0.12, 0.05])
bcalculo = Button(calculo, 'Calculate')
bcalculo.on_clicked(callback.calculoAutomatico)

#GUARDAR
guard = plt.axes([0.05, 0.20, 0.12, 0.05])
bguard = Button(guard, 'Save')
bguard.on_clicked(callback.guardar)

# #CARGAR
# cargaa = plt.axes([0.05, 0.14, 0.12, 0.05])
# bcargaar = Button(cargaa, 'Load Data')
# bcargaar.on_clicked(callback.cargarmat)

#AYUDA
ayuda = plt.axes([0.05, 0.05, 0.12, 0.05])
bayuda = Button(ayuda, 'Help')
bayuda.on_clicked(callback.help)

#Graficas Imagen principal y escudo
array = np.zeros( (100, 100), np.uint8 )
escudo = fig.add_subplot(4, 4,16)
imEsc = cv2.imread('logoupb.jpg')
plt.axis('off')
plt.xticks([])
plt.yticks([])

myobj2 = plt.imshow(cv2.cvtColor(imEsc, cv2.COLOR_BGR2RGB))
ImFractal = fig.add_subplot(4, 4, (1, 12))
plt.axis('off')
plt.xticks([])
plt.yticks([])
myobj = plt.imshow(cv2.cvtColor(array, cv2.COLOR_BGR2RGB),vmin=0,vmax=1)
#Tabla.... Terminar aun
tabla = plt.table(cellText = datos,loc = 'right',colWidths=[0.15, 0.15],cellLoc="left")
scatter = plt.scatter([],[],color='b',linewidths=0.2)
#scatter.remove()



# #ENTRADA PARA GUARDAR
# axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
# text_box = TextBox(axbox, 'Name')
# text_box.on_submit(save)

# #SUPERIOR DERECHA
# axcuadSD = plt.axes([0.32, 0.01, 0.1, 0.075])
# bcuadSD = Button(axcuadSD, 'SD')
# bcuadSD.on_clicked(callback.supder)

# #INFERIOR IZQUIERDA
# axcuadII = plt.axes([0.43, 0.01, 0.1, 0.075])
# bcuadII = Button(axcuadII, 'II')
# bcuadII.on_clicked(callback.infizq)

# #INFERIOR DERECHA
# axcuadID = plt.axes([0.54, 0.01, 0.1, 0.075])
# bcuadID = Button(axcuadID, 'ID')
# bcuadID.on_clicked(callback.infder)


# #GRID 64
# ax64 = plt.axes([0.1, 0.2, 0.1, 0.075])
# b64 = Button(ax64, 'Grid64')
# b64.on_clicked(callback.gridNum64)

# #GRID 256
# ax256 = plt.axes([0.1, 0.1, 0.1, 0.075])
# b256 = Button(ax256, 'Grid256')
# b256.on_clicked(callback.gridNum256)

# #GRID 512
# ax512 = plt.axes([0.1, 0.1, 0.1, 0.075])
# b512 = Button(ax512, 'Grid512')
# b512.on_clicked(callback.gridNum512)


plt.show()
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())

# root = Tk()
# root.withdraw() 
# root.update()
# filename = askopenfilename()
# root.quit()     # stops mainloop
# root.destroy()


# paths = filename#r'D:\MAESTRIA\AVANCES_TESIS\MICHAEL_INTERFAZ\curva-de-koch.jpg'
# im = cv2.imread(paths)
# print(type(im))
# alto, ancho, mat =im.shape
# escAlt =  768 / alto
# escAnc = 1024 / ancho
# print(im.shape)
# print (ancho, alto, mat, type(ancho))
# im_resized = cv2.resize(im, None,fx=escAnc, fy=escAlt, interpolation=cv2.INTER_LINEAR)
# print (im_resized.shape)
# for i in range(0,alto,100):
#         cv2.line(im_resized, (i, 0), (i, im_resized.shape[0]), color=(255, 0, 0), lineType=cv2.LINE_AA, thickness=1)

# for i in range(0,ancho,100):
#         cv2.line(im_resized, (0, i), (im_resized.shape[1], i), color=(255, 0, 0), lineType=cv2.LINE_AA, thickness=1)

# fig = plt.figure() 
# fig.canvas.set_window_title('DFBC_python_v1') 
# plt.imshow(im_resized)
# plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))

# #Elimina ejes y numeros
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])


# # # Elimina los numeros pero deja los ejes
# # ax = plt.gca()
# # ax.axes.xaxis.set_ticklabels([])
# # ax.axes.yaxis.set_ticklabels([])

# ##Cuadricula por defecto de matplotlib
# #plt.grid(True, color='r', linestyle='-', linewidth=1)

# callback = Index()
# axnext = plt.axes([0.81, 0.01, 0.1, 0.075])
# bnext = Button(axnext, 'Next')
# bnext.on_clicked(callback.Puntos)

# plt.show()