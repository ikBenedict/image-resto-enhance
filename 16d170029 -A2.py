#basic image editing and restoration program
#author:Benedict.K.I

from Tkinter import *
import os
import ctypes
import operator
from PIL import ImageTk
from PIL import ImageOps
from PIL import Image
from tkFileDialog import*
import tkMessageBox, tkSimpleDialog
import imghdr
from collections import*
import numpy as np
import cv2
import math
from skimage.measure import compare_ssim as ssim
import scipy.ndimage
from skimage import data
import scipy.misc as sc

                        #######################
######################## RESTORATION FUNCTIONS ###########################
                        #######################

def loadGround(canvas, w):
    if w==0:
        imgeName=canvas.data.imageLocation
        ground=cv2.imread(imgeName,0)
        il= Image.open(imgeName)
        il=il.convert('L')
        canvas.data.image2=il
        canvas.data.image2Size=il.size
        #calling functions to draw image
        canvas.data.imageForTk2=makeImageForTk(canvas, 1)
        drawImage(canvas, 1)
    if w==1:
        return cv2.imread(canvas.data.imageLocation,0)
    
        
def loadkernel(canvas):
    #storing the file location as string
    imgeName=askopenfilename()
    filetype=""
    #making sure it's an image file
    try: filetype=imghdr.what(imgeName)
    except:
        tkMessageBox.showinfo(title="Image File",\
        message="Choose an Image File!" , parent=canvas.data.mainWindow)
    # restricting filetypes to .jpg, .bmp, .png, .tiff
    if filetype in ['jpeg', 'bmp', 'png', 'tiff']:
        kernel=cv2.imread(imgeName,0)
        canvas.data.kernel=kernel
        return kernel

##############PSNR#############
    
def psnr(canvas):
    img1=cv2.imread('ioutput.jpg',0)
    img2=loadGround(canvas, 1)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    d = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    print(d)
    
#############SSIM###############

def compute_ssim(canvas):
    img1=cv2.imread('ioutput.jpg',0)
    img2=loadGround(canvas, 1)
    # k1,k2 & c1,c2 depend on L (width of color map)
    l = 255-0
    k_1 = 0.01
    c_1 = (k_1*l)**2
    k_2 = 0.03
    c_2 = (k_2*l)**2

    window = np.ones((8, 8))

    # window = gauss_2d((11, 11), 1.5)
    # Normalization
    window /= np.sum(window)

    # Convert image matrices to double precision (like in the Matlab version)
    im1 = img1.astype(np.float)
    im2 = img2.astype(np.float)

    # Means obtained by Gaussian filtering of inputs
    mu_1 = scipy.ndimage.filters.convolve(im1, window)
    mu_2 = scipy.ndimage.filters.convolve(im2, window)

    # Squares of means
    mu_1_sq = mu_1**2
    mu_2_sq = mu_2**2
    mu_1_mu_2 = mu_1 * mu_2

    # Squares of input matrices
    im1_sq = im1**2
    im2_sq = im2**2
    im12 = im1*im2

    # Variances obtained by Gaussian filtering of inputs' squares
    sigma_1_sq = scipy.ndimage.filters.convolve(im1_sq, window)
    sigma_2_sq = scipy.ndimage.filters.convolve(im2_sq, window)

    # Covariance
    sigma_12 = scipy.ndimage.filters.convolve(im12, window)

    # Centered squares of variances
    sigma_1_sq -= mu_1_sq
    sigma_2_sq -= mu_2_sq
    sigma_12 -= mu_1_mu_2

    if (c_1 > 0) & (c_2 > 0):
        ssim_map = ((2*mu_1_mu_2 + c_1) * (2*sigma_12 + c_2)) / ((mu_1_sq + mu_2_sq + c_1) * (sigma_1_sq + sigma_2_sq + c_2))
    else:
        numerator1 = 2 * mu_1_mu_2 + c_1
        numerator2 = 2 * sigma_12 + c_2

        denominator1 = mu_1_sq + mu_2_sq + c_1
        denominator2 = sigma_1_sq + sigma_2_sq + c_2

        ssim_map = np.ones(mu_1.size)

        index = (denominator1 * denominator2 > 0)

        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    # return MSSIM
    index = np.mean(ssim_map)

    print index


def gauss_2d(shape=(3, 3), sigma=0.5):
    """
    Code from Stack Overflow's thread
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

#################INVERSE FILTER###############

def num1(canvas):
    #input text box with (title, message)
    num1=tkSimpleDialog.askfloat("Filter radius", "Enter the inverse filtering redius in pixels")
    return num1

def pin(canvas):
    im1=canvas.data.blur
    im2=canvas.data.kernel
    r=num1(canvas)
    I=np.fft.fft2(im1)
    J=np.fft.fft2(im2)
    for u in xrange(I.shape[0]):
        for v in xrange(I.shape[1]):
            if u*u+v*v>r*r:
                if (800-u)**2.0+v*v>r*r:
                    if u*u+(800-v)**2.0>r*r:
                        if (800-u)**2.0+(800-v)**2.0>r*r:
                            J.itemset((u,v),1)
    k=abs(np.fft.ifftshift(np.fft.ifft2(I/J)))
    outpil(canvas, k, 1)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    drawImage(canvas, 0)
    #outputting the image for further access
    outim(canvas)
    canvas.data.undoQueue.append(canvas.data.image.copy())

################WEINER FILTER####################

def num2(canvas):
    #input text box with (title, message)
    num2=tkSimpleDialog.askfloat("K value", "Enter the value of K")
    return num2

def weiner(canvas):
    d=canvas.data.blur
    d=d+np.random.randn(d.shape[0],d.shape[1])*200.0
    D=np.fft.fft2(d)
    k=num2(canvas)
    H=np.fft.fft2(canvas.data.kernel)
    M = (np.conj(H)/(np.abs(H)**2.0 + k))*D
    m_map=abs(np.fft.fftshift(np.fft.ifft2(M)))
    outpil(canvas, m_map, 1)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    drawImage(canvas, 0)
    #outputting the image for further access
    outim(canvas)
    canvas.data.undoQueue.append(canvas.data.image.copy())

###############CLSF###################

def num3(canvas):
    #input text box with (title, message)
    num3=tkSimpleDialog.askfloat("Y value", "Enter the value of Y(Gamma)")
    return num3

def CLSF(canvas):
    e=canvas.data.blur
    e=e+np.random.randn(e.shape[0],e.shape[1])*200.0
    E=np.fft.fft2(e)
    k=num3(canvas)
    L=np.zeros((800,800), np.float32)
    for u in range(0,400):
        for v in range(0,400):
            L[u,v]=L[799-u,799-v]=L[799-u,v]=L[u,799-v]=(u**2.0+v**2.0)
    H=np.fft.fft2(canvas.data.kernel)
    N = (np.conj(H)/(np.abs(H)**2.0 + k*np.abs(L)**2.0))*E
    cl=abs(np.fft.fftshift(np.fft.ifft2(N)))
    outpil(canvas, cl, 1)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    drawImage(canvas, 0)
    #outputting the image for further access
    outim(canvas)
    canvas.data.undoQueue.append(canvas.data.image.copy())


#####################################################################
####################### EDITING FUNCTIONS ###########################
#####################################################################

##################### Gaussian Blur #####################
            
def une(canvas, float):
    #input text box with (title, message)
    une=tkSimpleDialog.askfloat("Blur intensity", "Enter the value of standard deviation")
    return une

def Gblur(canvas):
    pik=cv2.imread("ioutput.jpg")
    #sigma or standard deviation
    d=une(canvas, float)
    p=d*d
    q=1.0/p
    #creating a kernel with exp(-(x^2+y^2)/2)
    kernel=np.array(
        [[0.0183,0.0821,0.1353,0.0821,0.0183],
         [0.0821,0.3679,0.6065,0.3679,0.0821],
         [0.1353,0.6065,1,0.6065,0.1353],
         [0.0821,0.3679,0.6065,0.3679,0.0821],
         [0.0183,0.0821,0.1353,0.0821,0.0183]])
    s=0
    for i in xrange(kernel.shape[0]):
        for j in xrange(kernel.shape[1]):
            #raising each element in kernel to the power of (-1/d^2)
            kernel[i,j]=kernel[i,j]**q
            #dividing each element with (2*pi*d^2)
            kernel[i,j]=kernel[i,j]*7/(44*p)
            #storing the value of the element
            s=s+kernel[i,j]
    for i in xrange(kernel.shape[0]):
        for j in xrange(kernel.shape[1]):
            #normalizing the kernel
            kernel[i,j]=kernel[i,j]/s
    #convolving(actually correlation is done) the image with the kernel
    #using library function to make the process faster
    custom=cv2.filter2D(pik, -1, kernel)
    #converting cv2 image into pil image   
    outpil(canvas, custom, 0)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    drawImage(canvas, 0)
    #outputting the image for further access
    outim(canvas)
    #adding entry into the undo queue
    canvas.data.undoQueue.append(canvas.data.image.copy())
                

################### Equalize Histogram ################

def Equalize(canvas):
    emaje=canvas.data.image
    # calculating lookup table
    lut=Eqm(emaje.histogram())
    # mapping image throught the lookup table
    emaje=emaje.point(lut)
    canvas.data.image=emaje
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    drawImage(canvas, 0)
    outim(canvas)
    #adding entry into the undo queue
    canvas.data.undoQueue.append(canvas.data.image.copy())
    
def Eqm(canvas):
    #empty lookup table
    lut = []
    #for b in range(0, len(canvas), 256):
    for b in range(0, len(canvas), 256):
        # creating a step and calculating its size
        step = reduce(operator.add, canvas[b:b+256]) / 255
        n = 0
        #adding elements into the lookup table
        for i in range(256):
            lut.append(n / step)
            n = n + canvas[i+b]
    return lut

################## Log Transform ######################

def Logt(canvas):
    pik=cv2.imread("ioutput.jpg")
    #checking the image type: grayscale or colour
    if(len(pik.shape)<3):
        #changing the grayscale value
        for m in xrange(pik.shape[0]):
            for n in xrange(pik.shape[1]):
                v=float(pik.item(m,n))
                v=31.875*math.log(v+1,2)
                pik.itemset((m,n),v)
    else:
        #converting to hsv
        pik=cv2.cvtColor(pik, cv2.COLOR_BGR2HSV)
        for m in xrange(pik.shape[0]):
            for n in xrange(pik.shape[1]):
                # scaling down value'V' to (0,1)
                v=float(pik.item(m,n,2))
                #applying the log transform
                v=31.875*math.log(v+1,2)
                #changing the value'V' to the calculated value
                pik.itemset((m,n,2),v)
    #converting back to bgr
    pik=cv2.cvtColor(pik, cv2.COLOR_HSV2BGR)
    outpil(canvas, pik, 0)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    outim(canvas)
    drawImage(canvas, 0)
    #adding entry into the undo queue
    canvas.data.undoQueue.append(canvas.data.image.copy())
                
################# Gamma Correction ###################\

def unp(canvas, float):
    unp=tkSimpleDialog.askfloat("Gamma Correction", "Enter the power value")
    return unp

def Gamma(canvas):
    pic=cv2.imread("ioutput.jpg")
    #calling the input text box
    g=unp(canvas,float)
    gi=1.0/g
    #checking the colour type of image and applying the power law tranform
    #to the colour of each element by selecting them indivually
    #shape[0] is row length, shape[1] is coloumn length, shape[2] is number of colour channels
    if(len(pic.shape)<3):
        for a in xrange(pic.shape[0]):
            for b in xrange(pic.shape[1]):
                d=pic.item(a,b)
                #selecting the intensity(colour), scaling it to (0,1),
                #raising it to 1/(input value), scaling it back to (0,255).   
                d=int((((float(d))/255.0)**gi)*255)
                pic.itemset((a,b),d)

    else:
        #converting to hsv
        pic=cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
        for a in xrange(pic.shape[0]):
            for b in xrange(pic.shape[1]):
                #for c in xrange(pic.shape[2]):
                d=pic.item(a,b,2)
                d=int((((float(d))/255.0)**gi)*255)
                pic.itemset((a,b,2),d)
    pic=cv2.cvtColor(pic, cv2.COLOR_HSV2BGR)
    outpil(canvas, pic, 0)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    outim(canvas)
    drawImage(canvas, 0)
    #adding entry into the undo queue
    canvas.data.undoQueue.append(canvas.data.image.copy())
    
################## Sharpness ########################

def uni(canvas, float):
    un=tkSimpleDialog.askfloat("Sharpness intensity", "Enter the Sharpness intensity in number as a factor of the current value")
    return un

def Sharpen(canvas):
    #reading the image into cv2 format
    f=cv2.imread("ioutput.jpg")
    #calling the input text box
    unp=uni(canvas,float)
    #creating a 5*5 kernel with all zeroes
    kernel=np.zeros((5,5),np.float32)
    #setting the centre element to 1
    kernel[2,2]=1
    #creating another kernel with all ones and normalizing it
    box=np.ones((5,5),np.float32)/25.0
    #creating a kernel for edge detection
    #g(x,y)=f(x,y)-f'(x,y)
    kernel=kernel-box
    #creating a kernel with all zeroes and centre element set to 1
    sharp=np.zeros((5,5),np.float32)
    sharp[2,2]=1
    #superimposing the image with 'k' times the edge detected kernel 
    #G(x,y) = f(x,y)+k*g(x,y) 
    sharp=sharp+unp*kernel
    #convolving the image with 'sharp' kernel
    #using library function to make the process faster
    custom=cv2.filter2D(f, -1, sharp)
    outpil(canvas, custom, 0)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    outim(canvas)
    drawImage(canvas, 0)
    #adding entry into the undo queue
    canvas.data.undoQueue.append(canvas.data.image.copy())
    
##################### BRIGHTNESS ################    
    
def closeBrightnessWindow(canvas):
    #checking the presence of image on the canvas
    if canvas.data.image!=None:
        #outputting the image
        outim(canvas)
        #drawing the image on canvas
        canvas.data.imageForTk=makeImageForTk(canvas, 0)
        drawImage(canvas, 0)
        #adding the untransformed image into undo queue
        canvas.data.undoQueue.append(canvas.data.image.copy())
        #closing the window
        canvas.data.brightnessWindowClose=True

def changeBrightness(canvas, brightnessWindow, brightnessSlider, \
                     previousVal):
    if canvas.data.brightnessWindowClose==True:
        brightnessWindow.destroy()
        canvas.data.brightnessWindowClose=False
        
    else:
        # increasing pixel values according to difference between
        # slider current value and previous value
        # checking if there exists a image on canvas and the brightness window exist
        if canvas.data.image!=None and brightnessWindow.winfo_exists():
            # getting the slider value 
            sliderVal=brightnessSlider.get()
            # scaling the value to (-1,1)
            scale=(sliderVal-previousVal)/100.0
            #multiplying each element by the scaled value
            canvas.data.image=canvas.data.image.point(\
                lambda i: i+ int(round(i*scale)))
            canvas.after(200, \
            lambda: changeBrightness(canvas, brightnessWindow, \
                                     brightnessSlider, sliderVal))
       
def brightness(canvas):
    #displaying brightness window at the top with title
    brightnessWindow=Toplevel(canvas.data.mainWindow)
    brightnessWindow.title("Brightness")
    #setting brightness slider range and orientation
    brightnessSlider=Scale(brightnessWindow, from_=-100, to=100,\
                           orient=HORIZONTAL)
    #customizing the brightness window
    brightnessSlider.pack()
    #OK button, frame and position
    OkBrightnessFrame=Frame(brightnessWindow)
    #calling closeBrightnessWindow function on pressing OK
    OkBrightnessButton=Button(OkBrightnessFrame, text="OK", \
                              command=lambda: closeBrightnessWindow(canvas))
    OkBrightnessButton.grid(row=0,column=0)
    OkBrightnessFrame.pack(side=BOTTOM)
    changeBrightness(canvas, brightnessWindow, brightnessSlider,0)
    brightnessSlider.set(0)

################### REVERT #####################

def reset(canvas):
    # changing back to original image
    if canvas.data.image!=None:
        canvas.data.image=canvas.data.originalImage.copy()
        outim(canvas)
        canvas.data.undoQueue.append(canvas.data.image.copy())
        canvas.data.imageForTk=makeImageForTk(canvas, 0)
        drawImage(canvas,0)

################ ROTATE 90CW #################

def transpose(canvas):
    # treating the image as a continuous list of pixel values row-wise
    # and exchanging the rows and the coloums
    # in oder to make it rotate clockewise
    if canvas.data.image!=None:
        imageData=list(canvas.data.image.getdata())
        #creating an empty array
        newData=[]
        #new image with similar dimensions as the original image 
        newimg=Image.new(canvas.data.image.mode,\
                (canvas.data.image.size[1], canvas.data.image.size[0]))
        for i in xrange(canvas.data.image.size[0]):
            #creating another empty array
            addrow=[]
            for j in xrange(i, len(imageData), canvas.data.image.size[0]):
                #add the column data of the original image
                addrow.append(imageData[j])
            #convert the column into row
            addrow.reverse()
            #add a new row to the newData array
            newData+=addrow
        #put the data 
        newimg.putdata(newData)
        canvas.data.image=newimg.copy()
        outim(canvas)
        canvas.data.undoQueue.append(canvas.data.image.copy())
        canvas.data.imageForTk=makeImageForTk(canvas, 0)
        drawImage(canvas, 0)

################ Other functions #################

def outim(canvas) :
    #saves the current edited image in the directory in which the program is running
    #so that it can be used for importing and applying function in cv2 library
    if canvas.data.image!=None:
        im=canvas.data.image
        im.save("ioutput.jpg",canvas.data.Filetype)
    
def outpil(canvas, imaze, z):
    if z==0:
    #saves the cv2 or scipy.misc image and reads it as pil image
        cv2.imwrite("cvoutp.jpg", imaze)
        inpil=Image.open("cvoutp.jpg")
        canvas.data.image=inpil
    if z==1:
        sc.imsave('scoutp.jpg', imaze)
        inpil=Image.open("scoutp.jpg")
        canvas.data.image=inpil
 
################ EDIT MENU FUNCTIONS ############################

def keyPressed(canvas, event):
    #assigning keys for undo and redo actions
    if event.keysym=="z":
        undo(canvas)
    elif event.keysym=="y":
        redo(canvas)
        

# using deques to make Undo and Redo queues
# after each change, we add the new version of the image to
# the Undo queue
def undo(canvas):
    if len(canvas.data.undoQueue)>0:
        # the last element of the Undo Deque is the
        # current version of the image
        lastImage=canvas.data.undoQueue.pop()
        # we would want the current version if we hit redo after undo
        canvas.data.redoQueue.appendleft(lastImage)
    if len(canvas.data.undoQueue)>0:
        # the previous version of the image
        canvas.data.image=canvas.data.undoQueue[-1]
    outim(canvas)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    drawImage(canvas, 0)

def redo(canvas):
    if len(canvas.data.redoQueue)>0:
        canvas.data.image=canvas.data.redoQueue[0]
    outim(canvas)
    if len(canvas.data.redoQueue)>0:
        # we remove this version from the Redo Deque beacuase it
        # has become our current image
        lastImage=canvas.data.redoQueue.popleft()
        canvas.data.undoQueue.append(lastImage)
    canvas.data.imageForTk=makeImageForTk(canvas, 0)
    drawImage(canvas, 0)

############# MENU COMMANDS ################

def saveAs(canvas):
    # asks where the user wants to save the file
    if canvas.data.image!=None:
        filename=asksaveasfilename(defaultextension=".jpg")
        im=canvas.data.image
        im.save(filename)

def save(canvas):
    #saves the file
    if canvas.data.image!=None:
        im=canvas.data.image
        im.save(canvas.data.imageLocation)

def colorcon(canvas, float):
    #grayscale conversion dialog box
    con=tkSimpleDialog.askfloat("Load Grayscale image", "Enter 1 if you want to load image as Grayscale")
    return con

def newImage(canvas):
    #storing the file location as string
    imageName=askopenfilename()
    filetype=""
    #making sure it's an image file
    try: filetype=imghdr.what(imageName)
    except:
        tkMessageBox.showinfo(title="Image File",\
        message="Choose an Image File!" , parent=canvas.data.mainWindow)
    # restricting filetypes to .jpg, .bmp, .png, .tiff
    if filetype in ['jpeg', 'bmp', 'png', 'tiff']:
        #storing the image location in imageName
        canvas.data.imageLocation=imageName
        #opens image
        im= Image.open(imageName)
        #opens grayscale conversion dialog box
        con=colorcon(canvas, float)
        if con==1.0:
            #converts image to grayscale
            im=im.convert('L')
        #storing the extension of the image
        canvas.data.Filetype=filetype
        kerm=np.fft.fft2(loadkernel(canvas))
        im=abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(im)*kerm)))
        canvas.data.blur=im
        outpil(canvas, im, 1)
        outim(canvas)
        loadGround(canvas,0)
        #creating a copy to be used for reset function
        canvas.data.originalImage=im.copy()
        #appending the image to undo queue
        canvas.data.undoQueue.append(im.copy())
        #Original Image dimensions
        canvas.data.imageSize=im.size
        #calling functions to draw image
        canvas.data.imageForTk=makeImageForTk(canvas, 0)
        drawImage(canvas, 0)
    else:
        tkMessageBox.showinfo(title="Image File",\
        message="Choose an Image File!" , parent=canvas.data.mainWindow)

######## CREATE A VERSION OF IMAGE TO BE DISPLAYED ON THE CANVAS #########

def makeImageForTk(canvas, x):
    if x == 0:
        im=canvas.data.image
        if canvas.data.image!=None:
            imageWidth=canvas.data.image.size[0] 
            imageHeight=canvas.data.image.size[1]
        #resizing the image make biggest version to fit inside the canvas
            if imageWidth>imageHeight:
                resizedImage=im.resize((canvas.data.width,\
                int(round(float(imageHeight)*canvas.data.width/imageWidth))))
            # store the scale so as to use it later
                canvas.data.imageScale=float(imageWidth)/canvas.data.width
            else:
                resizedImage=im.resize((int(round(float(imageWidth)*canvas.data.height/imageHeight)),\
                                    canvas.data.height))
                canvas.data.imageScale=float(imageHeight)/canvas.data.height
        # storing the resized image attributes
            canvas.data.resizedIm=resizedImage
            return ImageTk.PhotoImage(resizedImage)
    else:
        il=canvas.data.image2
        #if canvas.data.image2!=None:
        image2Width=canvas.data.image2.size[0] 
        image2Height=canvas.data.image2.size[1]
        #resizing the image make biggest version to fit inside the canvas
        if image2Width>image2Height:
            resizedImage2=il.resize((canvas.data.width,\
                int(round(float(image2Height)*canvas.data.width/image2Width))))
            # store the scale so as to use it later
            canvas.data.image2Scale=float(image2Width)/canvas.data.width
        else:
            resizedImage2=il.resize((int(round(float(image2Width)*canvas.data.height/image2Height)),\
                                    canvas.data.height))
            canvas.data.image2Scale=float(image2Height)/canvas.data.height
        # storing the resized image attributes
        canvas.data.resizedIm2=resizedImage2
        return ImageTk.PhotoImage(resizedImage2)
        
    
 
def drawImage(canvas, y):
    if y==0:
        if canvas.data.image!=None:
        # making the canvas center and the image center the same
            canvas.create_image(canvas.data.width/2.0-canvas.data.resizedIm.size[0]/2.0,
                        canvas.data.height/2.0-canvas.data.resizedIm.size[1]/2.0,
                            anchor=NW, image=canvas.data.imageForTk)
            canvas.data.imageTopX=int(round(canvas.data.width/2.0-canvas.data.resizedIm.size[0]/2.0))
            canvas.data.imageTopY=int(round(canvas.data.height/2.0-canvas.data.resizedIm.size[1]/2.0))
    else:
        # making the canvas center and the image center the same
        canvas.create_image(501.0+canvas.data.width/2.0-canvas.data.resizedIm2.size[0]/2.0,
                        canvas.data.height/2.0-canvas.data.resizedIm2.size[1]/2.0,
                            anchor=NW, image=canvas.data.imageForTk2)
        canvas.data.imageTopX=int(round(501.0+canvas.data.width/2.0-canvas.data.resizedIm2.size[0]/2.0))
        canvas.data.imageTopY=int(round(canvas.data.height/2.0-canvas.data.resizedIm2.size[1]/2.0))

############ INITIALIZE ##############

def init(root, canvas):
    #initializing the program
    #buttons intialization
    buttonsInit(root, canvas)
    #menu bars initialization
    menuInit(root, canvas)
    canvas.data.image=None
    canvas.data.brightnessWindowClose=False
    canvas.data.brightnessLevel=None
    #initializing undo, redo deques and setting them to 100 list items
    canvas.data.undoQueue=deque([], 100)
    canvas.data.redoQueue=deque([], 100)
    canvas.pack()
                   
############## BUTTONS ################
                   
def buttonsInit(root, canvas):
    #initializing the buttons
    backgroundColour="white"
    buttonWidth=16
    buttonHeight=2
    toolKitFrame=Frame(root)
    #defining the buttons and setting their width, height, color and assigning funtctions
    inversefbutton=Button(toolKitFrame, text="Inverse filter",\
                      background=backgroundColour, width=buttonWidth,\
                      height=buttonHeight,command=lambda: pin(canvas))
    inversefbutton.grid(row=1,column=0)
    weinerfbutton=Button(toolKitFrame, text="Weiner Filtering",\
                      background=backgroundColour, width=buttonWidth,\
                      height=buttonHeight,command=lambda: weiner(canvas))
    weinerfbutton.grid(row=2,column=0)
    CLSFbutton=Button(toolKitFrame, text="CLS filtering",\
                      background=backgroundColour, width=buttonWidth,\
                      height=buttonHeight,command=lambda: CLSF(canvas))
    CLSFbutton.grid(row=3,column=0)
    psnrbutton=Button(toolKitFrame, text="PSNR",\
                      background=backgroundColour, width=buttonWidth,\
                      height=buttonHeight,command=lambda: psnr(canvas))
    psnrbutton.grid(row=4,column=0)
    ssimbutton=Button(toolKitFrame, text="SSIM",\
                      background=backgroundColour, width=buttonWidth,\
                      height=buttonHeight,command=lambda: compute_ssim(canvas))
    ssimbutton.grid(row=5,column=0)
    #assigning the position of the button frame
    toolKitFrame.pack(side=LEFT)
    frame2=Frame(root)
    undobutton=Button(frame2, text="Undo",\
                      background="white", width=4,\
                      height=1,command=lambda: undo(canvas))
    undobutton.grid(row=0,column=0)
    rotatebutton=Button(frame2, text="Rotate",\
                      background="white", width=6,\
                      height=1,command=lambda: transpose(canvas))
    rotatebutton.grid(row=0,column=1)
    redobutton=Button(frame2, text="Redo",\
                      background="white", width=4,\
                      height=1,command=lambda: redo(canvas))
    redobutton.grid(row=0,column=2)
    
    frame2.pack(side=BOTTOM)

###################### MENU #######################

def menuInit(root, canvas):
    menubar=Menu(root)
    ## File pull-down Menu
    #adding drop down menus, listing and assigning the functions in the menu
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="New", \
                         command=lambda:newImage(canvas))
    filemenu.add_command(label="Save", \
                         command=lambda:save(canvas))
    filemenu.add_command(label="Save As", \
                         command=lambda:saveAs(canvas))
    menubar.add_cascade(label="File", menu=filemenu)
    root.config(menu=menubar)
    ## Edit pull-down Menu
    editmenu = Menu(menubar, tearoff=0)
    editmenu.add_command(label="Undo (Z)", command=lambda:undo(canvas))
    editmenu.add_command(label="Redo (Y)", command=lambda:redo(canvas))
    editmenu.add_command(label="Revert", command=lambda:reset(canvas))
    editmenu.add_command(label="Rotate 90CW", command=lambda:transpose(canvas))
    menubar.add_cascade(label="Edit", menu=editmenu)
    root.config(menu=menubar)
    ## Enhancements pull down menu
    enhcmenu = Menu(menubar, tearoff=1)
    enhcmenu.add_command(label="Brightness", command=lambda:brightness(canvas))
    enhcmenu.add_command(label="Gaussian Blur", command=lambda:Gblur(canvas))
    enhcmenu.add_command(label="Equalize Histogram", command=lambda:Equalize(canvas))
    enhcmenu.add_command(label="Log transform", command=lambda:Logt(canvas))
    enhcmenu.add_command(label="Gamma correction", command=lambda:Gamma(canvas))
    enhcmenu.add_command(label="Sharpness", command=lambda:Sharpen(canvas))
    menubar.add_cascade(label="Enhance", menu=enhcmenu)
    root.config(menu=menubar)
    
def run():
    # create the root and the canvas
    root = Tk()
    root.title("Image Editor")
    canvasWidth=1001
    canvasHeight=500
    #creating the canvas using the paramenters defined above
    canvas = Canvas(root, width=canvasWidth, height=canvasHeight, \
                    background="silver")
    # Set up canvas data and call init
    class Struct: pass
    canvas.data = Struct()
    #defining some functions so as to use them later
    canvas.data.width=500
    canvas.data.height=500
    canvas.data.mainWindow=root
    init(root, canvas)
    root.bind("<Key>", lambda event:keyPressed(canvas, event))
    # and launch the app
    root.mainloop()  # This call BLOCKS (so your program waits)


run()
