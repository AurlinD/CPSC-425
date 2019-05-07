from PIL import Image
import numpy as np
import math
from scipy import signal

#Q1

#box filter takes in an integer n value and returns box filter of size n*n.
#if box filter is even, returns a assert error statement

def boxfilter(n) :
    #checks if matrix is an odd size
    assert (n % 2 != 0), 'Error: n matrix is even size'
    return np.ndarray(shape=(n,n), dtype=float, order='F')
    


#Q2
       
def gauss1d(sigma) : 
    round = int(math.ceil(sigma*6))
    
    if (round % 2 == 0): 
        round = round + 1
    # creates array of size 6*sigma
    array = np.random.randint(10, size = round)
    
    # apply exp equation to every item in list
    exp = map(lambda x: np.exp(- x**2 / (2*sigma**2)), array)
    # normalize equations
    result = map(lambda x: x/sum(exp), exp)
    return result

#Q3

def gauss2d(sigma) :
    # create a 1d array
    filter1D = np.array(gauss1d(sigma))
    # create the 2d version of 1d array
    filter2D = filter1D[np.newaxis]
    # create a transposed version of 2d array
    filter2DTranspose = filter2D.T
    #apply convolution
    result = signal.convolve2d(filter2D, filter2DTranspose, mode='full', boundary='fill')
    return result
    


#Q4 Part A

def gaussconvolve2d(array,sigma) :
    
    filter = gauss2d(sigma)
    #apply convolve onto filter
    convolve = signal.convolve2d(array,filter,'same')
    return convolve



#Q4 Part B and C

#Apply greyscale and blur
def greyImage(image,sigma):
    im = Image.open(image)
    print im.size, im.mode, im.format
    #Convert to grey scale
    im = im.convert('L')

    #convert image to numpy
    im_array = np.asarray(im)

    #apply gaussconvolve2d
    result =gaussconvolve2d(im_array,sigma)

    #convert back to image
    im3 = Image.fromarray(result)
    return im3
    
greyImage('/Users/aurlin/Desktop/dog.jpg',3).show()

# covolution are linear operations on signal/signal modifiers.
# correlation measures siilarity between two signals
# convolution rotates matrix upside down
    


#Q5
# A faster method would be using seperability which is to split the 2D array into 2 1D arrays and preform two
# seperability convultions which we would combine afterwards by taking the product
# this method would take 2m * n^2 multiplications

# Doing a single convultion over a 2D array would take m^2 * m^2 multiplications
# which is slower than splitting 2D array

    


#part2---------------------------------------------------------

#Q1
def blurPicture(image,sigma):
    im = Image.open(image)
    #convert image to numpy
    im_array = np.asarray(im)
    #Seperate into RGB channels
    red = im_array[:,:,0]
    green = im_array[:,:,1]
    blue = im_array[:,:,2]
    #Apply Convolve on the RGB channels
    blurRed = gaussconvolve2d(red,sigma)
    blurGreen= gaussconvolve2d(green,sigma)
    blurBlue = gaussconvolve2d(blue,sigma)

    #Combine Channels into array,convert array to image, and return
    rgb = np.dstack((blurRed, blurGreen, blurBlue))
    rgb = rgb.astype('uint8')

    hybridImage = Image.fromarray(rgb)
    hybridImage.save('hybrid-image.png','PNG')
    return hybridImage
    
    
blurPicture('/Users/aurlin/Desktop/dog.jpg',3).show()

#Q2
def HighFrequency(image,sigma):
    im = Image.open(image)
    #convert image to numpy
    im_arraystandard = np.asarray(im)
    #converts array values to doubles 
    im_array =im_arraystandard.astype(np.float)
    #Seperate into RGB channels
    red = im_array[:,:,0]
    green = im_array[:,:,1]
    blue = im_array[:,:,2]
    #Apply Convolve on the RGB channels
    blurRed = gaussconvolve2d(red,sigma)
    blurGreen= gaussconvolve2d(green,sigma)
    blurBlue = gaussconvolve2d(blue,sigma)
    #Subtract original values with the convolved values    
    freqRed = np.subtract(red,blurRed)
    freqGreen = np.subtract(green,blurGreen)
    freqBlue = np.subtract(blue,blurBlue)
    
    #Combine Channels into array, convert array to image, add 0.5 to values, return image
    rgb = np.dstack((freqRed, freqGreen, freqBlue))
    rgb2= np.add(rgb,np.full(rgb.shape,128))
   
    rgb2 = rgb2.astype('uint8')
    for array in rgb2:
        array = map(lambda x:x+0.5, array)
                
    hybridImage = Image.fromarray(rgb2)    
    hybridImage.save('hybrid-image.png','PNG')
    return hybridImage
    
HighFrequency('/Users/aurlin/Desktop/cat.jpg',3).show()


#part 3

def HighAndLowFrequency (image1, image2, sigma1, sigma2):
  
    
    
#--------------------------------------------------------
#code taken from part 2 Q1

    im = Image.open(image1)
   
    #convert image to numpy
    im_arraystandard = np.asarray(im)
    #converts array values to doubles 
    im_array =im_arraystandard.astype(np.float)
    #Seperate into RGB channels
    red = im_array[:,:,0]
    green = im_array[:,:,1]
    blue = im_array[:,:,2]
    #Apply Convolve on the RGB channels
    blurRed = gaussconvolve2d(red,sigma1)
    blurGreen= gaussconvolve2d(green,sigma1)
    blurBlue = gaussconvolve2d(blue,sigma1)   
    
#---------------------------------------------------------
    #code taken from part 2 Q2
    
    im1 = Image.open(image2)

    #convert image to numpy
    im_arraystandard1 = np.asarray(im1)
    #converts array values to doubles 
    im_array1 =im_arraystandard1.astype(np.float)
    #Seperate into RGB channels
    red1 = im_array1[:,:,0]
    green1 = im_array1[:,:,1]
    blue1 = im_array1[:,:,2]
    #Apply Convolve on the RGB channels
    blurRed1 = gaussconvolve2d(red1,sigma2)
    blurGreen1= gaussconvolve2d(green1,sigma2)
    blurBlue1 = gaussconvolve2d(blue1,sigma2)
    #Subtract original values with the convolved values      
    freqRed = np.subtract(red1,blurRed1)
    freqGreen = np.subtract(green1,blurGreen1)
    freqBlue = np.subtract(blue1,blurBlue1)
    
    # Add High and Low frequency values together for RGB channels
    mixRed = np.add(freqRed,blurRed)
    mixGreen = np.add(freqGreen,blurGreen)
    mixBlue = np.add(freqBlue,blurBlue)
    
    # Combine channels into array, convert array to image and return
    mixArray = np.dstack([mixRed,mixGreen,mixBlue])
    Result = mixArray.astype('uint8')
    Result1 = Image.fromarray(Result)
    Result1.show()
 
HighAndLowFrequency ('/Users/aurlin/Desktop/dog.jpg','/Users/aurlin/Desktop/cat.jpg', 3, 3)  

    
    
    




