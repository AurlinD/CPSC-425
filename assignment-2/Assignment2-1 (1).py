from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc


#Question 2

template = Image.open('/Users/aurlin/Desktop/425/HW2/hw2/faces/template.jpg')
template = template.convert('L')

student = Image.open('/Users/aurlin/Desktop/425/HW2/hw2/faces/students.jpg')
student = student.convert('L')

family = Image.open('/Users/aurlin/Desktop/425/HW2/hw2/faces/family.jpg')
family = family.convert('L')

judy = Image.open('/Users/aurlin/Desktop/425/HW2/hw2/faces/judybats.jpg')
judy = judy.convert('L')

fans = Image.open('/Users/aurlin/Desktop/425/HW2/hw2/faces/fans.jpg')
fans = fans.convert('L')

sports = Image.open('/Users/aurlin/Desktop/425/HW2/hw2/faces/sports.jpg')
sports = sports.convert('L')

tree = Image.open('/Users/aurlin/Desktop/425/HW2/hw2/faces/tree.jpg')
tree = tree.convert('L')



def MakePyramid(image, minsize):
    pyramidArray = [] #create empty array for list of images to be inserted to
    condition = True #boolean value for while condition

    while(condition):
        width = image.size[0]
        height = image.size[1]
        #insert image into array
        pyramidArray.append(image) 
        #image.show()
        
        #resize image function with given width and height
        image = image.resize((int(width*0.75),int(height*0.75)), Image.BICUBIC)
        
        #while loop boolean check if condition is satisfied
        if ((width <= minsize) or (height <= minsize)):
            condition = False;
            
    #print pyramidArray
    return pyramidArray
    
   
#MakePyramid(judy,10)

#Question 3

def ShowPyramid(pyramid):
    #dimensions for blank image
    width = 0 
    height = 0 
    offset_x = 0
    offset_y = 0 #constant
    
    #need to find tota width and height of all the images in given list
    for im in pyramid:
        currWidth = im.size[0]
        currHeight = im.size[1]
        # find total width of all images so they can stack side by side for blank image width size
        width = width + currWidth
        # find the tallest image for blank image height size
        if (currHeight > height):
            height = currHeight
        
     
    #creation of blank image
    image = Image.new("L", (width, height), "white")
    
     
    #loop in given array to paste images to blank image
    for im in pyramid:
        # get width distance of each image for offset_x
        width = im.size[0]
        #paste image onto blank image to form continuous image
        image.paste(im,(offset_x,offset_y))
        #offset value needed for next image to be inserted in x-direction
        offset_x = offset_x + width
        
    #show the singleimage
    image.show()
    
#ShowPyramid(MakePyramid(judy,10))


#Question 4

def FindTemplate(pyramid, template, threshold): 
	finder = []
	width = template.size[0]
	height = template.size[1]
	minWidth = 15
	
	

	#resize image with minWidth. Appy ratio of image on y-direction 
	ratio = width/minWidth
	yRatio=  height/ratio
	template = template.resize((minWidth, yRatio), Image.BICUBIC)

	#compute the NCC function in for loop
	for im in pyramid:
		thresholdLevel = np.where(ncc.normxcorr2D(im, template) > threshold)
		thresholdLevel0 = thresholdLevel[0]
		thresholdLevel1 = thresholdLevel[1]

		#pairs elements of list together
		finder.append(zip(thresholdLevel1, thresholdLevel0))

	# convert pyramid to RGB from L
	img = pyramid[0].convert('RGB')

	
	lengthFinder = len(finder)
	for index in range(lengthFinder):
		resize = 0.75 ** index
		resize2 = resize * 2
                #off-set the center values for x and y to get the 4 respective coordinate values
                #to make the rectangle
		for coord in finder[index]:
			x1 = (coord[0]//resize)- (template.size[0]//resize2)
			x2 = (coord[0]//resize)+ (template.size[0]//resize2)
			y1 = (coord[1]//resize)- (template.size[1]//resize2)
			y2 = (coord[1]//resize)+ (template.size[1]//resize2)

			#draw the recangle
			draw = ImageDraw.Draw(img)
			draw.rectangle((x1,y1,x2,y2),outline="red")
			del draw

	return img

    


  

         
         
         
        




    

    




    
        
            
            
    
        
        
        
    
    
    