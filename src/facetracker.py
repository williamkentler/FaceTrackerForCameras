#!/usr/bin/python
 
#----------------------------------------------------------------------------
# Face Detection Test (OpenCV)
#
# thanks to:
# http://japskua.wordpress.com/2010/08/04/detecting-eyes-with-python-opencv
#----------------------------------------------------------------------------
#import cv2 
import cv2.cv as cv
#import cv
import time
import Image
 
def DetectFace(image, CascadesAndColours, FaceDims):
 
    min_size = (20,20)
    image_scale = 1
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    
    border = 0.2
    
 
    # Allocate the temporary images
#    grayscale = cv.CreateImage((image.width, image.height), 8, 1)
#    smallImage = cv.CreateImage(
#            (
#                cv.Round(image.width / image_scale),
#                cv.Round(image.height / image_scale)
#            ), 8 ,1)
    
    #expand the area around the face slightly
    (x,y,w,h) = FaceDims
    
    #convert to two points and ensure they're within the main image
 
    pt1x = x - cv.Round(border*w)
    pt1y = y - cv.Round(border*h)
    pt2x = x + w + cv.Round(border*w)
    pt2y = y + h + cv.Round(border*h)
    
    pt1 = ( max(pt1x, 0), max(pt1y, 0) )
    pt2 = ( min(pt2x, image.width), min(pt2y, image.height) )
 
    
    newX = pt1[0]
    newY = pt1[1]
    newWidth  = pt2[0] - pt1[0]
    newHeight = pt2[1] - pt1[1] 
        
    ROI = (newX, newY, newWidth, newHeight)
    
    cv.SetImageROI(image, ROI)
    
    smallImage = cv.CreateImage((ROI[2], ROI[3]), 8 ,1)
    grayscale = cv.CreateImage((ROI[2], ROI[3]), 8 ,1)


 
    # Convert color input image to grayscale
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
    cv.ResetImageROI(image)

#    # Scale input image for faster processing
#    cv.Resize(grayscale, smallImage, cv.CV_INTER_LINEAR)
 
    # Equalize the histogram
    cv.EqualizeHist(grayscale, smallImage)

    cv.ShowImage("ROI", smallImage)
#    time.sleep(0.2)
 
    # Detect the objects
    for feature in CascadesAndColours:
        [cascade, colour] = feature #unpack the tuple
        faces = cv.HaarDetectObjects(
                smallImage, cascade, cv.CreateMemStorage(0),
                haar_scale, min_neighbors, haar_flags, min_size
            )
        if faces == (): #don't keep looking if nothing else found
            break
     
        # If faces are found
        if faces:
            for ((x, y, w, h), n) in faces:
                #convert location back to the main image coordinates 
                x = x + newX
                y = y + newY
                # the input to cv.HaarDetectObjects was resized, so scale the
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(image, pt1, pt2, colour, 5, 8, 0)
                
        else:
            x = 0
            y = 0
            w = image.width
            h = image.height
        
 
    return (image, (x,y,w,h))
 
#----------
# M A I N
#----------
showImage = True

 
capture = cv.CaptureFromCAM(-1)
#cv.SetCaptureProperty(capture,  cv.CV_CAP_PROP_FPS, 1)
cv.SetCaptureProperty(capture,  cv.CV_CAP_PROP_FRAME_WIDTH, 320)
cv.SetCaptureProperty(capture,  cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

#capture = cv.CaptureFromFile("C:\Users\wkentler\Desktop\untitled2.avi")

CascadesAndColours = ( (cv.Load("haarcascades/haarcascade_frontalface_alt.xml"), cv.RGB(255, 10, 190)), )

#CascadesAndColours = ( (cv.Load("haarcascades/HS.xml"), cv.RGB(0, 0, 190)), (cv.Load("haarcascades/haarcascade_frontalface_alt.xml"), cv.RGB(255, 10, 190)) )
#faceCascade = cv.Load("haarcascades/haarcascade_frontalface_default.xml")
#faceCascade = cv.Load("haarcascades/haarcascade_frontalface_alt2.xml")
#faceCascade = cv.Load("haarcascades/haarcascade_eyes.xml")
#faceCascade = cv.Load("haarcascades/haarcascade_frontalface_alt_tree.xml")
facedims = (0,0,320,240)
 
while (cv.WaitKey(15)==-1):
    img = cv.QueryFrame(capture)
    (image, facedims) = DetectFace(img, CascadesAndColours, facedims)
#    time.sleep(0.2)
    
    if showImage == True:
        cv.ShowImage("face detection test", image)
 
