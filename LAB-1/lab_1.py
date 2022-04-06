import cv2
import numpy as np

def read_and_display_image(img):

    # Display the image
    cv2.imshow('image', img)

    # Wait for a key to be pressed
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()

def check_colour_space_image(img):
    
    B,G,R = cv2.split(img)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Corresponding channels are seperated
    
    cv2.imshow("Original", img)
    cv2.waitKey(0)
    #RGB Color Spaces
    cv2.imshow("Red", R)
    cv2.waitKey(0)

    cv2.imshow("Green", G)
    cv2.waitKey(0)
    
    cv2.imshow("Blue", B)
    cv2.waitKey(0)
    
    cv2.imshow("Grayscale", grayscale)
    cv2.waitKey(0)
    # Close the window
    cv2.destroyAllWindows()

def seperate_color_planes_image(img):    
    
    # Below Follows the Code to Check Sequence of RGB Planes.
    row,col,plane = img.shape

    # here image is of class 'uint8', the range of values  
    # that each colour component can have is [0 - 255]

    # create a zero matrix of order same as
    # original image matrix order of same dimension
    temp = np.zeros((row,col,plane),np.uint8)

    # store blue plane contents or data of image matrix
    # to the corresponding plane(blue) of temp matrix
    temp[:,:,0] = img[:,:,0]

    # displaying the Blue plane image
    cv2.imshow('Blue plane image',temp)
    cv2.waitKey(0)

    # again take a zero matrix of image matrix shape
    temp = np.zeros((row,col,plane),np.uint8)

    # store green plane contents or data of image matrix
    # to the corresponding plane(green) of temp matrix
    temp[:,:,1] = img[:,:,1]

    # displaying the Green plane image
    cv2.imshow('Green plane image',temp)
    cv2.waitKey(0)

    # again take a zero matrix of image matrix shape
    temp = np.zeros((row,col,plane),np.uint8)

    # store red plane contents or data of image matrix
    # to the corresponding plane(red) of temp matrix
    temp[:,:,2] = img[:,:,2]

    # displaying the Red plane image
    cv2.imshow('Red plane image',temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_to_rgb_grayscale_bw(img):
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert to Black and White
    # Now, to convert our image to black and white, we will apply the thresholding operation.
    (thresh, blackAndWhiteImage) = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Display the image
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    cv2.imshow('RGB', img_rgb)
    cv2.waitKey(0)
    cv2.imshow('Grayscale', img_gray)
    cv2.waitKey(0)
    cv2.imshow('Black and White', blackAndWhiteImage)
    cv2.waitKey(0)
    # Close the window
    cv2.destroyAllWindows()

def scale_down_up_image(img):
    #scale down
    print('Original Dimensions : ',img.shape)
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Scale Down Dimensions : ',resized.shape)
    
    cv2.imshow("Scale Down image", resized)
    cv2.waitKey(0)

    #scale up
    print('Original Dimensions : ',img.shape)
    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Scale Up Dimensions : ',resized.shape)
    
    cv2.imshow("Scale Up image", resized)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def read_video(mode,video_file_name=None):
    # Read the video file
    if mode=="video":
        if video_file_name is None:
            print("Please give a video file name")
            return
        cap = cv2.VideoCapture(video_file_name)
    else:
        cap=cv2.VideoCapture(0)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame',frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()



if __name__ == '__main__':
    img = cv2.imread(r'images/Elephant.jpg')  # Read an image using the relative path

    message="""
    1.Read image and display it
    2.Check the colur space of image
    3.Seperate the color planes of image
    4.Convert image to RGB, Grayscale and Black and White
    5.Scale down and up image
    6.Read video from local path or from camera
    7.Exit
    """
    choice=int(input(message))
    while True:
        if choice==1:
            read_and_display_image(img)
            choice=int(input(message))
        elif choice==2:
            check_colour_space_image(img)
            choice=int(input(message))
        elif choice==3:
            seperate_color_planes_image(img)
            choice=int(input(message))
        elif choice==4:
            img_to_rgb_grayscale_bw(img)
            choice=int(input(message))
        elif choice==5:
            scale_down_up_image(img)
            choice=int(input(message))
        elif choice==6:
            mode=input("Enter video mode(video/camera)")
            read_video(mode,r'videos/elephant.mp4' if mode=="video" else None)
            choice=int(input(message))
        elif choice==7:
            break
