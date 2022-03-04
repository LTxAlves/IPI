import cv2 as cv
import numpy as np

# function to spread m elements evenly in a sequence from 0 to n-1 (a.k.a. range(n))
# (source: https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n)
spread = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]

def imageScaling():

    # Load the image
    resizedImg = img = cv.imread("images/notre-dame.pgm", cv.IMREAD_GRAYSCALE)

    # Initialize the scale factor
    scaleFactor = 0

    # Read scale factor until it is valid (i.e. greater than 0)
    while scaleFactor <= 0:
        try:
            scaleFactor = float(input("Enter a scale factor: "))
        except ValueError:
            print("Invalid input. Please enter a number greater than 0.")

        if scaleFactor <= 0:
            print("Invalid input. Please enter a number greater than 0.")

    # Get original image dimensions
    originalWidth, originalHeight = img.shape[:2]

    if scaleFactor < 1:

        # Calculate how many pixels to remove from each side of the image
        widthToDelete, heigthToDelete = int((1 - scaleFactor) * originalWidth), int((1 - scaleFactor) * originalHeight)

        # Create list of indices to delete
        rowsToDelete = spread(widthToDelete, originalWidth)
        colsToDelete = spread(heigthToDelete, originalHeight)

        # Delete the pixels
        resizedImg = np.delete(resizedImg, rowsToDelete, axis=0)
        resizedImg = np.delete(resizedImg, colsToDelete, axis=1)
    elif scaleFactor > 1: 

        # Calculate new size of the image
        newWidth, newHeight = int(originalWidth * scaleFactor), int(originalHeight * scaleFactor)

        # Create array of zeros with the new size
        resizedImg = np.zeros(shape=(newWidth, newHeight), dtype=np.uint8)

        # Use nearest nighbor to achieve resized image
        for i in range(newWidth):
            for j in range(newHeight):
                resizedImg[i, j] = img[int(i / scaleFactor), int(j / scaleFactor)]

    originalImageWindowName = "Original image"
    resizedImageWindowName = "Resized image"

    # Create a window to display the image
    cv.namedWindow(originalImageWindowName, cv.WINDOW_NORMAL)
    #Force focus on new window
    cv.setWindowProperty(originalImageWindowName, cv.WND_PROP_TOPMOST, 1)
    # Display the image
    cv.imshow(originalImageWindowName, img)

    # Create a window to display the resized image
    cv.namedWindow(resizedImageWindowName, cv.WINDOW_NORMAL)
    #Force focus on new window
    cv.setWindowProperty(resizedImageWindowName, cv.WND_PROP_TOPMOST, 1)
    # Display the resized image
    cv.imshow(resizedImageWindowName, resizedImg)

    cv.resizeWindow(originalImageWindowName, img.shape[1], img.shape[0])
    cv.resizeWindow(resizedImageWindowName, resizedImg.shape[1], resizedImg.shape[0])

    print(f"Original image size: {img.shape[0]} x {img.shape[1]}")
    print(f"Resized image size: {resizedImg.shape[0]} x {resizedImg.shape[1]}")

    # Wait for a key to be pressed
    print("\nPress any key to go back to main menu.\n")
    cv.waitKey(0)

    # Destroy all windows
    try:
        cv.destroyWindow(originalImageWindowName)
    except cv.error:
        pass
    try:
        cv.destroyWindow(resizedImageWindowName)
    except cv.error:
        pass

def homomorphicFiltering():
    print("Not yet implemented")
    #https://cadubentzen.github.io/pdi-ufrn/unit2/homomorphic.html

def notchRejectFilter():
    print("Not yet implemented")

if __name__ == "__main__":

    # Run until the user presses the exit key
    while True:
        key = 0
        while key not in [1, 2, 3, 4]:
            
            # Display the menu
            print("Select the part of the project to run:")
            print("\t1 - Part 1: Image scaling")
            print("\t2 - Part 2: Homomorphic filtering")
            print("\t3 - Part 3: Notch-reject filter")
            print("\t4 - Exit")
            # Get the user's choice
            try:
                key = int(input("Enter your selection: "))
            except ValueError:
                print("Invalid input")
            else:
                if key not in [1, 2, 3, 4]:
                    print("Invalid selection")

        # Run the selected part of the project/exits
        if key == 1:
            imageScaling()
        elif key == 2:
            homomorphicFiltering()
        elif key == 3:
            notchRejectFilter()
        elif key == 4:
            break
