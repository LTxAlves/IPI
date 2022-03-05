import cv2 as cv
import numpy as np
from datetime import datetime

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
    # Force focus on new window
    cv.setWindowProperty(originalImageWindowName, cv.WND_PROP_TOPMOST, 1)
    # Display the image
    cv.imshow(originalImageWindowName, img)

    # Create a window to display the resized image
    cv.namedWindow(resizedImageWindowName, cv.WINDOW_NORMAL)
    # Force focus on new window
    cv.setWindowProperty(resizedImageWindowName, cv.WND_PROP_TOPMOST, 1)
    # Display the resized image
    cv.imshow(resizedImageWindowName, resizedImg)

    cv.resizeWindow(originalImageWindowName, img.shape[1], img.shape[0])
    cv.resizeWindow(resizedImageWindowName, resizedImg.shape[1], resizedImg.shape[0])

    print(f"Original image size: {img.shape[0]} x {img.shape[1]}")
    print(f"Resized image size: {resizedImg.shape[0]} x {resizedImg.shape[1]}")

    # Save the resized image
    saveLocation = f"images/results/notre-dame-resized-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.pgm"
    cv.imwrite(saveLocation, resizedImg)
    print(f"Image saved to {saveLocation}")

    # Wait for a key to be pressed
    print("\nPress any key to go back to main menu.\n")
    cv.waitKey(0)

    # Destroy all windows
    try:
        cv.destroyAllWindows()
    except cv.error:
        pass

def homomorphicFiltering():

    # Load the image
    img = cv.imread("images/image2.jpg", cv.IMREAD_GRAYSCALE)
    
    # Take image dimensions
    height, width = img.shape[:2]

    # Parameters for homomorphic filter
    gamma_l = gamma_h = sigma = -1

    # Read gamma_l until it is valid (i.e. from 0 to 10)
    while gamma_l < 0 or gamma_l > 10:
        try:
            gamma_l = float(input("Enter value for gamma_l: "))
        except ValueError:
            print("Invalid input. Please enter a number from 0 to 10")

        if gamma_l < 0 or gamma_l > 10:
            print("Invalid input. Please enter a number from 0 to 10")

    # Read gamma_h until it is valid (i.e. from 0 to 10)
    while gamma_h < 0 or gamma_h > 10:
        try:
            gamma_h = float(input("Enter value for gamma_h: "))
        except ValueError:
            print("Invalid input. Please enter a number from 0 to 10")

        if gamma_h < 0 or gamma_h > 10:
            print("Invalid input. Please enter a number from 0 to 10")

    # Read sigma until it is valid (i.e. from 1 to 100)
    while sigma < 1 or sigma > 100:
        try:
            sigma = int(input("Enter value for sigma: "))
        except ValueError:
            print("Invalid input. Please enter a number from 1 to 100")

        if sigma < 1 or sigma > 100:
            print("Invalid input. Please enter a number from 1 to 100")

    # Take the natural log of image, do DFT and shift it
    img_log = np.log(np.float32(img) + 0.01)
    img_dft = np.fft.fft2(img_log)
    img_dft_shift = np.fft.fftshift(img_dft)

    # Get coordinates of each pixel
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Coordinates of the center of the image
    xCenter, yCenter = width // 2, height // 2

    distSquared = np.square(x - xCenter) + np.square(y - yCenter)

    c = 0.1

    G = (gamma_h - gamma_l) * (1 - np.exp(c * (- distSquared / (2 * sigma**2)))) + gamma_l

    # Invert the initial steps to get image back
    result_filtered = G * img_dft_shift
    result_mid = np.real(np.fft.ifft2(np.fft.ifftshift(result_filtered)))
    filteredImage = np.exp(result_mid)
    filteredImage = cv.normalize(filteredImage, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    originalImageWindowName = "Original image"
    filteredImageWindowName = "Filtered image"

    # Create a window to display the image
    cv.namedWindow(originalImageWindowName, cv.WINDOW_NORMAL)
    # Force focus on new window
    cv.setWindowProperty(originalImageWindowName, cv.WND_PROP_TOPMOST, 1)
    # Display the image
    cv.imshow(originalImageWindowName, img)

    # Create a window to display the homomomorphic filtered image
    cv.namedWindow(filteredImageWindowName, cv.WINDOW_NORMAL)
    # Force focus on new window
    cv.setWindowProperty(filteredImageWindowName, cv.WND_PROP_TOPMOST, 1)
    # Display the homomomorphic filtered image
    cv.imshow(filteredImageWindowName, filteredImage)

    cv.resizeWindow(originalImageWindowName, img.shape[1], img.shape[0])
    cv.resizeWindow(filteredImageWindowName, filteredImage.shape[1], filteredImage.shape[0])

    # Save the homomomorphic filtered image
    saveLocation = f"images/results/image2-homomorphic-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.jpg"
    cv.imwrite(saveLocation, filteredImage)
    print(f"Image saved to {saveLocation}")

    # Wait for a key to be pressed
    print("\nPress any key to go back to main menu.\n")
    cv.waitKey(0)

    # Destroy all windows
    try:
        cv.destroyAllWindows()
    except cv.error:
        pass

def notchRejectFilter():

    # Load the image
    img = cv.imread("images/moire.tif", cv.IMREAD_GRAYSCALE)
    
    # Take image dimensions
    height, width = img.shape[:2]

    # Coordinates of the center of the image
    # xCenter, yCenter = width // 2, height // 2

    # Take the natural log of image, do DFT and shift it
    img_log = np.log(np.float32(img) + 0.01)
    img_dft = np.fft.fft2(img_log)
    img_dft_shift = np.fft.fftshift(img_dft)

    # Center of the notch
    d0 = 50.0

    # List of points in magnitude spectrum to choose
    points = [[38.1, 30.2], [-42.5, 27.6], [80.2, 30.4], [-82.6, 28.1]]

    for u in range(height):
            for v in range(width):
                for d in range(len(points)):
                    u0 = points[d][0]
                    v0 = points[d][1]
                    d1 = pow(((u - u0) ** 2) + ((v - v0) ** 2), 0.5)
                    d2 = pow(((u + u0) ** 2) + ((v + v0) ** 2), 0.5)
                    img_dft_shift[u][v] *= (1.0 / (1 + pow((d0 * d0) / (d1 * d2), 4))) 
                    
    f_ishift = np.fft.ifftshift(img_dft_shift)
    filteredImage = np.real(np.fft.ifft2(f_ishift))
    filteredImage = cv.normalize(filteredImage, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    originalImageWindowName = "Original image"
    filteredImageWindowName = "Filtered image"

    # Create a window to display the image
    cv.namedWindow(originalImageWindowName, cv.WINDOW_NORMAL)
    # Force focus on new window
    cv.setWindowProperty(originalImageWindowName, cv.WND_PROP_TOPMOST, 1)
    # Display the image
    cv.imshow(originalImageWindowName, img)

    # Create a window to display the notch rejected filtered image
    cv.namedWindow(filteredImageWindowName, cv.WINDOW_NORMAL)
    # Force focus on new window
    cv.setWindowProperty(filteredImageWindowName, cv.WND_PROP_TOPMOST, 1)
    # Display the notch rejected filtered image
    cv.imshow(filteredImageWindowName, filteredImage)

    cv.resizeWindow(originalImageWindowName, img.shape[1], img.shape[0])
    cv.resizeWindow(filteredImageWindowName, filteredImage.shape[1], filteredImage.shape[0])

    # Save the homomomorphic filtered image
    saveLocation = f"images/results/moire-notch-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.tif"
    cv.imwrite(saveLocation, filteredImage)
    print(f"Image saved to {saveLocation}")

    # Wait for a key to be pressed
    print("\nPress any key to go back to main menu.\n")
    cv.waitKey(0)

    # Destroy all windows
    try:
        cv.destroyAllWindows()
    except cv.error:
        pass


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
