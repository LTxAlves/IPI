import cv2 as cv
import numpy as np


def imageScaling():

    # Load the image
    img = cv.imread("images/notre-dame.pgm", cv.IMREAD_GRAYSCALE)

    # Create a window to display the image
    cv.namedWindow("Original image", cv.WINDOW_AUTOSIZE)
    # Display the image
    cv.imshow("Original image", img)
    # Wait for a key to be pressed
    cv.waitKey(0)
    # Destroy the window
    cv.destroyWindow("Original image")

def homomorphicFiltering():
    print("Not yet implemented")

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
