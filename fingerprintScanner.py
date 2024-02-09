import cv2
import numpy as np
import pyautogui
import keyboard
from time import sleep

full_print = None


def take_screenshot():
    # Take a screenshot
    screenshot = pyautogui.screenshot()
    # screenshot = cv2.imread('./test.png')
    # Convert the screenshot to a numpy array and then to grayscale
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    # Return the screenshot
    return screenshot


def process_image():
    global full_print
    screenshot = take_screenshot()
    # Get the size of the screenshot
    height, width = screenshot.shape

    # Define the relative locations and sizes of the full print and the partial prints
    # On a 1920x1080 screen, the full print is located at (960, 160) and has a size of (384, 960)
    full_print_location = (width * 0.4979166666666667, height * 0.1388888888888889)
    full_print_size = (int(width // 5), int(height // 2.1))
    # On a 1920x1080 screen, the partial prints are located at:
    # (476, 272), (620, 272)
    # (476, 416), (620, 416)
    # (476, 560), (620, 560)
    # (476, 706), (620, 706)
    # and have a size of (118, 108)
    partial_print_locations = [(width * 0.24791666666666667, height * 0.2518518518518518), (width * 0.3229166666666667, height * 0.2518518518518518),
                               (width * 0.24791666666666667, height * 0.3851851851851852), (width * 0.3229166666666667, height * 0.3851851851851852),
                               (width * 0.24791666666666667, height * 0.5185185185185185), (width * 0.3229166666666667, height * 0.5185185185185185),
                               (width * 0.24791666666666667, height * 0.6518518518518519), (width * 0.3229166666666667, height * 0.6518518518518519),]
    partial_print_size = (int(width // 16.3), int(height // 9.2))

    # Calculate the center of the partial prints
    partial_print_centers = [(loc[0] + partial_print_size[0] // 2, loc[1] + partial_print_size[1] // 2) for loc in partial_print_locations]

    # Calculate the center of the full print
    full_print_center = (full_print_location[0] + full_print_size[0] // 2, full_print_location[1] + full_print_size[1] // 2)

    # Extract the full print
    full_print = cv2.getRectSubPix(screenshot, full_print_size, full_print_center)

    cv2.imshow('Full Print', full_print)

    # Extract the partial prints
    partial_prints = [cv2.getRectSubPix(screenshot, partial_print_size, loc) for loc in partial_print_centers]

    # Initialize the list of matching partial print indices
    matching_partial_print_indices = []

    # Create the SIFT detector
    sift = cv2.SIFT_create(edgeThreshold=5)

    # Initialize the list of matching partial print indices
    matching_partial_print_indices = []

    for i, partial_print in enumerate(partial_prints):
        # Compute the keypoints and descriptors for the partial print and the full print
        kp1, des1 = sift.detectAndCompute(partial_print, None)
        kp2, des2 = sift.detectAndCompute(full_print, None)

        # Create the BFMatcher
        bf = cv2.BFMatcher()

        # Match the descriptors
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.92 * n.distance:
                good.append([m])

        # If there are enough matches, add the index of the matching partial print to the list
        if len(good) > 25:
            matching_partial_print_indices.append(i)

    # Convert the grayscale image back to a colored image
    screenshot_colored = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)

    # Draw a rectangle around each matching partial print on the original screenshot
    for index in matching_partial_print_indices:
        # Get the location of the matching partial print
        loc = partial_print_centers[index]
        # Get the size of the partial prints
        w, h = partial_print_size
        # Draw the rectangle
        screenshot_colored = cv2.rectangle(screenshot_colored, (int(loc[0] - w / 2), int(loc[1] - h / 2)), (int(loc[0] + w / 2), int(loc[1] + h / 2)), (0, 255, 255), 2)

    # Display the image with the matches highlighted
    cv2.imshow('Matches', screenshot_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Set the hotkey to take a screenshot (in this case, 'ctrl + shift + a')
# keyboard.add_hotkey('ctrl + shift + a', process_image)
sleep(5)
process_image()
keyboard.wait()
