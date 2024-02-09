import cv2
import numpy as np
import pyautogui
import keyboard

full_print = None


def take_screenshot():
    # Take a screenshot
    screenshot = pyautogui.screenshot()
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
    # TODO: Placeholder values. Replace with actual values
    full_print_location = (width // 2, height // 2)
    full_print_size = (width // 3, height // 3)
    partial_print_locations = [(width * 0.9, height * 0.1), (width * 0.9, height * 0.2), (width * 0.9, height * 0.3),
                               (width * 0.9, height * 0.4), (width * 0.9, height * 0.5), (width * 0.9, height * 0.6)]
    partial_print_size = (width // 10, height // 10)

    # Extract the full print
    full_print = cv2.getRectSubPix(screenshot, full_print_size, full_print_location)

    # Extract the partial prints
    partial_prints = [cv2.getRectSubPix(screenshot, partial_print_size, loc) for loc in partial_print_locations]

    for partial_print in partial_prints:
        # Perform template matching
        res = cv2.matchTemplate(full_print, partial_print, cv2.TM_CCOEFF_NORMED)
        # Set the threshold
        # TODO: Placeholder value. Replace with actual value
        threshold = 0.8
        # Get the locations of the matches
        loc = np.where(res >= threshold)
        # Draw rectangles around the matches
        w, h = partial_print.shape[::-1]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(full_print, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    # Display the image with the matches highlighted
    cv2.imshow('Matches', full_print)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Set the hotkey to take a screenshot (in this case, 'ctrl + shift + a')
keyboard.add_hotkey('ctrl + shift + a', process_image)
keyboard.wait()
