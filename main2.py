import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.colab import files
from ipywidgets import interact, widgets, Output

# Global variables
uploaded_filename = None
img = None
output = Output()

# Upload image
def upload_image(change):
    global uploaded_filename
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        uploaded_filename = filename
        with output:
            output.clear_output(wait=True)
            display_image()

# Display image
def display_image():
    global img, uploaded_filename
    img = mpimg.imread(uploaded_filename)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Filter functions
def apply_filter(filter_type):
    global img
    filtered_img = img.copy()  # Make a copy of the original image

    # Apply selected filter
    if filter_type == 'Prewitt Edge Detector':
        filtered_img = cv2.filter2D(filtered_img, -1, kernel_prewitt)
    elif filter_type == 'Sobel Edge Detector':
        filtered_img = cv2.Sobel(filtered_img, cv2.CV_64F, 1, 0, ksize=5)
    elif filter_type == 'Erosion':
        kernel = np.ones((5,5), np.uint8)
        filtered_img = cv2.erode(filtered_img, kernel, iterations=1)
    elif filter_type == 'Dilation':
        kernel = np.ones((5,5), np.uint8)
        filtered_img = cv2.dilate(filtered_img, kernel, iterations=1)
    elif filter_type == 'Open':
        kernel = np.ones((5,5), np.uint8)
        filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel)

    # Update the displayed image
    with output:
        output.clear_output(wait=True)
        plt.imshow(filtered_img, cmap='gray')
        plt.axis('off')
        plt.show()




# Define filter kernels
kernel_HPF = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

kernel_LPF = np.ones((5, 5), np.float32) / 25  # 5x5 average filter

kernel_robert = np.array([[1, 0],
                          [0, -1]])

kernel_prewitt = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])

# Region Split and Merge function
def region_split_and_merge(img):
    def split(img):
        height, width = img.shape
        if height <= 1 or width <= 1:
            return [img]
        else:
            half_height = height // 2
            half_width = width // 2
            return split(img[:half_height, :half_width]) + \
                   split(img[:half_height, half_width:]) + \
                   split(img[half_height:, :half_width]) + \
                   split(img[half_height:, half_width:])

    def merge(segments):
        result = []
        for segment in segments:
            if isinstance(segment, list):
                result.extend(merge(segment))
            else:
                result.append(segment)
        return result

    segments = split(img)
    segments_to_merge = []
    for segment in segments:
        avg = np.mean(segment)
        if np.max(segment) - np.min(segment) > 50:
            segments_to_merge.append(segment)
        else:
            segments_to_merge.append(np.ones_like(segment) * avg)
    return merge(segments_to_merge)

# Callback function for button click
def on_button_click(btn):
    filter_type = btn.description
    apply_filter(filter_type)

# Create upload button
upload_button = widgets.Button(description="Upload Image")
upload_button.on_click(upload_image)

# Create buttons for filters
filter_buttons = [widgets.Button(description=filter_type) for filter_type in ['HPF', 'LPF', 'Mean Filter', 'Median Filter', 'Robert Edge Detector', 'Prewitt Edge Detector', 'Sobel Edge Detector', 'Erosion', 'Dilation', 'Open', 'Close', 'Hough Transform for Circle', 'Segmentation using Region Split and Merge', 'Segmentation using Thresholding']]
for button in filter_buttons:
    button.on_click(on_button_click)

# Display widgets
display(widgets.VBox([upload_button, widgets.HBox(filter_buttons), output]))