import streamlit as st
import cv2
import numpy as np
from PIL import Image

def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def apply_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_smoothing(image):
    return cv2.GaussianBlur(image, (7, 7), 0)

def scale_image(image, fx, fy):
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def translate_image(image, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def shear_image(image, shear_x, shear_y):
    (h, w) = image.shape[:2]
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    return cv2.warpAffine(image, M, (w, h))

def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_threshold(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_color_filter(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)

def convert_color_space(image, code):
    return cv2.cvtColor(image, code)

def median_filter(image):
    return cv2.medianBlur(image, 5)

def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

st.title("Image Processing App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    operation = st.selectbox("Choose an Operation", [
        "Image Enhancement", "Geometric Transformations",
        "Color & Intensity Transformations", "Image Filtering & Noise Reduction"])
    
    if operation == "Image Enhancement":
        enhancement = st.selectbox("Select Enhancement", ["Brightness", "Contrast", "Sharpening", "Smoothing", "Histogram Equalization"])
        if enhancement == "Brightness":
            value = st.slider("Brightness Value", -100, 100, 0)
            processed_image = adjust_brightness(image, value)
        elif enhancement == "Contrast":
            alpha = st.slider("Contrast Value", 0.5, 3.0, 1.0)
            processed_image = adjust_contrast(image, alpha)
        elif enhancement == "Sharpening":
            processed_image = apply_sharpening(image)
        elif enhancement == "Smoothing":
            processed_image = apply_smoothing(image)
        elif enhancement == "Histogram Equalization":
            processed_image = apply_histogram_equalization(image)

    elif operation == "Geometric Transformations":
        transformation = st.selectbox("Select Transformation", ["Scaling", "Rotation", "Translation", "Shearing"])
        if transformation == "Scaling":
            fx = st.slider("Scale X", 0.1, 3.0, 1.0)
            fy = st.slider("Scale Y", 0.1, 3.0, 1.0)
            processed_image = scale_image(image, fx, fy)
        elif transformation == "Rotation":
            angle = st.slider("Rotation Angle", -180, 180, 0)
            processed_image = rotate_image(image, angle)
        elif transformation == "Translation":
            tx = st.slider("Translate X", -100, 100, 0)
            ty = st.slider("Translate Y", -100, 100, 0)
            processed_image = translate_image(image, tx, ty)
        elif transformation == "Shearing":
            shear_x = st.slider("Shear X", -1.0, 1.0, 0.0)
            shear_y = st.slider("Shear Y", -1.0, 1.0, 0.0)
            processed_image = shear_image(image, shear_x, shear_y)

    elif operation == "Color & Intensity Transformations":
        color_transformation = st.selectbox("Select Transformation", ["Grayscale", "Thresholding", "Color Filtering", "Color Space Conversion"])
        if color_transformation == "Grayscale":
            processed_image = convert_grayscale(image)
        elif color_transformation == "Thresholding":
            threshold = st.slider("Threshold Value", 0, 255, 127)
            processed_image = apply_threshold(image, threshold)
        elif color_transformation == "Color Filtering":
            lower = np.array([0, 120, 70])
            upper = np.array([10, 255, 255])
            processed_image = apply_color_filter(image, lower, upper)
        elif color_transformation == "Color Space Conversion":
            processed_image = convert_color_space(image, cv2.COLOR_BGR2HSV)
    
    elif operation == "Image Filtering & Noise Reduction":
        filter_type = st.selectbox("Select Filter", ["Median", "Gaussian", "Bilateral"])
        if filter_type == "Median":
            processed_image = median_filter(image)
        elif filter_type == "Gaussian":
            processed_image = gaussian_filter(image)
        elif filter_type == "Bilateral":
            processed_image = bilateral_filter(image)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if 'processed_image' in locals():
            st.image(processed_image, caption="Processed Image", use_container_width=True)

# Footer
st.markdown("""
    ---
    👨‍💻 **Developed by Sharanya Sharma 229301571** | 📨 Section: C 
""")
