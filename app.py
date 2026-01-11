import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def get_centroid(contour):
    """Calculate the centroid of a contour."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def detect_shape(contour):
    """
    Classify the shape of a contour based on vertex count and geometry.
    Returns: shape_name (str)
    """
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

    # Classification Logic
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # Check Aspect Ratio for Square vs Rectangle
        shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
    else:
        # > 4 vertices: Check Circularity
        area = cv2.contourArea(contour)
        if area > 0:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            circularity = area / circle_area
            
            # If high circularity -> Circle, else Polygon
            if circularity > 0.82: # Using 0.82 as threshold for better robustness
                shape = "Circle"
            else:
                shape = "Polygon"
        else:
            shape = "Polygon"
            
    return shape

def process_image(image_np, blur_k, canny_min, canny_max):
    """
    Process the image: Grayscale -> Blur -> Canny -> Dilate/Erode -> Find Contours.
    Returns: dict with all processing steps and final results
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Optional: Enhance contrast (helps with dark shapes on dark background)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert grayscale to RGB for display (3 channels)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 2. Gaussian Blur
    # Ensure kernel size is odd
    if blur_k % 2 == 0: blur_k += 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
    
    # 3. Canny Edge Detection
    edged = cv2.Canny(blurred, canny_min, canny_max)
    edged_rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
    
    # 4. Dilate/Erode to close gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)
    dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    eroded_rgb = cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)
    
    # 5. Find Contours
    contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours only (outline)
    contours_image = image_np.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    
    results = []
    # Make a copy for drawing. PIL loads as RGB, so we draw in RGB colors.
    output_image = image_np.copy()
    
    obj_id = 1
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter small noise
        if area < 500:
            continue
            
        shape_name = detect_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Get Bounding Box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Draw Contours (Green) - (0, 255, 0) in RGB
        cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
        
        # Draw Bounding Box (Red) - (255, 0, 0) in RGB
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Put Shape Name, Area, and Perimeter text
        label = f"{shape_name}"
        area_text = f"Area: {int(area)}"
        peri_text = f"Peri: {int(perimeter)}"
        # Ensure text does not go off image
        t_y = y - 30 if y - 30 > 10 else y + 10
        cv2.putText(output_image, label, (x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 0), 2)
        cv2.putText(output_image, area_text, (x, t_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(output_image, peri_text, (x, t_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        results.append({
            "Object ID": obj_id,
            "Shape Type": shape_name,
            "Area (px)": area,
            "Perimeter (px)": perimeter
        })
        obj_id += 1
        
    return {
        "original": image_np,
        "grayscale": gray_rgb,
        "blurred": blurred_rgb,
        "edges": edged_rgb,
        "dilated": dilated_rgb,
        "eroded": eroded_rgb,
        "contours": contours_image,
        "final": output_image,
        "results": results
    }

def main():
    st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
    
    # 1. Sidebar Controls
    st.sidebar.title("Settings")
    blur_k = st.sidebar.slider("Gaussian Blur Kernel Size", 3, 15, 5, step=2)
    st.sidebar.subheader("Edge Detection Thresholds")
    canny_min = st.sidebar.slider("Min Threshold", 0, 255, 25)
    canny_max = st.sidebar.slider("Max Threshold", 0, 255, 100)
    
    st.title("Shape & Contour Analyzer")
    
    # 2. Image Input
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        # Process Image
        processed = process_image(image_np, blur_k, canny_min, canny_max)
        
        # Display Original Image
        st.subheader("Step 1: Original Image Detection")
        st.image(processed["original"], caption="Original Image - Input image loaded and detected")
        
        # Display Processing Pipeline
        st.write("---")
        st.subheader("Image Processing Pipeline")
        
        # Row 1: Grayscale and Blur
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed["grayscale"], caption="Step 2: Grayscale Conversion - Converted from RGB to grayscale")
        with col2:
            st.image(processed["blurred"], caption="Step 3: Gaussian Blur - Noise reduction applied")
        
        # Row 2: Edge Detection
        col3, col4 = st.columns(2)
        with col3:
            st.image(processed["edges"], caption="Step 4: Canny Edge Detection - Edges detected using Canny algorithm")
        with col4:
            st.image(processed["dilated"], caption="Step 5: Dilation - Gaps in edges are closed")
        
        # Row 3: Erosion and Contours
        col5, col6 = st.columns(2)
        with col5:
            st.image(processed["eroded"], caption="Step 6: Erosion - Refined edge boundaries")
        with col6:
            st.image(processed["contours"], caption="Step 7: Contour Detection - Outlines of shapes detected")
        
        # Final Result
        st.write("---")
        st.subheader("Final Result: Shape Detection with Area and Perimeter")
        st.image(processed["final"], caption="Step 8: Final Analysis - Shapes identified with area and perimeter measurements")
            
        # 6. Output Dashboard Metrics & Table
        st.write("---")
        st.subheader("Analysis Results")
        
        # Metrics
        st.metric("Total Objects Detected", len(processed["results"]))
        
        # Data Table
        if processed["results"]:
            df = pd.DataFrame(processed["results"])
            st.dataframe(df)
        else:
            st.info("No objects detected. Try adjusting the thresholds.")
            
    else:
        # Warning/Placeholder
        st.warning("Awaiting Image Upload... Please convert your image to JPG/PNG.")
        st.info("Upload an image to see the complete processing pipeline from detection to shape analysis.")

if __name__ == "__main__":
    main()
