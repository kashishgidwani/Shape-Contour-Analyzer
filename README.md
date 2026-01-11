# Shape & Contour Analyzer

A minimal interactive dashboard using **Streamlit** and **OpenCV** to detect, classify, and measure shapes in images.

## 🚀 How to Run

1.  **Install Dependencies** (if not already installed):
    ```bash
    pip install streamlit opencv-python-headless numpy pandas pillow
    ```
2.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
3.  **Open in Browser**:
    The app will usually open automatically at `http://localhost:8501`.

## 🎛️ Features
### Dashboard
1.  **Upload Image**: Supports JPG, PNG.
2.  **Original vs Processed**: Compare the raw input with the detected contours.
3.  **Metrics**: View total object count and a data table with:
    *   **Shape Type** (Triangle, Square, Rectangle, Circle, Polygon)
    *   **Area (px)**
    *   **Perimeter (px)**

## 🧪 Test Images
The project includes generated test images:
*   `test_image_1.png`: Basic verification (Triangle, Square, Circle).
*   `test_image_2.png`: High contrast colorful shapes.
*   `test_image_3.png`: Triangles and irregular polygons.
*   `test_image_4.png`: Dark shapes on dark background (good for testing sensitivity).
*   `test_image_5.png`: Stars, L-shapes, and rotated rectangles.

## 🛠️ Tech Stack
*   **Python**: Core Logic
*   **Streamlit**: UI / Frontend
*   **OpenCV**: Image Processing
*   **Pandas**: Data Display
