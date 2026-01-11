# Shape & Contour Analyzer

A minimal interactive dashboard using **Streamlit** and **OpenCV** to detect, classify, and measure shapes in images.

## How to Run

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

## Features
### Dashboard
1.  **Upload Image**: Supports JPG, PNG.
2.  **Original vs Processed**: Compare the raw input with the detected contours.
3.  **Metrics**: View total object count and a data table with:
    *   **Shape Type** (Triangle, Square, Rectangle, Circle, Polygon)
    *   **Area (px)**
    *   **Perimeter (px)**


## Tech Stack
*   **Python**: Core Logic
*   **Streamlit**: UI / Frontend
*   **OpenCV**: Image Processing
*   **Pandas**: Data Display
