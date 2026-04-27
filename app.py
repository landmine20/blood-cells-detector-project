import streamlit as st
import os
import sys
import subprocess

# --- STREAMLIT CLOUD OPENCV FIX ---
try:
    import cv2
    cv2.imread # test if it's functional
except (ImportError, AttributeError):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2
# ----------------------------------

import numpy as np
from PIL import Image
from collections import Counter

# Set up page layout and styling
st.set_page_config(page_title="Blood Cell Detector", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    /* Modern blue theme */
    .stApp {
        background-color: #f0f4f8;
    }
    h1, h2, h3 {
        color: #1a365d;
    }
    .stButton>button {
        background-color: #3182ce;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2b6cb0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #3182ce;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CONF = 0.15
IOU = 0.7
IMGSZ = 640

WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}
# Colors in RGB format for Streamlit/PIL
COLOR_RBC = (255, 0, 0)      # Red
COLOR_PLATELETS = (0, 200, 0)# Green
COLOR_WBC = (0, 80, 255)     # Blue

def color_for(name: str) -> tuple[int, int, int]:
    if name == "RBC":
        return COLOR_RBC
    if name == "Platelets":
        return COLOR_PLATELETS
    if name in WBC_SUBTYPES:
        return COLOR_WBC
    return (200, 200, 200)

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    model_path = os.path.join(os.path.dirname(__file__), "blood_detector_model.pt")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please make sure 'blood_detector_model.pt' is in the root directory.")
        st.stop()
    return YOLO(model_path)

def annotate(img, boxes_xyxy, classes, names) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft, bt = 0.4, 1, 1
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, classes):
        name = names[int(c)]
        col = color_for(name)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, bt)
        (tw, th), _ = cv2.getTextSize(name, font, fs, ft)
        ly = y1 - 2
        if ly - th - 2 < 0:
            ly = y1 + th + 4
        cv2.rectangle(img, (x1, ly - th - 2), (x1 + tw + 2, ly + 1), col, -1)
        # Using RGB colors, white text remains (255, 255, 255)
        cv2.putText(img, name, (x1 + 1, ly - 1), font, fs, (255, 255, 255), ft, cv2.LINE_AA)

def process_image(model, image_pil):
    # Convert PIL Image to numpy array (RGB)
    img_rgb = np.array(image_pil)
    
    # Run prediction
    results = model.predict(
        source=img_rgb,
        conf=CONF, iou=IOU, imgsz=IMGSZ,
        save=False, verbose=False,
    )
    
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    
    # We annotate directly on the RGB image because Streamlit expects RGB
    annotated_img = img_rgb.copy()
    annotate(annotated_img, boxes, classes, r.names)
    
    counts = Counter(r.names[int(c)] for c in classes)
    
    return annotated_img, counts

def main():
    st.title("🔬 Blood Cell Detection")
    st.markdown("Upload a peripheral blood smear image or use your webcam to automatically detect and classify blood cells.")
    
    model = load_model()
    
    st.sidebar.header("Input Options")
    input_source = st.sidebar.radio("Choose input method:", ["Upload Image", "Webcam"])
    
    image_pil = None
    
    if input_source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
    else:
        camera_input = st.camera_input("Take a picture")
        if camera_input is not None:
            image_pil = Image.open(camera_input).convert("RGB")
            
    if image_pil is not None:
        st.markdown("### Analysis Results")
        
        with st.spinner("Analyzing image..."):
            annotated_img, counts = process_image(model, image_pil)
            
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(annotated_img, caption="Processed Image", use_container_width=True)
            
        with col2:
            st.markdown("### Detected Cells")
            if not counts:
                st.info("No cells detected.")
            else:
                total_cells = sum(counts.values())
                st.markdown(f'<div class="metric-card"><h4>Total Cells</h4><h2 style="margin:0; color:#3182ce;">{total_cells}</h2></div>', unsafe_allow_html=True)
                
                # Separate counts into categories
                rbc_count = counts.get("RBC", 0)
                plt_count = counts.get("Platelets", 0)
                
                wbc_counts = {k: v for k, v in counts.items() if k in WBC_SUBTYPES}
                wbc_total = sum(wbc_counts.values())
                
                st.markdown(f"**🔴 RBC:** {rbc_count}")
                st.markdown(f"**🟢 Platelets:** {plt_count}")
                st.markdown(f"**🔵 WBC (Total):** {wbc_total}")
                
                if wbc_total > 0:
                    st.markdown("#### WBC Differential")
                    for wbc_type, count in wbc_counts.items():
                        percentage = (count / wbc_total) * 100
                        st.markdown(f"- {wbc_type}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
