import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
import os
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(
    page_title="YOLO Weapon Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --- Custom Premium UI CSS ---
st.markdown("""
<style>
    /* Global Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Gradient Headers */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Premium Buttons with Micro-Animations */
    .stButton > button {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.5);
        color: white;
    }
    
    .stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Interactive File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        transition: all 0.3s;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #00C9FF;
        background: rgba(0, 201, 255, 0.05);
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.2);
    }
    
    /* Elegant Info & Alert Boxes */
    [data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        color: #fff !important;
    }
    
    /* Card-like image containers */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Header ---
st.title("🛡️ YOLO Weapon & Person Detection Dashboard")
st.markdown("A real-time computer vision system built to detect people and weapons using YOLO11.")

# --- Model Loading & Caching ---
MODEL_PATH = "runs/detect/weapon_detector-5/weights/best.pt"

@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_yolo_model(MODEL_PATH)

if model is None:
    st.error(f"Could not load the model weights from `{MODEL_PATH}`. Please check if the weights exist.")
    st.stop()

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")

# 1. Sensitivity Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold (Sensitivity)",
    min_value=0.10,
    max_value=1.00,
    value=0.30,
    step=0.05,
    help="Higher threshold reduces false alarms. Lower threshold detects more objects but may cause false alarms."
)

# 2. Input Source Selector
input_source = st.sidebar.radio(
    "Select Input Source",
    (
        "🖼️ Upload Image",
        "📸 Live Webcam (Photo)",
        "📹 Live CCTV / Camera Stream",
        "🎥 Upload Video"
    )
)

# Show model info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Active:** YOLO11s")
st.sidebar.markdown(f"**Classes:** {', '.join(model.names.values())}")

# --- Core App Logic ---

# ----------------- 🖼️ UPLOAD IMAGE -----------------
if input_source == "🖼️ Upload Image":
    st.header("🖼️ Image Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Detections")
            with st.spinner("Processing image..."):
                # Run YOLO prediction
                results = model.predict(source=image, conf=conf_threshold)
                
                # Get annotated image
                annotated_img = results[0].plot()
                
                # Display annotated image
                st.image(annotated_img, channels="BGR", use_column_width=True)
                
                # Print detection summaries
                st.markdown("### Detected Objects:")
                if len(results[0].boxes) == 0:
                    st.info("No objects detected.")
                else:
                    for box in results[0].boxes:
                        label = model.names[int(box.cls[0])]
                        confidence = float(box.conf[0])
                        st.success(f"• **{label}** (Confidence: {confidence:.2%})")

# ----------------- 📸 LIVE WEBCAM (PHOTO) -----------------
elif input_source == "📸 Live Webcam (Photo)":
    st.header("📸 Webcam Photo Capture")
    st.markdown("Snap a photo using your web browser's camera to run weapon detection.")
    
    # Streamlit built-in camera input
    img_file = st.camera_input("Take a photo")
    
    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        
        with st.spinner("Analyzing photo..."):
            results = model.predict(source=image, conf=conf_threshold)
            annotated_img = results[0].plot()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detection Result")
                st.image(annotated_img, channels="BGR", use_column_width=True)
            with col2:
                st.subheader("Detection Details")
                if len(results[0].boxes) == 0:
                    st.info("No objects detected.")
                else:
                    for box in results[0].boxes:
                        label = model.names[int(box.cls[0])]
                        confidence = float(box.conf[0])
                        st.success(f"• **{label}** (Confidence: {confidence:.2%})")

# ----------------- 📹 LIVE CCTV / CAMERA STREAM -----------------
elif input_source == "📹 Live CCTV / Camera Stream":
    st.header("📹 Live Stream Connection (CCTV / RTSP / USB Webcam)")
    st.markdown("Connect to an external USB camera, IP camera, or RTSP surveillance stream.")
    
    # Option to select webcam index or input RTSP string
    stream_type = st.radio("Select connection type:", ("Default/External Webcam (USB)", "IP Camera / RTSP Link"))
    
    source = 0
    if stream_type == "Default/External Webcam (USB)":
        cam_index = st.selectbox("Select Camera Index:", (0, 1, 2, 3), index=0, 
                                 help="0 is usually your built-in camera. 1 or 2 is an external USB camera.")
        source = cam_index
    else:
        rtsp_url = st.text_input("Enter RTSP Stream URL:", placeholder="rtsp://username:password@ip_address:port/h264")
        source = rtsp_url
        
    start_stream = st.button("Start Live Stream")
    
    if start_stream:
        # Create UI elements
        stop_stream = st.button("Stop Live Stream")
        frame_placeholder = st.empty()
        status_text = st.empty()
        
        status_text.text("Connecting to camera stream...")
        
        # Open video capture
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            st.error("Failed to connect to the video stream. Please verify your camera connection or URL.")
        else:
            status_text.text("Stream active. Running detection...")
            
            # Loop until stop button is pressed or stream ends
            while cap.isOpened() and not stop_stream:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame or stream disconnected.")
                    break
                
                # Run YOLO prediction (verbose=False avoids cluttering console logs)
                results = model.predict(source=frame, conf=conf_threshold, verbose=False)
                
                # Draw boxes
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB for Streamlit compatibility
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display current frame
                frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                
            cap.release()
            status_text.text("Stream stopped.")
            frame_placeholder.empty()

# ----------------- 🎥 UPLOAD VIDEO -----------------
elif input_source == "🎥 Upload Video":
    st.header("🎥 Video File Detection")
    uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # We need to save the uploaded video to a temporary file locally so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()
        
        st.subheader("Processing Video Feed")
        
        # Stream control buttons
        start_processing = st.button("Start Processing Video")
        
        if start_processing:
            stop_processing = st.button("Stop Processing")
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.get(cv2.CAP_PROP_FRAME_COUNT))) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 100
            frame_count = 0
            
            while cap.isOpened() and not stop_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict
                results = model.predict(source=frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                # Convert
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update UI
                video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                
                # Progress bar
                frame_count += 1
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                
            cap.release()
            st.success("Finished processing video.")
            
            # Clean up the temp file
            try:
                os.unlink(tfile.name)
            except Exception:
                pass
