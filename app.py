import os
import streamlit as st
import tempfile
from src.main import run_pipeline

# Page Configuration
st.set_page_config(
    page_title="Football Match Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown(
    """
    <style>
        .reportview-container {
            background: #111;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            color: #FF4B4B;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-title">🏟️ AI Football Match Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A high-performance machine learning pipeline for sports analytics. Track players, calculate speed/distance covered, and estimate team ball possession.</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🔧 Pipeline Settings")

yolo_model = st.sidebar.text_input(
    "YOLO Model Path",
    value="data/models/best.pt",
    help="Path to fine-tuned YOLO model weight file (.pt)",
)

# Check if selected model is a Git LFS pointer
is_lfs_pointer = False
if os.path.exists(yolo_model):
    try:
        with open(yolo_model, "r", errors="ignore") as f:
            first_line = f.readline()
            if first_line.startswith("version https://git-lfs"):
                is_lfs_pointer = True
    except Exception:
        pass

model_ready = True
temp_model_path = None

if is_lfs_pointer:
    st.sidebar.warning(f"⚠️ `{yolo_model}` is a Git LFS pointer (not downloaded). Please upload the actual weights file below:")
    uploaded_model = st.sidebar.file_uploader(
        "Upload YOLO Model weights (.pt)",
        type=["pt"],
    )
    if uploaded_model is not None:
        # Check if the uploaded file is also an LFS pointer
        content_preview = uploaded_model.read(100)
        uploaded_model.seek(0)
        if content_preview.startswith(b"version https://git-lfs"):
            st.sidebar.error("❌ The file you uploaded is still a Git LFS pointer, not the actual weights file. Please upload the real 118MB weights file.")
            model_ready = False
        else:
            temp_model_dir = tempfile.gettempdir()
            temp_model_path = os.path.join(temp_model_dir, "temp_model.pt")
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.read())
            yolo_model = temp_model_path
    else:
        model_ready = False
        st.sidebar.error("Upload valid weights to proceed.")
elif not os.path.exists(yolo_model):
    st.sidebar.warning(f"⚠️ YOLO model not found at `{yolo_model}`. Please upload it below:")
    uploaded_model = st.sidebar.file_uploader(
        "Upload YOLO Model weights (.pt)",
        type=["pt"],
    )
    if uploaded_model is not None:
        # Check if the uploaded file is an LFS pointer
        content_preview = uploaded_model.read(100)
        uploaded_model.seek(0)
        if content_preview.startswith(b"version https://git-lfs"):
            st.sidebar.error("❌ The file you uploaded is a Git LFS pointer. Please upload the real 118MB weights file.")
            model_ready = False
        else:
            temp_model_dir = tempfile.gettempdir()
            temp_model_path = os.path.join(temp_model_dir, "temp_model.pt")
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.read())
            yolo_model = temp_model_path
    else:
        model_ready = False
        st.sidebar.error("Upload valid weights to proceed.")

conf_threshold = st.sidebar.slider(
    "YOLO Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05,
)

execution_device = st.sidebar.selectbox(
    "Execution Device",
    options=["auto", "cpu", "cuda"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Field Calibration")
roboflow_api_key = st.sidebar.text_input(
    "Roboflow API Key",
    type="password",
    help="Provide your Roboflow API key to enable homography camera estimation. If empty, the pipeline runs in pixel space.",
)

# Main Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Football Match Footage")
    uploaded_file = st.file_uploader(
        "Choose a video file...",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        # Show file metadata
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
        }
        st.write(file_details)
        
        if model_ready:
            process_btn = st.button("🚀 Process Video", use_container_width=True)
        else:
            st.warning("⚠️ Please upload the real YOLO model weights in the sidebar to enable video processing.")

with col2:
    st.subheader("📊 Output & Processing Status")
    
    if uploaded_file is not None and 'process_btn' in locals() and process_btn:
        # Create temp files for input/output to avoid naming conflicts
        temp_dir = tempfile.gettempdir()
        temp_input_path = os.path.join(temp_dir, "input_match.mp4")
        temp_output_path = os.path.join(temp_dir, "output_match_annotated.mp4")
        
        # Save uploaded file
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.read())
            
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Streamlit progress callback
        def streamlit_progress_callback(current_frame, total_frames):
            percent = float(current_frame / total_frames)
            progress_bar.progress(percent)
            status_text.text(f"Processed frame {current_frame} / {total_frames} ({percent * 100:.1f}%)")

        status_text.text("Initializing YOLO model and processing pipeline...")
        
        # Resolve device option
        device_opt = None if execution_device == "auto" else execution_device
        
        try:
            # Run the integrated pipeline
            run_pipeline(
                model_path=yolo_model,
                source_video=temp_input_path,
                target_video=temp_output_path,
                conf=conf_threshold,
                device=device_opt,
                roboflow_key=roboflow_api_key if roboflow_api_key else None,
                progress_callback=streamlit_progress_callback,
            )
            
            status_text.text("Finished processing!")
            st.success("Analysis complete!")
            
            # Download button
            with open(temp_output_path, "rb") as file:
                btn = st.download_button(
                    label="📥 Download Annotated Video",
                    data=file,
                    file_name="annotated_football_match.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
                
            # HTML5 player tip
            st.info("💡 Tip: If you download the video, you can view it on your local media player (VLC, QuickTime, etc.).")
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.exception(e)
    else:
        st.info("Upload a video and click 'Process Video' to start analysis.")
