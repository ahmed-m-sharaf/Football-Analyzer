import os
import subprocess
import imageio_ffmpeg
from dotenv import load_dotenv
load_dotenv()
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

debug_mode = st.sidebar.checkbox(
    "Enable Pipeline Debug Mode",
    value=False,
    help="Enable optional debug diagnostics for camera homography estimation, reprojection errors, and player tracking speeds in console output."
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
        # Resolve folders for input/output videos
        input_dir = os.path.join("data", "videos", "input")
        output_dir = os.path.join("data", "videos", "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        input_path = os.path.join(input_dir, uploaded_file.name)
        output_path = os.path.join(output_dir, uploaded_file.name)

        # Save uploaded file only if it doesn't exist or size is different
        if not os.path.exists(input_path) or os.path.getsize(input_path) != uploaded_file.size:
            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

        # Show file metadata
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
        }
        st.write(file_details)

        # Display uploaded video with play capability
        st.video(input_path)

        # Check if already processed
        cached_processed = os.path.exists(output_path) and not st.session_state.get("force_reprocess", False)

        if cached_processed:
            st.success("✅ Previously processed version found! Displaying cached results.")
            if st.button("🔄 Re-process Video", use_container_width=True):
                st.session_state["force_reprocess"] = True
                st.rerun()
        else:
            if model_ready:
                process_btn = st.button("🚀 Process Video", use_container_width=True)
            else:
                st.warning("⚠️ Please upload the real YOLO model weights in the sidebar to enable video processing.")

with col2:
    st.subheader("📊 Output & Processing Status")

    if uploaded_file is not None:
        if cached_processed:
            # Display processed video with play capability
            st.video(output_path)

            # Download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Annotated Video",
                    data=file,
                    file_name=uploaded_file.name,
                    mime="video/mp4",
                    use_container_width=True,
                )
        elif 'process_btn' in locals() and process_btn:
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
                temp_output_path = output_path + ".temp.mp4"

                # Run the integrated pipeline
                run_pipeline(
                    model_path=yolo_model,
                    source_video=input_path,
                    target_video=temp_output_path,
                    conf=conf_threshold,
                    device=device_opt,
                    roboflow_key=roboflow_api_key if roboflow_api_key else None,
                    progress_callback=streamlit_progress_callback,
                    debug=debug_mode,
                )

                status_text.text("Converting video to web-playable H.264 format...")
                
                # Convert the output video to H.264 codec using imageio_ffmpeg static binary
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", temp_output_path,
                    "-vcodec", "libx264",
                    "-f", "mp4",
                    output_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Clean up temp file
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

                # Reset reprocess flag and rerun to trigger cached display state
                st.session_state["force_reprocess"] = False
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)
        else:
            st.info("Upload a video and click 'Process Video' to start analysis.")
    else:
        st.info("Upload a video and click 'Process Video' to start analysis.")
