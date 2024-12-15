import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
import time


model = YOLO("fire_detector.pt")


st.title("Fire Detection App by YOLOv11")
st.write("Upload a video to detect fire using an AI-based detection system.")


st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.35, step=0.05)


uploaded_file = st.file_uploader("Upload Video (MP4)", type=["mp4"])


output_directory = tempfile.mkdtemp()

if uploaded_file is not None:
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    
    file_extension = os.path.splitext(temp_path)[1].lower()
    supported_extensions = ['.mp4']

    if file_extension not in supported_extensions:
        st.error(f"Unsupported file format: {file_extension}. Please upload MP4 files.")
    else:
        
        st.video(temp_path)

       
        if st.button("Run Detection"):
            st.write("Running detection... This may take a moment.")

            try:
                
                cap = cv2.VideoCapture(temp_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                
                video_placeholder = st.empty()

               
                output_video_path = os.path.join(output_directory, "processed_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                   
                    results = model.predict(source=frame, conf=confidence_threshold, show=False)

                    
                    if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame  

                    
                    out.write(annotated_frame)

                  
                    video_placeholder.image(annotated_frame, channels="BGR", caption="Detection in Progress", use_column_width=True)

                   
                    time.sleep(1 / fps)

                cap.release()
                out.release()

               
                st.write("### Processed Video")
                with open(output_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.download_button(
                        label="Download Processed Video",
                        data=video_bytes,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"An error occurred during detection: {e}")
