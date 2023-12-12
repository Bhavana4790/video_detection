import streamlit as st
import tempfile
import shutil
import os
import base64
from demo import *

def run_object_detection(input_video_path, frames):
    # Create a temporary directory to store the results
    temp_dir = tempfile.mkdtemp()
    
    # Run the object detection code
    count, result = main_func(input_video_path, frames)
    
    # Move the results to the temporary directory
    result_video_path = os.path.join(temp_dir, "output_video.avi")
    result_csv_path = os.path.join(temp_dir, "output_results.csv")
    
    # Optionally, you can copy the result files to the output directory
    shutil.copy("outputs/output.csv", result_csv_path)
    shutil.copy("outputs/out_temp_video.avi", result_video_path)
    
    return count, result_video_path, result_csv_path

def main():
    st.title("Object Detection Streamlit App")

    # Upload video file
    st.sidebar.header("Upload Video File")
    video_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    # Set the number of frames to process
    frames = st.sidebar.number_input("Number of Frames", value=100, min_value=1, step=1)

    # Display video
    if video_file is not None:
        st.header("Input Video Player")
        video_bytes = video_file.read()
        
        # Save the video file temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as temp_video_file:
            temp_video_file.write(video_bytes)

        st.video(temp_video_path)

        # Perform object detection when the user clicks the button
        if st.button("Perform Object Detection"):
            print("video file path",video_file)
            count, result_video_path, result_csv_path = run_object_detection(temp_video_path, frames)

            # Display output video
            st.header("Output Video Player")
            result_video_bytes = open(result_video_path, "rb").read()
            st.video(result_video_bytes)

            # Provide download links for the result files
            st.markdown(f"Download [output video](data:video/avi;base64,{base64.b64encode(result_video_bytes).decode()})")
            st.markdown(f"Download [CSV results](data:file/csv;base64,{base64.b64encode(open(result_csv_path, 'rb').read()).decode()})")

        if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
if __name__ == "__main__":
    main()
