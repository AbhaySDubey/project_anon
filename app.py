import gradio as gr
import os
import cv2
import numpy as np
import asyncio
from utils import detect_faces_frame, apply_blur, load_caffe_models
from ultralight import UltraLightDetector
import tempfile
import json

# Create output directories
os.makedirs("output/image", exist_ok=True)
os.makedirs("output/video", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Initialize detector once
detector = UltraLightDetector()

# Age and gender options for filters
AGE_OPTIONS = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
GENDER_OPTIONS = ['Male', 'Female']

# Operation options
OPERATION_OPTIONS = {
    "Gaussian Blur": 0,
    "Black Patch": 1,
    "Pixelation": 2
}

def convert_for_json(obj):
    """Convert NumPy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj

def process_image(image, operation_name, age_filters=[], gender_filters=[], selected_face_indices=[]):
    """Process an image with face blurring"""
    # Convert from PIL to cv2 format
    if image is None:
        return None, "Please upload an image"
    
    # Convert from RGB (gradio) to BGR (OpenCV)
    if isinstance(image, str):  # If it's a path
        image_cv = cv2.imread(image)
    else:  # If it's a numpy array
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Get operation code
    operation = OPERATION_OPTIONS.get(operation_name, 0)
    
    # Detect faces
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    predictions = loop.run_until_complete(detect_faces_frame(detector=detector, frame=image_cv))
    loop.close()
    
    # Create a temporary copy for drawing face boxes
    image_with_boxes = image_cv.copy()
    
    face_thumbnails = [] 
    # Draw boxes around all detected faces with indices
    for i, pred in enumerate(predictions):
        box = np.array(pred['box'])
        x1, y1, x2, y2 = box.astype(int)
        # Draw box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 1)
        face_img = image_cv[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        caption = f"Face #{i} | {pred['gender']} | {pred['age']}"
        face_thumbnails.append((face_rgb, caption))
        # Draw index
        # cv2.putText(image_with_boxes, f"#{i}: {pred['gender']}, {pred['age']}", 
        #            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Convert to RGB for display
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Create filters dictionary
    filters = {
        "gender": gender_filters,
        "age": age_filters
    }
    
    # Create selected_faces list based on indices
    selected_faces = []
    if selected_face_indices:
        indices = [int(idx.strip()) for idx in selected_face_indices.split(",") if idx.strip().isdigit()]
        for i in indices:
            if i < len(predictions):
                selected_faces.append({"box": predictions[i]["box"]})
    
    # Apply blur
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    processed_image = loop.run_until_complete(
        apply_blur(
            detected_faces=predictions,
            frame=image_cv.copy(),
            filters=filters,
            selected_faces=selected_faces,
            operation=operation
        )
    )
    loop.close()
    
    # Convert back to RGB for Gradio
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Save results as JSON
    results_data = {
        "faces_detected": len(predictions),
        "predictions": convert_for_json(predictions),
        "operation": operation_name,
        "filters": {
            "gender": gender_filters,
            "age": age_filters
        },
        "selected_faces": [int(idx.strip()) for idx in selected_face_indices.split(",") if idx.strip().isdigit()] if selected_face_indices else []
    }
    
    return [image_with_boxes_rgb, processed_image_rgb, json.dumps(results_data, indent=2), face_thumbnails]

def process_video(video_path, operation_name, age_filters=[], gender_filters=[], progress=gr.Progress()):
    """Process a video with face blurring"""
    if video_path is None:
        return None, "Please upload a video"
    
    # Get operation code
    operation = OPERATION_OPTIONS.get(operation_name, 0)
    
    # Create a temporary file for the output
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video file"
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame skipping (process every nth frame for speed)
    frame_skip = max(1, round(fps / 15))  # Process at most 15 fps
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create filters dictionary
    filters = {
        "gender": gender_filters,
        "age": age_filters
    }
    
    # Process frames
    frame_count = 0
    face_count = 0
    
    # Process limited frames to prevent timeout (Gradio has a 60s limit by default)
    max_frames_to_process = min(300, total_frames)  # Limit to 300 frames
    
    for _ in progress.tqdm(range(max_frames_to_process)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame (for efficiency)
        if frame_count % frame_skip == 0:
            # Detect faces
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            predictions = loop.run_until_complete(detect_faces_frame(detector=detector, frame=frame))
            loop.close()
            
            face_count += len(predictions)
            
            # Apply blur
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            processed_frame = loop.run_until_complete(
                apply_blur(
                    detected_faces=predictions,
                    frame=frame,
                    filters=filters,
                    operation=operation
                )
            )
            loop.close()
            
            # Write processed frame
            out.write(processed_frame)
        else:
            # Write original frame for skipped frames
            out.write(frame)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Summary message
    summary = f"Processed {frame_count} frames, detected {face_count} faces"
    if frame_count < total_frames:
        summary += f" (limited to first {frame_count} frames out of {total_frames})"
    
    return output_path, summary

# Create Gradio interface
with gr.Blocks(title="Face Privacy Protection Tool") as demo:
    gr.Markdown("# Face Privacy Protection Tool")
    gr.Markdown("Upload an image or video to detect faces and apply privacy filters")
    
    with gr.Tabs():
        with gr.TabItem("Image Processing"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Upload Image", type="pil")
                    operation_dropdown = gr.Dropdown(
                        choices=list(OPERATION_OPTIONS.keys()),
                        value="Gaussian Blur",
                        label="Blur Operation"
                    )
                    
                    with gr.Accordion("Advanced Filtering", open=False):
                        age_filter = gr.CheckboxGroup(
                            choices=AGE_OPTIONS,
                            label="Filter by Age (select to blur)"
                        )
                        gender_filter = gr.CheckboxGroup(
                            choices=GENDER_OPTIONS,
                            label="Filter by Gender (select to blur)"
                        )
                        selected_faces = gr.Textbox(
                            label="Select Specific Faces to Blur (comma-separated indices, e.g., 0,1,3)",
                            placeholder="Enter face indices separated by commas"
                        )
                    
                    image_button = gr.Button("Process Image")
                
                with gr.Column():
                    output_tabs = gr.Tabs()
                    with output_tabs:
                        with gr.TabItem("Face Detection"):
                            image_with_boxes = gr.Image(label="Detected Faces")
                        
                        with gr.TabItem("Processed Image"):
                            image_output = gr.Image(label="Processed Image")
                        
                        with gr.TabItem("JSON Results"):
                            json_output = gr.JSON(label="Detection Results")
                        
                        with gr.TabItem("Detected Faces (Metadata)"):
                            face_gallery = gr.Gallery(
                                label="Detected Faces",
                                show_label=True,
                                columns=4,
                                height="auto",
                                object_fit="contain"
                            )
                            
            
            image_button.click(
                process_image,
                inputs=[image_input, operation_dropdown, age_filter, gender_filter, selected_faces],
                outputs=[image_with_boxes, image_output, json_output, face_gallery]
            )
        
        with gr.TabItem("Video Processing"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    video_operation = gr.Dropdown(
                        choices=list(OPERATION_OPTIONS.keys()),
                        value="Gaussian Blur",
                        label="Blur Operation"
                    )
                    
                    with gr.Accordion("Advanced Filtering", open=False):
                        video_age_filter = gr.CheckboxGroup(
                            choices=AGE_OPTIONS,
                            label="Filter by Age (select to blur)"
                        )
                        video_gender_filter = gr.CheckboxGroup(
                            choices=GENDER_OPTIONS,
                            label="Filter by Gender (select to blur)"
                        )
                    
                    video_button = gr.Button("Process Video")
                
                with gr.Column():
                    video_output = gr.Video(label="Processed Video")
                    video_summary = gr.Textbox(label="Processing Summary")
            
            video_button.click(
                process_video,
                inputs=[video_input, video_operation, video_age_filter, video_gender_filter],
                outputs=[video_output, video_summary]
            )
    
    gr.Markdown("""
    ## How to Use
    
    1. **Upload** an image or video using the respective tab
    2. **Choose** your preferred blur operation:
       - **Gaussian Blur**: Blurs facial features while maintaining face shape
       - **Black Patch**: Completely covers faces with black rectangles
       - **Pixelation**: Creates a mosaic effect over faces
    3. **Advanced Filtering**:
       - Filter by age group (select which age groups to blur)
       - Filter by gender (select which genders to blur)
       - For images, you can select specific face indices to blur
    4. **Process** the media and view the results
    
    Note: Video processing may take some time depending on the file size.
    """)

if __name__ == "__main__":
    demo.launch()