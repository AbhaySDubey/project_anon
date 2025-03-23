import gradio as gr
import cv2
import numpy as np
import asyncio
from utils import detect_faces_frame, apply_blur, load_caffe_models
from ultralight import UltraLightDetector
import ssl

# Initialize models
print("Loading models...")
age_net, gender_net = load_caffe_models()
detector = UltraLightDetector()

# Options
AGE_OPTIONS = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
GENDER_OPTIONS = ['Male', 'Female']
OPERATIONS = {"Gaussian Blur": 0, "Black Patch": 1, "Pixelation": 2}

def process_image(image, operation_name, age_filters=[], gender_filters=[]):
    if image is None:
        return None, "Please upload an image."

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    operation = OPERATIONS.get(operation_name, 0)

    # Detect faces
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    predictions = loop.run_until_complete(detect_faces_frame(detector=detector, frame=image_cv))
    loop.close()

    if not predictions:
        return image, "No faces detected."

    # Apply blur
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    processed_image = loop.run_until_complete(
        apply_blur(detected_faces=predictions, frame=image_cv.copy(), filters={"age": age_filters, "gender": gender_filters}, operation=operation)
    )
    loop.close()

    # Convert back to RGB
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    return processed_image_rgb, f"Processed {len(predictions)} face(s)."

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(choices=list(OPERATIONS.keys()), value="Gaussian Blur", label="Blur Operation"),
        gr.CheckboxGroup(choices=AGE_OPTIONS, label="Filter by Age (Optional)"),
        gr.CheckboxGroup(choices=GENDER_OPTIONS, label="Filter by Gender (Optional)")
    ],
    outputs=[
        gr.Image(label="Processed Image"),
        gr.Text(label="Processing Summary")
    ],
    title="Face Detection and Privacy Protection App",
    description="Upload an image to detect faces, apply blur, and optionally filter by age or gender."
)

if __name__ == "__main__":
    demo.launch()