import cv2
from ultralight import UltraLightDetector
from ultralight.utils import draw_faces
import time
import asyncio

def load_caffe_models():
    age_net = cv2.dnn.readNet(
        r"D:\hackathons\we_hack_2025\project_anon\opencv_models\age_net.caffemodel",
        r"D:\hackathons\we_hack_2025\project_anon\opencv_models\age_deploy.prototxt"
    )
    gender_net = cv2.dnn.readNet(
        r"D:\hackathons\we_hack_2025\project_anon\opencv_models\gender_net.caffemodel",
        r"D:\hackathons\we_hack_2025\project_anon\opencv_models\gender_deploy.prototxt"
    )
    
    return age_net, gender_net

async def detect_faces_frame(detector, frame=None):
    if frame is None:
        print("No frames obtained!")
        return list()
    else:
        # define age and gender list
        age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
        gender_list = ['Male', 'Female']
        
        # detect the faces and load age & gender detection models
        boxes, scores = detector.detect_one(frame)
        age_model, gender_model = load_caffe_models()
        
        results = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            padding = int((x2 - x1) * 0.2)
            height, width = frame.shape[:2]
            
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(width, x2 + padding)
            y2_pad = min(height, y2 + padding)
            
            face_img = frame[y1_pad:y2_pad, x1_pad:x2_pad] 
                
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            gender_model.setInput(blob)
            gender_preds = gender_model.forward()
            gender = gender_list[gender_preds[0].argmax()]
            gender_confidence = float(gender_preds[0].max())
            
            age_model.setInput(blob)
            age_preds = age_model.forward()
            age = age_list[age_preds[0].argmax()]
            age_confidence = float(age_preds[0].max())
            
            results.append({
                'index': i,
                'box': box.tolist(),
                'score': float(scores[i]),
                'gender': gender,
                'gender_confidence': gender_confidence,
                'age': age,
                'age_confidence': age_confidence
            })
            
        return results

    
def apply_black_patch(frame, face_img, color=(0,0,0)):
    """Apply black patch to face region"""
    if frame is None or not face_img:
        return frame
    x1, y1, x2, y2 = map(int, face_img["box"])
    frame[y1:y2, x1:x2] = color
    return frame

def apply_gaussian_blur(frame, face_img, blur_factor=25):
    """Apply strong Gaussian blur to face region"""
    if frame is None or not face_img:
        return frame
    x1, y1, x2, y2 = map(int, face_img['box'])
    face_region = frame[y1:y2, x1:x2]
    
    # Ensure blur_factor is odd (requirement for GaussianBlur)
    if blur_factor % 2 == 0:
        blur_factor += 1
        
    # Apply blur
    blurred = cv2.GaussianBlur(face_region, (blur_factor, blur_factor), 0)
    frame[y1:y2, x1:x2] = blurred

    return frame

def apply_pixelation(frame, face_img, blocks=8):
    """Apply pixelation effect to face region"""
    if frame is None or not face_img:
        return frame
    
    x1, y1, x2, y2 = map(int, face_img['box'])
    face_region = frame[y1:y2, x1:x2]
    
    height, width = face_region.shape[:2]
    
    # Create heavy pixelation (small number of blocks = larger pixels)
    temp = cv2.resize(face_region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = pixelated
    return frame

async def apply_blur(
    detected_faces, # Can not be None, mandatory argument
    frame=None, 
    filters=None,
    selected_faces=[],
    operation=0, # default Gaussian Blur(High Kernel, blur factor 30); Other values are: 1 -> Black Patch(Simple Occlusion), 2 -> Pixelation(Heavy Mosaicing)
    output_path_root=None
):
    # suppose i'm processing the video, how can i play it while processing it? also, post processing how do i play it? i'd be uing flask as the backend
    """
    if filters is None:
        # blur all faces
    else:
        if filters["age"] is None or len(filters["age"]) == 0:
            # blur faces based on gender
        elif filters["gender"] is None or len(filters["gender"]) == 0:
            # blur faces based on age
        else:
            # blur faces based on both age and gender
    
    # Implement only if you have time
    if selected_faces is not None:
        # blur selected_faces
    """

    if filters is None:
        # blur all faces
        for face_img in detected_faces:
            selected_faces.append({"box": face_img["box"]})
            
    else:
        if len(filters["gender"]) != 0:
            for face_img in detected_faces:
                if face_img["gender"] in filters["gender"]:
                    selected_faces.append({"box": face_img["box"]})
                
        if len(filters["age"]) != 0:
            for face_img in detected_faces:
                if face_img["age"] in filters["age"]:
                    selected_faces.append({"box": face_img["box"]})
    
    if len(selected_faces) != 0:
        for face_img in selected_faces:
            if operation == 0:
                frame = apply_gaussian_blur(frame, face_img=face_img)
            elif operation == 1:
                frame = apply_black_patch(frame, face_img=face_img)
            elif operation == 2:
                frame = apply_pixelation(frame, face_img=face_img)
    
    return frame


async def process_input(
    image_path=None,
    video_path=None,
    output_path_root=None,
    filters=None,
    operation=0,
    selected_faces=[]
):
    detector = UltraLightDetector()
    if image_path is not None:
        output_path = f"{output_path_root}/image/output_image.jpeg"
        frame = cv2.imread(image_path)
        predictions = await detect_faces_frame(frame=frame, detector=detector)
        frame = await apply_blur(
            predictions,
            frame=frame,
            filters=filters,
            selected_faces=selected_faces,
            operation=operation,
            output_path_root=output_path_root
        )
        cv2.imwrite(output_path, frame)
        print("Output saved successfully!")
        cv2.imshow("Result Image", frame)
        cv2.waitKey(0)
    
    elif video_path is not None:
        output_path = f"{output_path_root}/video/output_video.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return [], []
        
        org_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, round(org_fps / 30))
        
        frame_count = 0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, org_fps, (width, height))
        all_predictions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                predictions = await detect_faces_frame(frame=frame, detector=detector)
                all_predictions.append(predictions)
                
                out.write(frame)
                     
            frame_count += 1
        cap.release()
        out.release()
        print(f"Video processing complete. Output saved to {output_path}")
    
    else:
        print("Error: No input provided")
    
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    image_path = r"D:\hackathons\we_hack_2025\project_anon\trials\teenage_girls.png"
    output_path_root = r"D:\hackathons\we_hack_2025\project_anon\output"
    asyncio.run(process_input(image_path=image_path, output_path_root=output_path_root, operation=2))