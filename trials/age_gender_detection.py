import cv2
import ultralight_face_detection
from keras.models import load_model
import numpy as np
import asyncio
import time
import json

def load_ssr_models():
    age_model = load_model('models/ssrnet_age.h5')
    gender_model = load_model('models/ssrnet_gender.h5')
    return age_model, gender_model
    
def load_caffe_models():
    age_net = cv2.dnn.readNet(
        r"D:\hackathons\we_hack_2025\project_anon\opencv_dnn_face_detection\age_net.caffemodel",
        r"D:\hackathons\we_hack_2025\project_anon\opencv_dnn_face_detection\age_deploy.prototxt"
    )
    gender_net = cv2.dnn.readNet(
        r"D:\hackathons\we_hack_2025\project_anon\opencv_dnn_face_detection\gender_net.caffemodel",
        r"D:\hackathons\we_hack_2025\project_anon\opencv_dnn_face_detection\gender_deploy.prototxt"
    )
    
    return age_net, gender_net

def predict_with_ssrnet(face_img, age_model, gender_model):
    face = cv2.resize(face_img, (64, 64))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    
    # Predict
    gender_pred = gender_model.predict(face)[0]
    gender = "Male" if gender_pred[0] > 0.5 else "Female"
    
    age_pred = age_model.predict(face)[0]
    age = int(age_pred * 100)  # Scale to years
    
    return {"gender": gender, "age": age}


age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
gender_list = ['Male', 'Female']


async def process_image(image_path, model=0):
    """ use model=1 for SSR-Net models
        use model=2 for OpenCV's Caffe models
    """
    boxes, scores = await ultralight_face_detection.detect_faces(image_path=image_path)
    image = cv2.imread(image_path)
    
    results = []
    
    if model == 1:
        age_model, gender_model = load_ssr_models()
        for box in boxes:
            height, width = image.shape[:2]
            x1, y1, x2, y2 = box.astype(int)
    
            face_img = image[y1:y2, x1:x2]
            prediction = predict_with_ssrnet(face_img, age_model, gender_model)   
            results.append({
                "box": box,
                "age": prediction["age"],
                "gender": prediction["gender"]
            })
        
    elif model == 2:
        age_model, gender_model = load_caffe_models()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            padding = int((x2 - x1) * 0.2)
            height, width = image.shape[:2]
            
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(width, x2 + padding)
            y2_pad = min(height, y2 + padding)
            
            face_img = image[y1_pad:y2_pad, x1_pad:x2_pad] 
                
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
            
    else:
        print("Error: No valid model chosen!")
            
    return results


def draw_on_image(predictions, image_path, output_path_root=None, save=False):
    output_image = cv2.imread(image_path)
    for i, pred in enumerate(predictions):
        box = np.array(pred['box'])
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, f"{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"For index {i}, \n{pred}")
        
    output_path = ""
    if save:
        output_path = f'{output_path_root}/image/output_image.jpeg'
        cv2.imwrite(output_path, output_image)
    
    cv2.imshow("Face Detection Result", output_image)
    cv2.waitKey(0)
    return {'results': predictions, 'output_image': output_path}
        
        
if __name__ == "__main__":
    image_path = r"D:\hackathons\we_hack_2025\project_anon\trials\teenage_girls.png"
    start_time = time.perf_counter()
    predictions = asyncio.run(process_image(image_path=image_path, model=2))
    end_time = time.perf_counter()
    output_path_root = r"D:\hackathons\we_hack_2025\project_anon\trials\output\age_gender_detection"
    print()
    print()
    print(f"Elapsed Time(process_image()): {end_time-start_time}:.6f seconds")
    
    start_time = time.perf_counter()
    output = draw_on_image(
        predictions=predictions,
        image_path=image_path,
        output_path_root=output_path_root,
        save=True
    )
    end_time = time.perf_counter()
    print()
    print()
    print(f"Elapsed Time (draw_on_image()): {end_time-start_time}:.6f seconds")
    
    with open(f"{output_path_root}/image/results.json", 'w') as f:
        json.dump({"input": image_path, "output": output}, f, indent=2)
        
    cv2.destroyAllWindows()