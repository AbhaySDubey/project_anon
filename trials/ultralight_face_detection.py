import cv2
from ultralight import UltraLightDetector
from ultralight.utils import draw_faces
import time
import asyncio

async def detect_faces_frame(detector, frame=None):
    if frame is None:
        print("No frames obtained!")
        return list()
    else:
        boxes, scores = detector.detect_one(frame)
        return boxes, scores


async def detect_faces(image_path=None, video_path=None, save=False, output_path_root=None):
    detector = UltraLightDetector()
    if image_path is not None:
        output_path = f"{output_path_root}/image/output_image.jpeg"
        frame = cv2.imread(image_path)
        boxes, scores = await detect_faces_frame(frame=frame, detector=detector)
        if save ==True:
            draw_faces(frame, boxes=boxes, scores=scores)
            cv2.imwrite(output_path, frame)
        return list([boxes, scores])
    
    elif video_path is not None:
        output_path = f"{output_path_root}/video/output_video.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return [], []
        
        org_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, round(org_fps / 30))
        
        all_boxes = []
        all_scores = []
        frame_count = 0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, org_fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                boxes, scores = await detect_faces_frame(frame=frame, detector=detector)
                all_boxes.append(boxes)
                all_scores.append(scores)
                
                if save: 
                    draw_faces(frame, boxes=boxes, scores=scores)
                    out.write(frame)
                     
            frame_count += 1
        cap.release()
        out.release()
        print(f"Video processing complete. Output saved to {output_path}")
        return list([all_boxes, all_scores])
    
    else:
        print("Error: No input provided")
        return [], []
    

if __name__ == "__main__":
    image_path = r"D:\hackathons\we_hack_2025\project_anon\trials\crowd.jpeg"
    video_path = r"D:\hackathons\we_hack_2025\project_anon\trials\crowd_of_people_walking.mp4"
    output_path_root = r"D:\hackathons\we_hack_2025\project_anon\trials\output"
    # boxes, scores = asyncio.run(detect_faces(image_path=image_path, save=True, output_path_root=output_path_root)) # detect faces in image
    start_time = time.perf_counter()
    boxes, scores = asyncio.run(detect_faces(video_path=video_path, save=True, output_path_root=output_path_root)) # detect faces in video
    end_time = time.perf_counter()
    print()
    print(f"Time elapsed = {(end_time - start_time):.6f} seconds.")