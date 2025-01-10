from ultralytics import YOLO
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

class EmotionDetectionPipeline:
    def __init__(self, yolo_model_path='yolov8n.pt'):
        # Initialize YOLOv8 model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Dictionary to store emotion colors for visualization
        self.emotion_colors = {
            'angry': (0, 0, 255),    # Red
            'disgust': (0, 255, 0),  # Green
            'fear': (128, 0, 128),   # Purple
            'happy': (255, 255, 0),  # Yellow
            'sad': (255, 0, 0),      # Blue
            'surprise': (0, 255, 255),# Cyan
            'neutral': (128, 128, 128)# Gray
        }

    def detect_and_analyze(self, frame):
        # Run YOLOv8 detection
        results = self.yolo_model(frame, classes=[0])  # class 0 is person in COCO
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Process each detection
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                # Get person coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Extract person region
                person_roi = frame[y1:y2, x1:x2]
                
                try:
                    # Analyze emotion using DeepFace
                    emotion_result = DeepFace.analyze(
                        person_roi, 
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if emotion_result:
                        # Get dominant emotion
                        emotion = emotion_result[0]['dominant_emotion']
                        
                        # Draw bounding box with emotion color
                        color = self.emotion_colors.get(emotion, (255, 255, 255))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add emotion label
                        label = f"{emotion}"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2
                        )
                except Exception as e:
                    print(f"Error analyzing emotion: {str(e)}")
                    continue
        
        return annotated_frame

    def process_video_stream(self, source=0):
        """Process video stream from camera or video file"""
        cap = cv2.VideoCapture(source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            annotated_frame = self.detect_and_analyze(frame)
            
            # Display result
            cv2.imshow('Emotion Detection', annotated_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = EmotionDetectionPipeline()
    pipeline.process_video_stream()