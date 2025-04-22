import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn.preprocessing import LabelEncoder
import os

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize face detection using OpenCV's pre-trained model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize the face recognition model
        self.recognition_model = self._build_recognition_model()
        self.label_encoder = LabelEncoder()
        
    def _build_recognition_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')  # Number of classes to recognize
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def preprocess_face(self, image, face):
        x, y, w, h = face
        face_roi = image[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (96, 96))
        face_roi = face_roi / 255.0  # Normalize pixel values
        return face_roi
    
    def train(self, faces_dir):
        X = []
        y = []
        
        # Load training data
        for person_name in os.listdir(faces_dir):
            person_dir = os.path.join(faces_dir, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    image = cv2.imread(image_path)
                    if image is not None:
                        faces = self.detect_faces(image)
                        for face in faces:
                            face_roi = self.preprocess_face(image, face)
                            X.append(face_roi)
                            y.append(person_name)
        
        # Convert labels to numerical format
        y = self.label_encoder.fit_transform(y)
        X = np.array(X)
        
        # Train the model
        self.recognition_model.fit(X, y, epochs=10, validation_split=0.2)
    
    def recognize_face(self, image, face):
        try:
            # Check if the model has been trained
            if not hasattr(self.label_encoder, 'classes_'):
                raise ValueError("Face recognition model has not been trained yet. Please train the model first.")
            
            face_roi = self.preprocess_face(image, face)
            face_roi = np.expand_dims(face_roi, axis=0)
            prediction = self.recognition_model.predict(face_roi)
            person_id = np.argmax(prediction)
            person_name = self.label_encoder.inverse_transform([person_id])[0]
            confidence = prediction[0][person_id]
            return person_name, confidence
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return "Unknown", 0.0
    
    def process_image(self, image):
        faces = self.detect_faces(image)
        results = []
        
        for face in faces:
            x, y, w, h = face
            person_name, confidence = self.recognize_face(image, face)
            results.append({
                'bbox': face,
                'name': person_name,
                'confidence': confidence
            })
            
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Put name and confidence
            label = f'{person_name} ({confidence:.2f})'
            cv2.putText(image, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image, results
    
    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, _ = self.process_image(frame)
            cv2.imshow('Face Recognition', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    # Initialize the system
    face_system = FaceRecognitionSystem()
    
    # Create a sample training directory structure
    sample_dir = 'training_faces'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"Created training directory: {sample_dir}")
        print("Please add face images to this directory with the following structure:")
        print("training_faces/")
        print("├── person1/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("└── person2/")
        print("    ├── image1.jpg")
        print("    └── image2.jpg")
        print("\nAfter adding images, uncomment the training line below.")
    
    # Train the system (uncomment after adding images to the training directory)
    # face_system.train('training_faces')
    
    # Run real-time recognition
    face_system.run_webcam()