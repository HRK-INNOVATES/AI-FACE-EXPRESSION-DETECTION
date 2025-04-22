import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from face_expression_detection import create_model, EMOTIONS

def load_fer2013():
    try:
        data = pd.read_csv('fer2013.csv')
    except FileNotFoundError:
        print("Error: fer2013.csv not found!")
        print("\nPlease follow these steps to set up the dataset:")
        print("1. Go to https://www.kaggle.com/datasets/deadskull7/fer2013")
        print("2. Download the fer2013.csv file")
        print("3. Place fer2013.csv in this directory")
        print("\nNote: You'll need a Kaggle account to download the dataset.")
        raise
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = face.astype('float32')
        face /= 255.0
        faces.append(face)
    
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = to_categorical(data['emotion'])
    
    return faces, emotions

def train_emotion_classifier():
    # Load dataset
    print("Loading dataset...")
    faces, emotions = load_fer2013()
    
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(faces, emotions,
                                                        test_size=0.2,
                                                        random_state=42)
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    
    # Training parameters
    batch_size = 64
    epochs = 50
    
    # Train the model
    print("Training model...")
    history = model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_test, y_test),
                       shuffle=True)
    
    # Save the model
    print("Saving model...")
    model.save('emotion_model.h5')
    
    # Evaluate the model
    scores = model.evaluate(x_test, y_test)
    print(f"Test loss: {scores[0]}")
    print(f"Test accuracy: {scores[1]}")

if __name__ == "__main__":
    train_emotion_classifier()