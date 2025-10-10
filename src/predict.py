import disable_warning_tensorflow

import numpy as np
import pickle
from pathlib import Path
from tensorflow import keras
from utils import features_extractor

MODEL_SAVE_PATH = Path('./models/baby_cry_model.keras')
LABEL_ENCODER_PATH = Path('./models/label_encoder.pkl')

def predict_baby_cry(audio_file_path, model_path=MODEL_SAVE_PATH, 
                     label_encoder_path=LABEL_ENCODER_PATH):
    """Classify a baby cry audio file"""
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return None, None, None
    
    # Load model and label encoder
    print("Loading model...")
    model = keras.models.load_model(model_path)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Extract features
    print(f"Extracting features from {audio_file_path}...")
    features = features_extractor(audio_file_path)
    
    if features is None:
        print("Failed to extract features")
        return None, None, None
    
    # Reshape and predict
    features = features.reshape(1, -1)
    predictions = model.predict(features, verbose=0)
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = predictions[0][predicted_class_index]
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Prediction Results for: {Path(audio_file_path).name}")
    print(f"{'='*50}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}\n")
    print("All class probabilities:")

    # Create a list of (class_name, probability) tuples
    class_probs = []
    for i, prob in enumerate(predictions[0]):
        class_name = label_encoder.inverse_transform([i])[0]
        class_probs.append((class_name, prob))

    # Sort by probability in descending order
    class_probs.sort(key=lambda x: x[1], reverse=True)

    # Display sorted probabilities
    for class_name, prob in class_probs:
        print(f"  {class_name}: {prob:.2%}")

    print(f"{'='*50}\n")
    
    return predicted_class, confidence, predictions[0]

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file_path>")
        print("Example: python predict.py ./test_audio/baby_cry.wav")
    else:
        audio_path = sys.argv[1]
        predict_baby_cry(audio_path)