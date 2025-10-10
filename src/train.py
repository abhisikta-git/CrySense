import disable_warning_tensorflow

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from utils import features_extractor

# training data path
BASE_PATH = Path('./data/train_data')

# if 'models' folder does not exist, create it
# if it exists, ignore the instruction and move on
MODELS = Path('./models/')
MODELS.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODELS / 'baby_cry_model.keras'
LABEL_ENCODER_PATH = MODELS / 'label_encoder.pkl'

def load_data_from_folders(base_path):
    """Load audio files from subfolders"""
    extracted_features = []
    
    if not base_path.exists():
        print(f"ERROR: Base path does not exist: {base_path} \n")
        return extracted_features
    
    categories = [d for d in base_path.iterdir() if d.is_dir()]
    print(f"Found {len(categories)} categories: {[c.name for c in categories]}")
    
    for category_folder in categories:
        category_name = category_folder.name
        audio_files = list(category_folder.glob('*.wav'))
        
        print(f"\nProcessing category: {category_name} ({len(audio_files)} files)")
        
        success_count = 0
        for audio_file in tqdm(audio_files, desc=category_name):
            features = features_extractor(audio_file)
            if features is not None:
                extracted_features.append([features, category_name])
                success_count += 1
        
        print(f"  Successfully extracted: {success_count}/{len(audio_files)} files")
    
    return extracted_features

def train_model(base_path):
    """Train the baby cry classification model"""
    print("Loading and extracting features...")
    extracted_features = load_data_from_folders(base_path)
    
    if len(extracted_features) == 0:
        print("No features extracted. Check your data path.")
        return None, None
    
    # Create DataFrame
    extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
    print(f"\nTotal samples: {len(extracted_features_df)}")
    print(f"Class distribution:\n{extracted_features_df['class'].value_counts()}")
    
    # Prepare data
    X = np.array(extracted_features_df['feature'].tolist())
    y = np.array(extracted_features_df['class'].tolist())
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = pd.get_dummies(y_encoded).values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    # Build model
    num_classes = y_onehot.shape[1]
    
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Save
    model.save(MODEL_SAVE_PATH)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")
    
    return model, label_encoder

if __name__ == "__main__":
    print("Starting model training...")
    model, label_encoder = train_model(BASE_PATH)
    print("\nTraining complete!")