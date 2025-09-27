"""
Example script for creating sample training data in the expected format
"""

import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2

def create_sample_dataset(output_dir: str = 'sample_data', num_samples: int = 100):
    """
    Create a sample dataset with random images and annotations for testing
    
    Args:
        output_dir: Directory to save the sample dataset
        num_samples: Number of sample images to create
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Emotion labels
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    # Create sample data
    samples = []
    
    for i in range(num_samples):
        # Generate random face-like image (224x224)
        # Create a simple face-like pattern
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add some random color variations
        image[:, :, 0] = np.random.randint(100, 200)  # Red channel
        image[:, :, 1] = np.random.randint(80, 180)   # Green channel
        image[:, :, 2] = np.random.randint(90, 190)   # Blue channel
        
        # Add some simple geometric shapes to simulate face features
        # Eyes
        cv2.circle(image, (70, 80), 10, (50, 50, 50), -1)
        cv2.circle(image, (150, 80), 10, (50, 50, 50), -1)
        
        # Nose
        cv2.circle(image, (110, 120), 8, (120, 100, 100), -1)
        
        # Mouth
        cv2.ellipse(image, (110, 160), (20, 10), 0, 0, 180, (80, 80, 80), -1)
        
        # Add some noise
        noise = np.random.normal(0, 25, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        image_filename = f'sample_{i:04d}.jpg'
        image_path = os.path.join(images_dir, image_filename)
        pil_image = Image.fromarray(image)
        pil_image.save(image_path, 'JPEG')
        
        # Generate random landmarks (68 points, 136 values)
        landmarks = np.random.uniform(0, 224, 136).astype(np.float32)
        landmarks_str = ','.join([f'{val:.2f}' for val in landmarks])
        
        # Generate random labels
        emotion = np.random.choice(emotions)
        valence = np.random.uniform(-1, 1)
        arousal = np.random.uniform(-1, 1)
        
        # Add to samples
        samples.append({
            'image_path': image_filename,
            'landmarks': landmarks_str,
            'emotion': emotion,
            'valence': valence,
            'arousal': arousal
        })
    
    # Create DataFrame and save CSV files
    df = pd.DataFrame(samples)
    
    # Split into train/validation (80/20)
    train_split = int(0.8 * len(df))
    train_df = df[:train_split]
    val_df = df[train_split:]
    
    # Save CSV files
    train_csv_path = os.path.join(output_dir, 'train_annotations.csv')
    val_csv_path = os.path.join(output_dir, 'val_annotations.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"Sample dataset created in '{output_dir}':")
    print(f"  - {len(train_df)} training samples")
    print(f"  - {len(val_df)} validation samples")
    print(f"  - Images saved in '{images_dir}'")
    print(f"  - Training annotations: '{train_csv_path}'")
    print(f"  - Validation annotations: '{val_csv_path}'")
    
    return output_dir, train_csv_path, val_csv_path


def create_sample_config():
    """Create a sample configuration file for testing"""
    config_content = """# Sample configuration for testing
dataset:
  image_size: [224, 224]
  num_landmarks: 68
  emotions: ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
  valence_range: [-1.0, 1.0]
  arousal_range: [-1.0, 1.0]

model:
  backbone: 'resnet18'  # Use smaller model for faster training
  num_emotion_classes: 7
  dropout_rate: 0.3
  pretrained: true

training:
  batch_size: 8  # Small batch size for testing
  learning_rate: 0.001
  num_epochs: 2  # Few epochs for testing
  weight_decay: 0.0001
  scheduler_step_size: 1
  scheduler_gamma: 0.9

data_augmentation:
  horizontal_flip: 0.5
  rotation_degrees: 15
  brightness: 0.2
  contrast: 0.2

paths:
  data_root: './sample_data'
  model_save_dir: './test_checkpoints'
  log_dir: './test_logs'
"""
    
    config_path = 'test_config.yaml'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Sample configuration saved to: {config_path}")
    return config_path


if __name__ == '__main__':
    print("Creating sample dataset for testing...")
    
    # Create sample dataset
    data_dir, train_csv, val_csv = create_sample_dataset()
    
    # Create sample config
    config_path = create_sample_config()
    
    print("\nSample dataset created successfully!")
    print("\nTo test training, run:")
    print(f"python train.py --config {config_path} --task multitask --data_dir {data_dir}/images --train_annotations {train_csv} --val_annotations {val_csv}")
    
    print("\nTo test inference, run:")
    print("python inference.py --model_path ./test_checkpoints/best_accuracy.pth --mode single --input ./sample_data/images/sample_0000.jpg")