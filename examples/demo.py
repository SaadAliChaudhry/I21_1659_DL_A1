"""
Example script demonstrating how to use the facial expression recognition system
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.face_models import MultiTaskFaceModel, EmotionClassifier, ValenceArousalRegressor
from data.dataset import get_transforms
try:
    from utils.landmarks import FacialLandmarkDetector
except ImportError as e:
    print(f"Warning: Could not import FacialLandmarkDetector: {e}")
    FacialLandmarkDetector = None


def create_sample_data():
    """Create sample data for demonstration"""
    # Create a dummy face image (224x224 RGB)
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sample_image = Image.fromarray(sample_image)
    
    # Sample landmarks (68 points, 136 values)
    sample_landmarks = np.random.randn(136).astype(np.float32)
    
    # Sample labels
    sample_emotion = 3  # happiness
    sample_valence = 0.7
    sample_arousal = 0.5
    
    return sample_image, sample_landmarks, sample_emotion, sample_valence, sample_arousal


def demo_emotion_classifier():
    """Demonstrate emotion classification model"""
    print("=== Emotion Classification Demo ===")
    
    # Create model
    model = EmotionClassifier(
        num_classes=7,
        backbone='resnet18',  # Use smaller model for demo
        pretrained=True,
        dropout_rate=0.3
    )
    
    model.eval()
    
    # Create sample data
    sample_image, _, _, _, _ = create_sample_data()
    
    # Preprocess image
    transform = get_transforms(is_training=False)
    input_tensor = transform(sample_image).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Print results
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    print(f"Predicted emotion: {emotions[predicted_class]}")
    print(f"Confidence: {probabilities[0][predicted_class].item():.4f}")
    
    print("All probabilities:")
    for i, emotion in enumerate(emotions):
        print(f"  {emotion}: {probabilities[0][i].item():.4f}")
    print()


def demo_valence_arousal_regressor():
    """Demonstrate valence-arousal regression model"""
    print("=== Valence-Arousal Regression Demo ===")
    
    # Create model
    model = ValenceArousalRegressor(
        backbone='resnet18',  # Use smaller model for demo
        pretrained=True,
        dropout_rate=0.3
    )
    
    model.eval()
    
    # Create sample data
    sample_image, _, _, _, _ = create_sample_data()
    
    # Preprocess image
    transform = get_transforms(is_training=False)
    input_tensor = transform(sample_image).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        valence, arousal = model(input_tensor)
    
    # Print results
    print(f"Predicted valence: {valence.item():.4f} (range: -1 to 1)")
    print(f"Predicted arousal: {arousal.item():.4f} (range: -1 to 1)")
    print()


def demo_multitask_model():
    """Demonstrate multi-task model"""
    print("=== Multi-Task Model Demo ===")
    
    # Create model
    model = MultiTaskFaceModel(
        num_emotion_classes=7,
        backbone='resnet18',  # Use smaller model for demo
        pretrained=True,
        dropout_rate=0.3
    )
    
    model.eval()
    
    # Create sample data
    sample_image, _, _, _, _ = create_sample_data()
    
    # Preprocess image
    transform = get_transforms(is_training=False)
    input_tensor = transform(sample_image).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Process emotion results
    emotion_probs = torch.softmax(outputs['emotion_logits'], dim=1)
    predicted_emotion = torch.argmax(emotion_probs, dim=1).item()
    
    # Print results
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    print(f"Predicted emotion: {emotions[predicted_emotion]}")
    print(f"Emotion confidence: {emotion_probs[0][predicted_emotion].item():.4f}")
    print(f"Predicted valence: {outputs['valence'].item():.4f}")
    print(f"Predicted arousal: {outputs['arousal'].item():.4f}")
    print()


def demo_landmark_detection():
    """Demonstrate facial landmark detection"""
    print("=== Facial Landmark Detection Demo ===")
    
    if FacialLandmarkDetector is None:
        print("Landmark detection not available - dlib not installed")
        print("To enable landmark detection, install dlib: pip install dlib")
        print()
        return
    
    try:
        # Initialize landmark detector
        detector = FacialLandmarkDetector()
        
        # Create sample image (using numpy array format)
        sample_image_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Detect landmarks (this will likely fail with random data, but shows the API)
        landmarks = detector.detect_landmarks(sample_image_np)
        
        if landmarks is not None:
            print(f"Detected {len(landmarks)} facial landmarks")
            print(f"First few landmarks: {landmarks[:5]}")
            
            # Extract features
            from utils.landmarks import extract_landmark_features
            features = extract_landmark_features(landmarks)
            print(f"Extracted {len(features)} geometric features")
        else:
            print("No face detected in the sample image (expected with random data)")
            
    except Exception as e:
        print(f"Landmark detection demo failed: {e}")
        print("This is expected if dlib models are not installed")
    
    print()


def demo_data_loading():
    """Demonstrate data loading functionality"""
    print("=== Data Loading Demo ===")
    
    from data.dataset import FaceExpressionDataset, get_transforms
    
    # Show transform operations
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    print("Training transforms:")
    print(train_transform)
    print("\nValidation transforms:")
    print(val_transform)
    
    # Create sample image and apply transforms
    sample_image, _, _, _, _ = create_sample_data()
    
    train_transformed = train_transform(sample_image)
    val_transformed = val_transform(sample_image)
    
    print(f"\nOriginal image size: {sample_image.size}")
    print(f"Transformed tensor shape: {train_transformed.shape}")
    print(f"Tensor value range: [{train_transformed.min():.3f}, {train_transformed.max():.3f}]")
    print()


def demo_model_summary():
    """Show model architecture summary"""
    print("=== Model Architecture Summary ===")
    
    models_to_test = [
        ("Emotion Classifier", EmotionClassifier(num_classes=7, backbone='resnet18')),
        ("Valence-Arousal Regressor", ValenceArousalRegressor(backbone='resnet18')),
        ("Multi-Task Model", MultiTaskFaceModel(num_emotion_classes=7, backbone='resnet18'))
    ]
    
    for name, model in models_to_test:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
        print()


def main():
    """Run all demos"""
    print("Facial Expression Recognition System - Demo\n")
    
    # Run all demonstrations
    demo_model_summary()
    demo_emotion_classifier()
    demo_valence_arousal_regressor()
    demo_multitask_model()
    demo_landmark_detection()
    demo_data_loading()
    
    print("Demo completed successfully!")
    print("\nTo train models, use: python train.py --data_dir <path> --train_annotations <path> --val_annotations <path>")
    print("To run inference, use: python inference.py --model_path <path> --input <path>")


if __name__ == '__main__':
    main()