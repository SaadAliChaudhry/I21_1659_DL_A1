"""
Inference script for facial expression recognition and affective computing
"""

import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import cv2
import os
import sys
import json
from typing import Dict, List, Tuple, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.face_models import EmotionClassifier, ValenceArousalRegressor, MultiTaskFaceModel
from data.dataset import get_transforms
from utils.landmarks import FacialLandmarkDetector


class FaceExpressionPredictor:
    """
    Predictor class for facial expression recognition and affective computing
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        task: str = 'multitask',
        device: str = 'cuda'
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            task: Type of model ('emotion', 'valence_arousal', 'multitask')
            device: Device for inference
        """
        self.task = task
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create and load model
        self.model = self._create_model()
        self._load_model(model_path)
        
        # Prepare transforms
        self.transform = get_transforms(is_training=False)
        
        # Emotion labels
        self.emotion_labels = self.config['dataset']['emotions']
        
        # Initialize landmark detector (optional)
        self.landmark_detector = None
        try:
            self.landmark_detector = FacialLandmarkDetector()
        except Exception as e:
            print(f"Warning: Could not initialize landmark detector: {e}")
    
    def _create_model(self):
        """Create model based on task and configuration"""
        model_config = self.config['model']
        
        if self.task == 'emotion':
            model = EmotionClassifier(
                num_classes=model_config['num_emotion_classes'],
                backbone=model_config['backbone'],
                pretrained=False,
                dropout_rate=model_config['dropout_rate']
            )
        elif self.task == 'valence_arousal':
            model = ValenceArousalRegressor(
                backbone=model_config['backbone'],
                pretrained=False,
                dropout_rate=model_config['dropout_rate']
            )
        else:  # multitask
            model = MultiTaskFaceModel(
                num_emotion_classes=model_config['num_emotion_classes'],
                backbone=model_config['backbone'],
                pretrained=False,
                dropout_rate=model_config['dropout_rate']
            )
        
        return model
    
    def _load_model(self, model_path: str):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
    
    def predict_image(self, image_path: str) -> Dict:
        """
        Predict emotions and/or valence-arousal for a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing predictions
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            if self.task == 'emotion':
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                results = {
                    'emotion': self.emotion_labels[predicted_class],
                    'confidence': confidence,
                    'probabilities': {
                        self.emotion_labels[i]: prob.item() 
                        for i, prob in enumerate(probabilities[0])
                    }
                }
                
            elif self.task == 'valence_arousal':
                valence, arousal = self.model(input_tensor)
                results = {
                    'valence': valence.item(),
                    'arousal': arousal.item()
                }
                
            else:  # multitask
                outputs = self.model(input_tensor)
                
                # Emotion prediction
                emotion_probs = torch.softmax(outputs['emotion_logits'], dim=1)
                predicted_emotion = torch.argmax(emotion_probs, dim=1).item()
                emotion_confidence = emotion_probs[0][predicted_emotion].item()
                
                results = {
                    'emotion': self.emotion_labels[predicted_emotion],
                    'emotion_confidence': emotion_confidence,
                    'emotion_probabilities': {
                        self.emotion_labels[i]: prob.item() 
                        for i, prob in enumerate(emotion_probs[0])
                    },
                    'valence': outputs['valence'].item(),
                    'arousal': outputs['arousal'].item()
                }
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict emotions and/or valence-arousal for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_webcam(self, save_results: bool = False, output_dir: str = './webcam_results'):
        """
        Real-time prediction from webcam feed
        
        Args:
            save_results: Whether to save predictions to file
            output_dir: Directory to save results
        """
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            results = []
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit, 's' to save current prediction")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Preprocess
                input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    if self.task == 'multitask':
                        outputs = self.model(input_tensor)
                        emotion_probs = torch.softmax(outputs['emotion_logits'], dim=1)
                        predicted_emotion = torch.argmax(emotion_probs, dim=1).item()
                        confidence = emotion_probs[0][predicted_emotion].item()
                        valence = outputs['valence'].item()
                        arousal = outputs['arousal'].item()
                        
                        # Display results on frame
                        text = f"Emotion: {self.emotion_labels[predicted_emotion]} ({confidence:.3f})"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        text = f"Valence: {valence:.3f}, Arousal: {arousal:.3f}"
                        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Facial Expression Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_results:
                    # Save current prediction
                    timestamp = frame_count
                    prediction = {
                        'frame': timestamp,
                        'emotion': self.emotion_labels[predicted_emotion],
                        'confidence': confidence,
                        'valence': valence,
                        'arousal': arousal
                    }
                    results.append(prediction)
                    print(f"Saved prediction for frame {timestamp}")
                
                frame_count += 1
                
            except Exception as e:
                print(f"Error in frame processing: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        
        if save_results and results:
            results_path = os.path.join(output_dir, 'webcam_predictions.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {results_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Facial Expression Recognition Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--task', type=str, choices=['emotion', 'valence_arousal', 'multitask'],
                        default='multitask', help='Model task type')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'webcam'],
                        default='single', help='Inference mode')
    parser.add_argument('--input', type=str,
                        help='Input image path (single mode) or directory (batch mode)')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Output file for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda/cpu)')
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    # Initialize predictor
    predictor = FaceExpressionPredictor(
        model_path=args.model_path,
        config_path=args.config,
        task=args.task,
        device=args.device
    )
    
    if args.mode == 'single':
        if not args.input:
            print("Error: --input is required for single image mode")
            return
        
        # Single image prediction
        result = predictor.predict_image(args.input)
        print("\nPrediction Results:")
        print(json.dumps(result, indent=2))
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")
        
    elif args.mode == 'batch':
        if not args.input:
            print("Error: --input directory is required for batch mode")
            return
        
        # Get all image files from directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images")
        
        # Batch prediction
        results = predictor.predict_batch(image_paths)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
        
    elif args.mode == 'webcam':
        # Webcam prediction
        output_dir = os.path.dirname(args.output) if args.output else './webcam_results'
        predictor.predict_webcam(save_results=True, output_dir=output_dir)


if __name__ == '__main__':
    main()