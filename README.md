# Facial Expression Recognition and Affective Computing

This project implements a comprehensive deep learning system for **Facial Expression Recognition** and **Affective Computing**. The system can analyze facial expressions from images to predict both discrete emotions and continuous affective dimensions (valence and arousal).

## Features

- **Multi-modal Input Support**: Processes RGB face images (224×224) and optional 68-point facial landmarks
- **Multi-task Learning**: Simultaneous prediction of:
  - Categorical emotions (7 classes: anger, disgust, fear, happiness, sadness, surprise, neutral)
  - Continuous valence values (range: -1 to 1)
  - Continuous arousal values (range: -1 to 1)
- **Flexible Architecture**: Support for different CNN backbones (ResNet, EfficientNet)
- **Real-time Inference**: Webcam support for live emotion detection
- **Comprehensive Training Pipeline**: Complete training, validation, and inference scripts

## Dataset Requirements

The system expects data with the following structure:
- **Images**: Cropped RGB face images (224×224 pixels)
- **Landmarks**: 68 facial landmark points (optional, stored as 136 comma-separated values)
- **Emotion Labels**: Categorical emotion labels (anger, disgust, fear, happiness, sadness, surprise, neutral)
- **Valence/Arousal**: Continuous values in the range [-1, 1]

### Expected CSV Format

```csv
image_path,landmarks,emotion,valence,arousal
image001.jpg,"x1,y1,x2,y2,...,x68,y68",happiness,0.7,0.5
image002.jpg,"x1,y1,x2,y2,...,x68,y68",sadness,-0.6,-0.2
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd I21_1659_DL_A1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install dlib for landmark detection**:
   ```bash
   # For Ubuntu/Debian
   sudo apt-get install cmake
   pip install dlib
   
   # Download shape predictor model (optional)
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

## Project Structure

```
I21_1659_DL_A1/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── train.py                 # Main training script
├── inference.py             # Inference script
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── data/
│   │   └── dataset.py       # Dataset and data loading utilities
│   ├── models/
│   │   └── face_models.py   # Neural network models
│   ├── training/
│   │   └── trainer.py       # Training utilities and trainer class
│   └── utils/
│       └── landmarks.py     # Facial landmark utilities
├── examples/
│   └── demo.py              # Example usage and demonstrations
└── tests/                   # Test files (if any)
```

## Quick Start

### 1. Run Demo

To see the system in action with sample data:

```bash
python examples/demo.py
```

### 2. Training

Train a multi-task model (emotion + valence/arousal):

```bash
python train.py \
    --task multitask \
    --data_dir /path/to/your/images \
    --train_annotations /path/to/train.csv \
    --val_annotations /path/to/val.csv \
    --output_dir ./outputs
```

Train emotion classification only:

```bash
python train.py \
    --task emotion \
    --data_dir /path/to/your/images \
    --train_annotations /path/to/train.csv \
    --val_annotations /path/to/val.csv \
    --output_dir ./outputs
```

Train valence/arousal regression only:

```bash
python train.py \
    --task valence_arousal \
    --data_dir /path/to/your/images \
    --train_annotations /path/to/train.csv \
    --val_annotations /path/to/val.csv \
    --output_dir ./outputs
```

### 3. Inference

**Single Image Prediction**:
```bash
python inference.py \
    --model_path ./outputs/checkpoints/best_accuracy.pth \
    --mode single \
    --input /path/to/image.jpg \
    --output predictions.json
```

**Batch Prediction**:
```bash
python inference.py \
    --model_path ./outputs/checkpoints/best_accuracy.pth \
    --mode batch \
    --input /path/to/images/directory \
    --output batch_predictions.json
```

**Real-time Webcam Prediction**:
```bash
python inference.py \
    --model_path ./outputs/checkpoints/best_accuracy.pth \
    --mode webcam
```

## Configuration

The system uses a YAML configuration file (`config/config.yaml`) to manage hyperparameters and settings:

```yaml
dataset:
  image_size: [224, 224]
  num_landmarks: 68
  emotions: ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

model:
  backbone: 'resnet50'
  num_emotion_classes: 7
  dropout_rate: 0.3
  pretrained: true

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 50
  weight_decay: 0.0001
```

## Model Architecture

### Supported Backbones
- **ResNet-18/50**: Efficient and proven CNN architectures
- **EfficientNet-B0**: Modern, efficient architecture with excellent performance

### Model Types

1. **EmotionClassifier**: Single-task model for emotion classification
2. **ValenceArousalRegressor**: Single-task model for valence/arousal regression  
3. **MultiTaskFaceModel**: Multi-task model combining emotion classification with valence/arousal regression

### Key Features
- **Transfer Learning**: Uses pre-trained ImageNet weights
- **Landmark Integration**: Optional facial landmark features
- **Multi-task Learning**: Joint optimization of multiple objectives
- **Regularization**: Dropout and weight decay for better generalization

## Performance Monitoring

The system provides comprehensive monitoring through:

- **TensorBoard**: Real-time training visualization
- **Training History Plots**: Loss and accuracy curves
- **Confusion Matrices**: Detailed classification performance
- **Checkpointing**: Automatic saving of best models

## Example Usage

```python
from src.models.face_models import MultiTaskFaceModel
from src.data.dataset import get_transforms
from PIL import Image
import torch

# Load model
model = MultiTaskFaceModel(num_emotion_classes=7)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load and preprocess image
image = Image.open('face.jpg')
transform = get_transforms(is_training=False)
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    emotion_probs = torch.softmax(outputs['emotion_logits'], dim=1)
    predicted_emotion = torch.argmax(emotion_probs, dim=1).item()
    
    print(f"Emotion: {predicted_emotion}")
    print(f"Valence: {outputs['valence'].item():.3f}")
    print(f"Arousal: {outputs['arousal'].item():.3f}")
```

## Requirements

- Python 3.7+
- PyTorch 1.12+
- OpenCV 4.5+
- NumPy, Pandas, Matplotlib
- Optional: dlib (for landmark detection)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{facial_expression_recognition_2024,
  title={Facial Expression Recognition and Affective Computing},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or support, please open an issue or contact [your-email@domain.com].
