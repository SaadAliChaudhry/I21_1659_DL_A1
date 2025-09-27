"""
Main training script for facial expression recognition and affective computing
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset import FaceExpressionDataset, get_transforms, create_data_loaders
from models.face_models import EmotionClassifier, ValenceArousalRegressor, MultiTaskFaceModel
from training.trainer import Trainer, plot_training_history
from utils.landmarks import FacialLandmarkDetector


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train facial expression recognition models')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--task', type=str, choices=['emotion', 'valence_arousal', 'multitask'],
                        default='multitask', help='Training task')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing training data')
    parser.add_argument('--train_annotations', type=str, required=True,
                        help='Path to training annotations CSV file')
    parser.add_argument('--val_annotations', type=str, required=True,
                        help='Path to validation annotations CSV file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, task: str) -> nn.Module:
    """
    Create model based on task and configuration
    
    Args:
        config: Configuration dictionary
        task: Training task ('emotion', 'valence_arousal', 'multitask')
        
    Returns:
        Initialized model
    """
    model_config = config['model']
    
    if task == 'emotion':
        model = EmotionClassifier(
            num_classes=model_config['num_emotion_classes'],
            backbone=model_config['backbone'],
            pretrained=model_config['pretrained'],
            dropout_rate=model_config['dropout_rate'],
            use_landmarks=False  # Can be made configurable
        )
    elif task == 'valence_arousal':
        model = ValenceArousalRegressor(
            backbone=model_config['backbone'],
            pretrained=model_config['pretrained'],
            dropout_rate=model_config['dropout_rate'],
            use_landmarks=False  # Can be made configurable
        )
    elif task == 'multitask':
        model = MultiTaskFaceModel(
            num_emotion_classes=model_config['num_emotion_classes'],
            backbone=model_config['backbone'],
            pretrained=model_config['pretrained'],
            dropout_rate=model_config['dropout_rate'],
            use_landmarks=False  # Can be made configurable
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return model


def create_optimizer_and_scheduler(model: nn.Module, config: dict):
    """
    Create optimizer and learning rate scheduler
    
    Args:
        model: Neural network model
        config: Configuration dictionary
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    train_config = config['training']
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_config['scheduler_step_size'],
        gamma=train_config['scheduler_gamma']
    )
    
    return optimizer, scheduler


def create_datasets_and_loaders(
    data_dir: str,
    train_annotations: str,
    val_annotations: str,
    config: dict,
    task: str
):
    """
    Create datasets and data loaders
    
    Args:
        data_dir: Root data directory
        train_annotations: Training annotations file
        val_annotations: Validation annotations file
        config: Configuration dictionary
        task: Training task
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    batch_size = config['training']['batch_size']
    
    # Determine what to include based on task
    include_emotions = task in ['emotion', 'multitask']
    include_valence_arousal = task in ['valence_arousal', 'multitask']
    
    # Create datasets
    train_dataset = FaceExpressionDataset(
        data_dir=data_dir,
        annotations_file=train_annotations,
        transform=get_transforms(is_training=True),
        include_landmarks=False,  # Can be made configurable
        include_emotions=include_emotions,
        include_valence_arousal=include_valence_arousal
    )
    
    val_dataset = FaceExpressionDataset(
        data_dir=data_dir,
        annotations_file=val_annotations,
        transform=get_transforms(is_training=False),
        include_landmarks=False,  # Can be made configurable
        include_emotions=include_emotions,
        include_valence_arousal=include_valence_arousal
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config, args.task)
    print(f"Created {args.task} model with backbone {config['model']['backbone']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Create datasets and loaders
    train_loader, val_loader = create_datasets_and_loaders(
        args.data_dir,
        args.train_annotations,
        args.val_annotations,
        config,
        args.task
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
    
    # Resume training if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed training from epoch {start_epoch}")
    
    # Train model
    num_epochs = config['training']['num_epochs']
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Task: {args.task}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print("-" * 50)
    
    history = trainer.train(num_epochs)
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    print(f"Training history plot saved to: {plot_path}")
    
    print("Training completed!")


if __name__ == '__main__':
    main()