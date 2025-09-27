"""
Training utilities and functions for facial expression recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    """
    Trainer class for facial expression recognition models
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize trainer
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device for training ('cuda' or 'cpu')
            log_dir: Directory for tensorboard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_emotion_acc': [],
            'val_valence_mae': [],
            'val_arousal_mae': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        running_loss = 0.0
        running_emotion_loss = 0.0
        running_valence_loss = 0.0
        running_arousal_loss = 0.0
        
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {}
            
            if 'emotion' in batch:
                targets['emotion'] = batch['emotion'].to(self.device)
            if 'valence' in batch:
                targets['valence'] = batch['valence'].to(self.device)
            if 'arousal' in batch:
                targets['arousal'] = batch['arousal'].to(self.device)
            
            landmarks = batch.get('landmarks', None)
            if landmarks is not None:
                landmarks = landmarks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'compute_loss'):
                # Multi-task model
                predictions = self.model(images, landmarks)
                losses = self.model.compute_loss(predictions, targets)
                loss = losses['total_loss']
            else:
                # Single task model
                if 'valence' in targets or 'arousal' in targets:
                    # Valence-Arousal model
                    valence_pred, arousal_pred = self.model(images, landmarks)
                    loss = 0
                    if 'valence' in targets:
                        loss += nn.MSELoss()(valence_pred, targets['valence'])
                    if 'arousal' in targets:
                        loss += nn.MSELoss()(arousal_pred, targets['arousal'])
                else:
                    # Emotion classification model
                    emotion_pred = self.model(images, landmarks)
                    loss = nn.CrossEntropyLoss()(emotion_pred, targets['emotion'])
                    
                    # Store predictions for accuracy calculation
                    _, predicted = torch.max(emotion_pred.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets['emotion'].cpu().numpy())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Log losses if multi-task
            if hasattr(self.model, 'compute_loss'):
                if 'emotion_loss' in losses:
                    running_emotion_loss += losses['emotion_loss'].item()
                if 'valence_loss' in losses:
                    running_valence_loss += losses['valence_loss'].item()
                if 'arousal_loss' in losses:
                    running_arousal_loss += losses['arousal_loss'].item()
        
        # Calculate metrics
        num_batches = len(self.train_loader)
        metrics = {
            'loss': running_loss / num_batches,
            'emotion_loss': running_emotion_loss / num_batches,
            'valence_loss': running_valence_loss / num_batches,
            'arousal_loss': running_arousal_loss / num_batches
        }
        
        # Calculate accuracy for emotion classification
        if all_predictions and all_targets:
            metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        
        all_emotion_predictions = []
        all_emotion_targets = []
        all_valence_predictions = []
        all_valence_targets = []
        all_arousal_predictions = []
        all_arousal_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                targets = {}
                
                if 'emotion' in batch:
                    targets['emotion'] = batch['emotion'].to(self.device)
                if 'valence' in batch:
                    targets['valence'] = batch['valence'].to(self.device)
                if 'arousal' in batch:
                    targets['arousal'] = batch['arousal'].to(self.device)
                
                landmarks = batch.get('landmarks', None)
                if landmarks is not None:
                    landmarks = landmarks.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'compute_loss'):
                    # Multi-task model
                    predictions = self.model(images, landmarks)
                    losses = self.model.compute_loss(predictions, targets)
                    loss = losses['total_loss']
                    
                    # Store predictions
                    if 'emotion_logits' in predictions:
                        _, predicted = torch.max(predictions['emotion_logits'].data, 1)
                        all_emotion_predictions.extend(predicted.cpu().numpy())
                        all_emotion_targets.extend(targets['emotion'].cpu().numpy())
                    
                    if 'valence' in predictions:
                        all_valence_predictions.extend(predictions['valence'].cpu().numpy())
                        all_valence_targets.extend(targets['valence'].cpu().numpy())
                    
                    if 'arousal' in predictions:
                        all_arousal_predictions.extend(predictions['arousal'].cpu().numpy())
                        all_arousal_targets.extend(targets['arousal'].cpu().numpy())
                
                else:
                    # Single task model
                    if 'valence' in targets or 'arousal' in targets:
                        # Valence-Arousal model
                        valence_pred, arousal_pred = self.model(images, landmarks)
                        loss = 0
                        if 'valence' in targets:
                            loss += nn.MSELoss()(valence_pred, targets['valence'])
                            all_valence_predictions.extend(valence_pred.cpu().numpy())
                            all_valence_targets.extend(targets['valence'].cpu().numpy())
                        if 'arousal' in targets:
                            loss += nn.MSELoss()(arousal_pred, targets['arousal'])
                            all_arousal_predictions.extend(arousal_pred.cpu().numpy())
                            all_arousal_targets.extend(targets['arousal'].cpu().numpy())
                    else:
                        # Emotion classification model
                        emotion_pred = self.model(images, landmarks)
                        loss = nn.CrossEntropyLoss()(emotion_pred, targets['emotion'])
                        
                        _, predicted = torch.max(emotion_pred.data, 1)
                        all_emotion_predictions.extend(predicted.cpu().numpy())
                        all_emotion_targets.extend(targets['emotion'].cpu().numpy())
                
                running_loss += loss.item()
        
        # Calculate metrics
        num_batches = len(self.val_loader)
        metrics = {'loss': running_loss / num_batches}
        
        # Emotion classification metrics
        if all_emotion_predictions and all_emotion_targets:
            metrics['emotion_accuracy'] = accuracy_score(all_emotion_targets, all_emotion_predictions)
        
        # Valence regression metrics
        if all_valence_predictions and all_valence_targets:
            metrics['valence_mse'] = mean_squared_error(all_valence_targets, all_valence_predictions)
            metrics['valence_mae'] = mean_absolute_error(all_valence_targets, all_valence_predictions)
        
        # Arousal regression metrics
        if all_arousal_predictions and all_arousal_targets:
            metrics['arousal_mse'] = mean_squared_error(all_arousal_targets, all_arousal_predictions)
            metrics['arousal_mae'] = mean_absolute_error(all_arousal_targets, all_arousal_predictions)
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            
            if 'accuracy' in train_metrics:
                self.history['train_acc'].append(train_metrics['accuracy'])
            if 'emotion_accuracy' in val_metrics:
                self.history['val_emotion_acc'].append(val_metrics['emotion_accuracy'])
            if 'valence_mae' in val_metrics:
                self.history['val_valence_mae'].append(val_metrics['valence_mae'])
            if 'arousal_mae' in val_metrics:
                self.history['val_arousal_mae'].append(val_metrics['arousal_mae'])
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            
            if 'accuracy' in train_metrics:
                self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            if 'emotion_accuracy' in val_metrics:
                self.writer.add_scalar('Accuracy/Validation', val_metrics['emotion_accuracy'], epoch)
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, 'best_loss.pth')
            
            if 'emotion_accuracy' in val_metrics and val_metrics['emotion_accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['emotion_accuracy']
                self.save_checkpoint(epoch, 'best_accuracy.pth')
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            
            if 'accuracy' in train_metrics:
                print(f"  Train Acc: {train_metrics['accuracy']:.4f}")
            if 'emotion_accuracy' in val_metrics:
                print(f"  Val Acc: {val_metrics['emotion_accuracy']:.4f}")
            
            print()
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded: {filename}")
        return checkpoint['epoch']


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Accuracy plot
    if 'train_acc' in history and 'val_emotion_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(history['val_emotion_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Valence MAE plot
    if 'val_valence_mae' in history:
        axes[1, 0].plot(history['val_valence_mae'], label='Valence MAE')
        axes[1, 0].set_title('Validation Valence MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Arousal MAE plot
    if 'val_arousal_mae' in history:
        axes[1, 1].plot(history['val_arousal_mae'], label='Arousal MAE')
        axes[1, 1].set_title('Validation Arousal MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], save_path: Optional[str] = None):
    """
    Plot confusion matrix for emotion classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Emotion Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()