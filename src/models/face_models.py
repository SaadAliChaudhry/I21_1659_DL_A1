"""
Neural network models for facial expression recognition and affective computing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple


class EmotionClassifier(nn.Module):
    """
    CNN model for categorical emotion classification
    """
    
    def __init__(
        self, 
        num_classes: int = 7, 
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_landmarks: bool = False,
        landmark_dim: int = 136
    ):
        """
        Initialize emotion classifier
        
        Args:
            num_classes: Number of emotion classes
            backbone: Backbone CNN architecture
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            use_landmarks: Whether to incorporate landmark features
            landmark_dim: Dimension of landmark features
        """
        super(EmotionClassifier, self).__init__()
        
        self.use_landmarks = use_landmarks
        self.backbone_name = backbone
        
        # Load backbone CNN
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Landmark processing branch
        if use_landmarks:
            self.landmark_branch = nn.Sequential(
                nn.Linear(landmark_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            combined_dim = feature_dim + 64
        else:
            combined_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
            landmarks: Facial landmarks (batch_size, landmark_dim) if used
            
        Returns:
            Emotion logits (batch_size, num_classes)
        """
        # Extract features from image
        img_features = self.backbone(x)
        
        if self.use_landmarks and landmarks is not None:
            # Process landmarks
            landmark_features = self.landmark_branch(landmarks)
            # Concatenate features
            combined_features = torch.cat([img_features, landmark_features], dim=1)
        else:
            combined_features = img_features
        
        # Classify emotions
        output = self.classifier(combined_features)
        return output


class ValenceArousalRegressor(nn.Module):
    """
    CNN model for continuous valence and arousal prediction
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_landmarks: bool = False,
        landmark_dim: int = 136
    ):
        """
        Initialize valence-arousal regressor
        
        Args:
            backbone: Backbone CNN architecture
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            use_landmarks: Whether to incorporate landmark features
            landmark_dim: Dimension of landmark features
        """
        super(ValenceArousalRegressor, self).__init__()
        
        self.use_landmarks = use_landmarks
        self.backbone_name = backbone
        
        # Load backbone CNN
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Landmark processing branch
        if use_landmarks:
            self.landmark_branch = nn.Sequential(
                nn.Linear(landmark_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            combined_dim = feature_dim + 64
        else:
            combined_dim = feature_dim
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Separate heads for valence and arousal
        self.valence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        landmarks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
            landmarks: Facial landmarks (batch_size, landmark_dim) if used
            
        Returns:
            Tuple of (valence, arousal) predictions
        """
        # Extract features from image
        img_features = self.backbone(x)
        
        if self.use_landmarks and landmarks is not None:
            # Process landmarks
            landmark_features = self.landmark_branch(landmarks)
            # Concatenate features
            combined_features = torch.cat([img_features, landmark_features], dim=1)
        else:
            combined_features = img_features
        
        # Extract shared features
        shared_features = self.shared_layers(combined_features)
        
        # Predict valence and arousal
        valence = self.valence_head(shared_features)
        arousal = self.arousal_head(shared_features)
        
        return valence.squeeze(-1), arousal.squeeze(-1)


class MultiTaskFaceModel(nn.Module):
    """
    Multi-task model that jointly predicts emotions, valence, and arousal
    """
    
    def __init__(
        self,
        num_emotion_classes: int = 7,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_landmarks: bool = False,
        landmark_dim: int = 136,
        emotion_weight: float = 1.0,
        valence_weight: float = 1.0,
        arousal_weight: float = 1.0
    ):
        """
        Initialize multi-task model
        
        Args:
            num_emotion_classes: Number of emotion classes
            backbone: Backbone CNN architecture
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            use_landmarks: Whether to incorporate landmark features
            landmark_dim: Dimension of landmark features
            emotion_weight: Weight for emotion classification loss
            valence_weight: Weight for valence regression loss
            arousal_weight: Weight for arousal regression loss
        """
        super(MultiTaskFaceModel, self).__init__()
        
        self.use_landmarks = use_landmarks
        self.emotion_weight = emotion_weight
        self.valence_weight = valence_weight
        self.arousal_weight = arousal_weight
        
        # Load backbone CNN
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Landmark processing branch
        if use_landmarks:
            self.landmark_branch = nn.Sequential(
                nn.Linear(landmark_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            combined_dim = feature_dim + 64
        else:
            combined_dim = feature_dim
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads
        self.emotion_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_emotion_classes)
        )
        
        self.valence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        landmarks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
            landmarks: Facial landmarks (batch_size, landmark_dim) if used
            
        Returns:
            Dictionary containing emotion_logits, valence, and arousal predictions
        """
        # Extract features from image
        img_features = self.backbone(x)
        
        if self.use_landmarks and landmarks is not None:
            # Process landmarks
            landmark_features = self.landmark_branch(landmarks)
            # Concatenate features
            combined_features = torch.cat([img_features, landmark_features], dim=1)
        else:
            combined_features = img_features
        
        # Extract shared features
        shared_features = self.shared_layers(combined_features)
        
        # Predict all tasks
        emotion_logits = self.emotion_head(shared_features)
        valence = self.valence_head(shared_features).squeeze(-1)
        arousal = self.arousal_head(shared_features).squeeze(-1)
        
        return {
            'emotion_logits': emotion_logits,
            'valence': valence,
            'arousal': arousal
        }
    
    def compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Emotion classification loss
        if 'emotion_logits' in predictions and 'emotion' in targets:
            emotion_loss = F.cross_entropy(predictions['emotion_logits'], targets['emotion'])
            losses['emotion_loss'] = emotion_loss * self.emotion_weight
        
        # Valence regression loss
        if 'valence' in predictions and 'valence' in targets:
            valence_loss = F.mse_loss(predictions['valence'], targets['valence'])
            losses['valence_loss'] = valence_loss * self.valence_weight
        
        # Arousal regression loss
        if 'arousal' in predictions and 'arousal' in targets:
            arousal_loss = F.mse_loss(predictions['arousal'], targets['arousal'])
            losses['arousal_loss'] = arousal_loss * self.arousal_weight
        
        # Total loss
        losses['total_loss'] = sum(losses.values())
        
        return losses