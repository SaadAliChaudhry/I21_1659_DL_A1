"""
Utility functions for facial landmark detection and processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image

# Optional imports - graceful fallback if not available
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Landmark detection will use face_recognition fallback.")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Landmark detection will be limited.")


class FacialLandmarkDetector:
    """
    Facial landmark detection using dlib's 68-point model
    """
    
    def __init__(self, predictor_path: Optional[str] = None):
        """
        Initialize facial landmark detector
        
        Args:
            predictor_path: Path to dlib's shape predictor model file
                           If None, will try to use face_recognition library
        """
        if not DLIB_AVAILABLE and not FACE_RECOGNITION_AVAILABLE:
            raise ImportError("Neither dlib nor face_recognition is available. Please install one of them.")
        
        if DLIB_AVAILABLE:
            self.detector = dlib.get_frontal_face_detector()
        
        if predictor_path and predictor_path.endswith('.dat') and DLIB_AVAILABLE:
            self.predictor = dlib.shape_predictor(predictor_path)
            self.use_dlib = True
        else:
            # Fallback to face_recognition library
            if FACE_RECOGNITION_AVAILABLE:
                self.use_dlib = False
                print("Using face_recognition library for landmark detection")
            else:
                raise ImportError("No suitable landmark detection library available")
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect 68 facial landmarks in an image
        
        Args:
            image: Input image as numpy array (RGB or BGR)
            
        Returns:
            Array of shape (68, 2) containing (x, y) coordinates of landmarks
            Returns None if no face is detected
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale for dlib
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if self.use_dlib:
            return self._detect_landmarks_dlib(gray)
        else:
            return self._detect_landmarks_face_recognition(image)
    
    def _detect_landmarks_dlib(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """Detect landmarks using dlib"""
        if not DLIB_AVAILABLE:
            raise ImportError("dlib not available")
            
        faces = self.detector(gray_image)
        
        if len(faces) == 0:
            return None
        
        # Use the first detected face
        face = faces[0]
        landmarks = self.predictor(gray_image, face)
        
        # Convert to numpy array
        points = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)
        
        return points
    
    def _detect_landmarks_face_recognition(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect landmarks using face_recognition library"""
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError("face_recognition not available")
            
        face_landmarks_list = face_recognition.face_landmarks(image)
        
        if not face_landmarks_list:
            return None
        
        # Use first detected face
        face_landmarks = face_landmarks_list[0]
        
        # Convert to 68-point format
        points = []
        
        # Face outline (17 points)
        points.extend(face_landmarks['chin'])
        
        # Eyebrows (10 points)
        points.extend(face_landmarks['left_eyebrow'])
        points.extend(face_landmarks['right_eyebrow'])
        
        # Nose (9 points)
        points.extend(face_landmarks['nose_bridge'])
        points.extend(face_landmarks['nose_tip'])
        
        # Eyes (12 points)
        points.extend(face_landmarks['left_eye'])
        points.extend(face_landmarks['right_eye'])
        
        # Mouth (20 points)
        points.extend(face_landmarks['top_lip'])
        points.extend(face_landmarks['bottom_lip'])
        
        return np.array(points[:68], dtype=np.int32)
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Visualize facial landmarks on an image
        
        Args:
            image: Input image
            landmarks: Landmark points array of shape (68, 2)
            
        Returns:
            Image with landmarks drawn
        """
        vis_image = image.copy()
        
        for point in landmarks:
            cv2.circle(vis_image, tuple(point), 2, (0, 255, 0), -1)
        
        # Draw face outline
        for i in range(16):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        
        # Draw eyebrows
        for i in range(17, 21):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        for i in range(22, 26):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        
        # Draw nose
        for i in range(27, 30):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        for i in range(31, 35):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        cv2.line(vis_image, tuple(landmarks[35]), tuple(landmarks[31]), (255, 0, 0), 1)
        
        # Draw eyes
        for i in range(36, 41):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        cv2.line(vis_image, tuple(landmarks[41]), tuple(landmarks[36]), (255, 0, 0), 1)
        
        for i in range(42, 47):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        cv2.line(vis_image, tuple(landmarks[47]), tuple(landmarks[42]), (255, 0, 0), 1)
        
        # Draw mouth
        for i in range(48, 59):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        cv2.line(vis_image, tuple(landmarks[59]), tuple(landmarks[48]), (255, 0, 0), 1)
        
        for i in range(60, 67):
            cv2.line(vis_image, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 0, 0), 1)
        cv2.line(vis_image, tuple(landmarks[67]), tuple(landmarks[60]), (255, 0, 0), 1)
        
        return vis_image


def normalize_landmarks(landmarks: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Normalize landmark coordinates to [0, 1] range
    
    Args:
        landmarks: Landmark points array of shape (68, 2)
        image_size: (width, height) of the image
        
    Returns:
        Normalized landmarks
    """
    normalized = landmarks.astype(np.float32)
    normalized[:, 0] /= image_size[0]  # normalize x coordinates
    normalized[:, 1] /= image_size[1]  # normalize y coordinates
    return normalized


def landmarks_to_vector(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert landmarks array to flat vector for neural network input
    
    Args:
        landmarks: Landmark points array of shape (68, 2)
        
    Returns:
        Flattened vector of shape (136,)
    """
    return landmarks.flatten()


def extract_landmark_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract geometric features from facial landmarks
    
    Args:
        landmarks: Landmark points array of shape (68, 2)
        
    Returns:
        Feature vector with geometric properties
    """
    features = []
    
    # Eye aspect ratios
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    
    left_ear = calculate_eye_aspect_ratio(left_eye)
    right_ear = calculate_eye_aspect_ratio(right_eye)
    features.extend([left_ear, right_ear])
    
    # Mouth aspect ratio
    mouth = landmarks[48:68]
    mar = calculate_mouth_aspect_ratio(mouth)
    features.append(mar)
    
    # Face width and height
    face_width = np.abs(landmarks[16][0] - landmarks[0][0])
    face_height = np.abs(landmarks[8][1] - landmarks[27][1])
    features.extend([face_width, face_height])
    
    # Eyebrow height
    left_eyebrow_height = np.mean(landmarks[17:22, 1]) - np.mean(landmarks[36:42, 1])
    right_eyebrow_height = np.mean(landmarks[22:27, 1]) - np.mean(landmarks[42:48, 1])
    features.extend([left_eyebrow_height, right_eyebrow_height])
    
    return np.array(features, dtype=np.float32)


def calculate_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """Calculate eye aspect ratio for blink detection"""
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mouth_aspect_ratio(mouth_landmarks: np.ndarray) -> float:
    """Calculate mouth aspect ratio for expression analysis"""
    # Vertical distances
    A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])  # 50 - 58
    B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])   # 52 - 56
    
    # Horizontal distance
    C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])   # 48 - 54
    
    # Mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar