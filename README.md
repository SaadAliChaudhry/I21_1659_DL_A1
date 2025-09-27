# I21_1659_DL_A1
This project implements Facial Expression Recognition and Affective Computing using deep learning. The dataset consists of cropped RGB face images (224×224), 68 facial landmarks, categorical emotion labels, and continuous valence/arousal annotations.
Facial Expression Recognition with Valence & Arousal Prediction
This project implements facial expression recognition and affective state prediction (valence & arousal) using a deep learning approach.
The model performs multi-task learning:
•	Categorical classification → 8 emotions
•	Continuous regression → Valence ([-1, +1]) & Arousal ([-1, +1])
Repository Structure
│── I21_1659_A1.ipynb    # Jupyter Notebook with preprocessing, training & evaluation  
│── README.md            # Project documentation  
│── data/                # (Dataset Not included) Images Uploaded as Png’s 

Dataset
•	Images: Cropped & resized RGB faces (224×224).
•	Annotations:
o	Emotion labels (0–7 → Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
o	Valence & Arousal scores in range [-1, +1]
•	Facial landmarks: 68 points per face.
•	Split: 80% train / 20% validation
 Preprocessing
•	Training augmentations: random crop, horizontal flip, color jitter
•	Validation preprocessing: resize to 224×224
•	Normalization: ImageNet mean & std
Model Architecture
•	Backbone: ResNet18 (ImageNet pretrained)
•	Heads:
o	Classification → Softmax over 8 classes
o	Regression (Valence) → Single output neuron
o	Regression (Arousal) → Single output neuron
Training Configuration
•	Optimizer: Adam (lr=1e-4)
•	Scheduler: ReduceLROnPlateau
•	Batch size: 32
•	Epochs: 30
•	Loss: Weighted combination of classification + regression losses
 Evaluation Metrics
•	Classification: Accuracy, F1-Score, Cohen’s Kappa, (optional ROC-AUC/PR-AUC)
•	Regression: RMSE, CORR, SAGR, CCC

 How to Run
1.	Clone this repo:
2.	git clone https://github.com/<your-username>/<repo-name>.git
3.	cd <repo-name>
4.	Install requirements:
5.	pip install -r requirements.txt
6.	Place dataset inside data/ directory.
7.	Open notebook and run:
8.	jupyter notebook I21_1659_A1.ipynb

Results & Discussion
•	Multi-task learning improved representation sharing.
•	Pretrained backbone sped up convergence.
•	Regression of valence/arousal was more challenging than categorical classification.
•	Class imbalance (e.g., contempt, disgust) impacted accuracy.
