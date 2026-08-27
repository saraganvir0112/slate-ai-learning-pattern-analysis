# Slate - AI Learning Pattern Analysis

Slate is an AI-based student learning analytics and academic performance prediction system designed to analyze student learning behaviour across a 12-week trajectory.

The system combines Pattern Recognition and Machine Learning techniques to identify learning patterns, analyze learner profiles, predict academic performance, and prioritize students who may require attention.

## Features

- 12-week student learning trajectory analysis
- Academic performance prediction
- Student trajectory pattern detection
- PCA-based dimensionality reduction
- K-Means clustering
- K-Nearest Neighbors classification
- Decision Tree classification
- Gaussian Naive Bayes classification
- Accuracy, Precision, Recall and F1-Score evaluation
- Confusion Matrix analysis
- Interactive web-based predictor
- Teacher dashboard with prioritized students
- Flask-based Python backend
- Real-time communication between frontend and ML backend

## Learning Signals

Slate analyzes five major learning signals:

- Attendance
- Quiz and test scores
- Assignment completion
- Time on task
- Engagement score

These signals are used to represent each learner as a numerical pattern and analyze changes in learning behaviour over time.

## Trajectory Patterns

Slate analyzes student data across multiple weeks to identify four major learning trajectory patterns.

### Steady Climber

A learner showing a consistent upward trend across academic and engagement signals.

### Late Bloomer

A learner showing slower initial progress followed by increasing engagement and improved performance.

### Plateaued

A learner whose performance remains relatively stable after an initial period of progress.

### At-Risk Decliner

A learner showing a sustained downward trend in academic performance and engagement.

## Machine Learning Pipeline

```text
Student Learning Data
        |
        v
Data Preprocessing
        |
        v
Feature Scaling
        |
        +----------------------+
        |                      |
        v                      v
       PCA                 K-Means
        |                 Clustering
        |                      |
        v                      v
Pattern Visualization    Learner Profiles
        |
        v
Classification
        |
   +----+----+-------------+
   |         |             |
   v         v             v
  KNN   Decision Tree  Naive Bayes
   |         |             |
   +---------+-------------+
             |
             v
      Performance Prediction
             |
             v
       Trajectory Analysis
             |
             v
       Risk Prioritization
             |
             v
       Teacher Dashboard
```

## Project Architecture

```text
slate-ai-learning-pattern-analysis/
│
├── index.html
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── student_data.csv
│
└── model/
    ├── train_model.py
    ├── evaluate_model.py
    ├── visualize_model.py
    ├── student_model.pkl
    └── pca_clustering.png
```

## Technologies Used

### Frontend

- HTML
- CSS
- JavaScript

### Backend

- Python
- Flask

### Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib

## Dataset

The current project uses a generated 12-week student learning trajectory dataset containing:

- 60 students
- 12 weeks per student
- 720 total records

Each weekly record contains:

```text
StudentID
Week
Attendance
QuizScore
AssignmentCompletion
TimeOnTask
EngagementScore
Performance
TrajectoryPattern
```

## Pattern Recognition Techniques Used

### PCA

Principal Component Analysis is used to reduce the dimensionality of the five learning signals and visualize student learning patterns in a two-dimensional space.

The current implementation uses two principal components for visualization.

### K-Means Clustering

K-Means clustering is used to group students based on similarity in their learning signal profiles.

The clustering stage currently uses three learner-profile clusters.

### Classification

Three classification algorithms are trained and compared:

- K-Nearest Neighbors
- Decision Tree
- Gaussian Naive Bayes

The best-performing model is selected based on F1-Score.

## Model Evaluation

The classification models were evaluated using a held-out test set.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| KNN | 97.92% | 97.92% | 97.92% | 97.91% |
| Decision Tree | 98.61% | 98.67% | 98.61% | 98.62% |
| Naive Bayes | 97.22% | 97.52% | 97.22% | 97.27% |

### Best Performing Model

**Decision Tree**

- Accuracy: **98.61%**
- Precision: **98.67%**
- Recall: **98.61%**
- F1 Score: **98.62%**

## PCA Analysis

The first two principal components currently explain approximately **97.5% of the variance** in the learning-signal dataset.

The PCA visualization is used to observe the separation of student learning profiles.

## Teacher Dashboard

The teacher dashboard analyzes all available student records and prioritizes the six learners requiring the most immediate attention.

Students are ranked using risk level and predicted performance.

The dashboard displays:

- Student
- Attendance
- Learning trajectory
- Predicted performance
- Risk level
- Learning pattern
- Cluster information

## Interactive Predictor

The web-based predictor allows users to provide:

- Attendance rate
- Average quiz score
- Assignment completion
- Engagement score

The frontend sends these values to the Flask backend.

The backend then:

1. Preprocesses the input
2. Generates the required feature representation
3. Applies the trained Machine Learning model
4. Predicts student performance
5. Calculates prediction confidence
6. Identifies the learner cluster
7. Returns the result to the website

## API Endpoints

### Home

```text
GET /
```

Serves the Slate web application.

### Prediction

```text
POST /predict
```

Accepts learner input values and returns:

- Predicted performance
- Estimated grade
- Risk level
- Cluster
- Confidence
- Model information

### Student Dashboard

```text
GET /students
```

Analyzes the student dataset and returns the six highest-priority learners.

### Model Information

```text
GET /model-info
```

Returns model performance information and dataset statistics.

## Project Workflow

```text
Collect
   ↓
Preprocess
   ↓
Feature Scaling
   ↓
PCA / Pattern Analysis
   ↓
K-Means Clustering
   ↓
Classification
   ↓
Performance Prediction
   ↓
Trajectory Analysis
   ↓
Risk Prioritization
   ↓
Teacher Dashboard
```

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/saraganvir0112/slate-ai-learning-pattern-analysis.git
cd slate-ai-learning-pattern-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Machine Learning Model

```bash
python model/train_model.py
```

This generates the trained model file:

```text
model/student_model.pkl
```

### 4. Start the Flask Server

```bash
python app.py
```

### 5. Open the Application

```text
http://127.0.0.1:5000
```

## Visualization

The project includes a PCA-based visualization of student learning profiles.

The visualization is generated using:

```bash
python model/visualize_model.py
```

The resulting visualization is saved as:

```text
model/pca_clustering.png
```

## Project Purpose

The purpose of Slate is to demonstrate how Pattern Recognition and Machine Learning can be used to analyze student learning behaviour rather than relying only on final examination results.

The system focuses on identifying patterns in learning signals, recognizing learner trajectories, predicting academic performance, and helping teachers prioritize students who may need additional attention.

## Disclaimer

Slate is an academic demonstration project.

The dataset used in this repository is generated for experimentation and does not represent real student records.

The predictions produced by the system are intended for educational demonstration and should not be treated as definitive judgments about individual students.

## Author

Developed as an academic Pattern Recognition project.