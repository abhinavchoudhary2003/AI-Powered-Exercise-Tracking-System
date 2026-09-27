"# 🏋️ AI-Powered-Exercise-Tracking-System" 

An intelligent, camera-based **Exercise Tracking System** that uses computer vision to automatically recognize gym users and track their exercise performance — with **zero manual input**.

The system combines **face recognition, vector-based identity matching, pose detection, and exercise-specific tracking logic** into a single kiosk-style experience. A user simply steps in front of the camera, selects an exercise, and starts training.

---

## ✨ Features

### 👤 Face-Based Recognition

* Automatically identifies returning users using facial recognition.
* New users are guided through a one-time registration process.
* Captures multiple face angles during registration to improve recognition robustness.
* Generates **Facenet512 facial embeddings** using DeepFace.
* Stores facial embeddings in **ChromaDB**, a vector database optimized for similarity search.
* Uses **cosine similarity-based vector matching** to identify registered users.
* No traditional username/password login is required.

### 🧠 Vector Database for Face Embeddings

The system uses a hybrid database architecture:

```text
                    Face Image
                        │
                        ▼
              ┌──────────────────┐
              │ recognition_face │
              │      .py         │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    DeepFace      │
              │   Facenet512     │
              └────────┬─────────┘
                       │
                       ▼
             512-D Face Embedding
                       │
                       ▼
              ┌──────────────────┐
              │    ChromaDB      │
              │  Vector Storage  │
              └────────┬─────────┘
                       │
                       ▼
              Similarity Search
                       │
                       ▼
                 User Identity
```

**ChromaDB** is used specifically for storing and retrieving facial embeddings, while **SQLite** handles structured application data such as users, exercises, and workout sessions.

This separation allows the project to use the appropriate database technology for each type of data:

| Data                | Storage  |
| ------------------- | -------- |
| Face embeddings     | ChromaDB |
| User information    | SQLite   |
| Exercise sessions   | SQLite   |
| Rep counts          | SQLite   |
| Plank hold duration | SQLite   |
| Similarity search   | ChromaDB |

---

## 🏋️ Push-Up & Squat Tracking

* Counts repetitions automatically in real time.
* Uses body pose landmarks and joint angles rather than raw motion detection.
* Detects transitions between exercise states.
* Helps distinguish complete movements from incomplete movements.

---

## 🧘 Plank Hold Tracking

* Tracks plank duration instead of repetitions.
* Uses torso/body alignment to determine valid form.
* Brief form breaks pause the timer instead of resetting the session.
* Accumulates valid hold time throughout the exercise.

---

## 🧠 Session State Machine

A centralized session manager controls the complete workout lifecycle:

```text
IDLE
  ↓
FACE SCANNING
  ↓
USER IDENTIFICATION
  ↓
REGISTRATION (if new)
  ↓
SESSION LOCK
  ↓
EXERCISE TRACKING
  ↓
SESSION COMPLETE
  ↓
SAVE RESULTS
```

This provides a consistent pipeline across all supported exercises.

---

## 💻 Interactive Streamlit Interface

* Touch-friendly kiosk-style interface.
* Exercise selection from the home screen.
* Live camera feed.
* Real-time repetition/hold-time statistics.
* Progress indicators.
* Session timer.
* Automatic workout result storage.

---

# 🏗️ System Architecture

```text
                         ┌───────────────┐
                         │    Webcam     │
                         └───────┬───────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Face Detection       │
                    │      MediaPipe         │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ recognition_face.py     │
                    │   Face Recognition      │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ DeepFace / Facenet512  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Face Embedding        │
                    │    512-D Vector        │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │       ChromaDB         │
                    │   Vector Database      │
                    └────────────┬───────────┘
                                 │
                         Similarity Search
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    User Identified     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    Session Manager     │
                    └────────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       ┌────────────┐     ┌────────────┐     ┌────────────┐
       │  Push-Up   │     │   Squat    │     │   Plank    │
       │  Tracker   │     │  Tracker   │     │  Tracker   │
       └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     SQLite      │
                       │ Session Storage │
                       └─────────────────┘
```

---

# 🧠 Face Recognition Pipeline

The face-recognition system follows a vector-search architecture.

### Registration

When a new user registers:

```text
Camera
   ↓
Face Detection
   ↓
Multi-Angle Face Capture
   ↓
DeepFace / Facenet512
   ↓
Generate Face Embeddings
   ↓
Store Embeddings → ChromaDB
   ↓
Store User Details → SQLite
```

Multiple facial samples can be associated with the same user to improve recognition across different head orientations and camera conditions.

### Returning User

When an existing user approaches the camera:

```text
Camera
   ↓
Face Detection
   ↓
Face Crop
   ↓
Facenet512 Embedding
   ↓
ChromaDB Similarity Search
   ↓
Nearest Matching Embedding
   ↓
User ID
   ↓
Session Manager
```

The system therefore avoids scanning every stored embedding manually and instead uses a dedicated vector database for embedding retrieval.

---

# 🗄️ Hybrid Database Architecture

The project separates **structured data** from **high-dimensional vector data**.

### SQLite

SQLite is responsible for application and workout information:

```text
Users
 ├── User ID
 ├── Name
 └── Registration Details

Sessions
 ├── Session ID
 ├── User ID
 ├── Exercise
 ├── Repetitions
 ├── Hold Duration
 └── Session Timestamp
```

### ChromaDB

ChromaDB manages facial embeddings:

```text
Face Embedding Collection
│
├── Embedding
├── User ID
├── Metadata
└── Similarity Search
```

This architecture demonstrates the use of both **relational storage and vector databases** within the same computer-vision application.

---

# 🛠️ Technology Stack

| Component            | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python 3.12               |
| User Interface       | Streamlit                 |
| Computer Vision      | OpenCV                    |
| Pose Detection       | MediaPipe Pose Landmarker |
| Face Detection       | MediaPipe Face Detection  |
| Face Recognition     | DeepFace                  |
| Face Embeddings      | Facenet512                |
| Vector Database      | ChromaDB                  |
| Similarity Search    | Cosine Similarity         |
| Structured Database  | SQLite                    |
| Numerical Processing | NumPy                     |

---

# 📂 Project Structure

```text
Exercise_Tracker/
│
├── Backend/
│   ├── services/
│   │   ├── recognition_face.py
│   │   ├── pose_detection.py
│   │   ├── face_detection.py
│   │   ├── exercise_trackers/
│   │   ├── session_manager.py
│   │   └── registration.py
│   │
│   ├── models/
│   │   └── database.py
│   │
│   ├── utils/
│   │   └── angle_utils.py
│   │
│   └── test_session.py
│
├── FrontEnd/
│   └── app.py
│
├── ml_models/
│   └── pose_landmarker_lite.task
│
├── chroma_db/
│   └── face_embeddings/
│
├── exercise_tracker.db
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation & Setup

## Prerequisites

Make sure you have:

* Python 3.12
* Git
* A working webcam
* Sufficient system resources for MediaPipe and DeepFace

---

## 1. Clone the Repository

```bash
git clone https://github.com/abhinavchoudhary2003/Exercise_Tracker.git
cd Exercise_Tracker
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the main dependencies manually:

```bash
pip install streamlit opencv-python mediapipe deepface numpy chromadb
```

---

## 4. Download the Pose Model

Download:

```text
pose_landmarker_lite.task
```

and place it inside:

```text
ml_models/
```

Expected structure:

```text
ml_models/
└── pose_landmarker_lite.task
```

DeepFace downloads the required Facenet512 model weights automatically during its first execution.

---

# ▶️ Running the Application

### Terminal/OpenCV Mode

```bash
python Backend/test_session.py
```

### Streamlit Mode

```bash
streamlit run FrontEnd/app.py
```

---

# 💡 Usage

1. Launch the application.
2. Stand in front of the camera.
3. Face detection locates the user.
4. `recognition_face.py` generates a Facenet512 embedding.
5. ChromaDB searches the stored face embeddings.
6. The system identifies the returning user or starts registration.
7. Select an exercise.
8. Begin exercising.
9. The pose tracker counts repetitions or tracks plank duration.
10. Session results are stored in SQLite.

---

# 📌 Resume Highlights

This project demonstrates practical experience with:

* **Computer Vision**
* **Real-Time Pose Estimation**
* **Face Recognition**
* **Deep Learning Embeddings**
* **Vector Databases(chroma DB)**
* **Similarity Search**
* **SQLite Database Design**
* **State Machine Architecture**
* **Real-Time Video Processing**
* **Streamlit Application Development**

###

---

# 👨‍💻 Author

**Abhinav Choudhary**

GitHub: [@abhinavchoudhary2003](https://github.com/abhinavchoudhary2003)
