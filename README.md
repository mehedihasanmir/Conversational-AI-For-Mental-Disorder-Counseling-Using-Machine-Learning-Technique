# 🧠 Conversational AI for Mental Disorder Counseling Using Machine Learning

This project integrates **machine learning**, **explainable AI (LIME)**, and **conversational AI (OpenAI)** to build a comprehensive mental health support system. It detects psychological disorders through a 10-question assessment and provides personalized counseling through an intelligent chatbot interface.

---

## 🔗 Project Demo

![Well-being Assessment Form](./photo/Home%20Page.png)
![Prediction Result](./photo/prediction.png)
![Chatbot Welcome Screen after Skip Question](./photo/Ahter%20skiping%20Question.png)

---

## 🧩 Features

- **Mental Health Assessment** based on 10 key psychological indicators.
- **Multimodal Input Interface** (numeric scale, dropdowns, and radio buttons).
- **Machine Learning Classifier** for detecting:
  - Depression
  - Bipolar Type-1
  - Bipolar Type-2
  - Normal Mental State
- **LIME-based Explainable AI** to highlight which features influenced the model's prediction with detailed explanations.
- **Conversational AI Chatbot** powered by OpenAI GPT-3.5-turbo for empathetic and intelligent counseling.
- **Frontend built with HTML, CSS, JavaScript**, and **backend with Flask**.
- Deployed locally using `python app.py`.

---

## 🚀 Technologies Used

### 🧠 Machine Learning & Deep Learning
- `Pandas`, `NumPy`, `Seaborn`, `Matplotlib` – Data manipulation & visualization
- `Scikit-learn` – Preprocessing, modeling, and evaluation
- `CatBoostClassifier`, `XGBoostClassifier`, `RandomForest`, `Logistic Regression`, `KNN`, `SVM`
- `TensorFlow / Keras` – Basic ANN for comparison
- `LIME` – Explainable AI framework for local interpretable explanations

### 💬 Conversational AI
- **OpenAI GPT-3.5-turbo** – Intelligent chatbot responses
- **LIME (Local Interpretable Model-agnostic Explanations)** – Explainability framework
- Chatbot integrates seamlessly after disorder prediction step

### 🌐 Web Development
- **HTML, CSS, JS** – Frontend interface
- **Flask** – Backend logic and API connection
- **Localhost Deployment** using `app.py`

---

## 🧪 Dataset & Features

dataset:- https://www.kaggle.com/datasets/cid007/mental-disorder-classification

| Feature                | Type        | Description                                 |
|------------------------|-------------|---------------------------------------------|
| Mood Swing             | Binary      | Recent mood fluctuation (Yes/No)            |
| Optimism               | Numeric     | Scale from 1 to 10                          |
| Sadness                | Categorical | Frequency (Never, Sometimes, Usually)       |
| Exhaustion             | Categorical | Frequency (Never, Seldom, Often)            |
| Authority Respect      | Binary      | Struggle with authority (Yes/No)            |
| Euphoric               | Categorical | Frequency (Never, Sometimes, Usually)       |
| Suicidal Thoughts      | Binary      | Any suicidal thoughts (Yes/No)              |
| Sleep Disorder         | Categorical | Suffering from sleep issues (Yes/No)        |
| Sexual Activity        | Numeric     | Scale from 1 to 10                          |
| Concentration Level    | Numeric     | Scale from 1 to 10                          |

---

## 🧠 Workflow

### 1. **Data Preprocessing**
- Cleaning & encoding
- Scaling using `StandardScaler`
- Feature engineering via `mutual_info_classif` to reduce columns

### 2. **Model Training**
- Multiple ML models trained and evaluated
- Best-performing: **CatBoostClassifier** for tabular data
- Model saved as `catboost_mental_health_model.pkl`

### 3. **Model Evaluation**
- Accuracy Score
- Cross-validation
- Classification Report

### 4. **Explainability with LIME**
- **LIME (Local Interpretable Model-agnostic Explanations)** provides local explanations
- Identifies top 3 features that influenced each prediction
- Feature importance scores help users understand model decisions
- Requires preprocessed training data sample (`training_data_sample.npy`)

### 5. **Web Integration**
- Responsive frontend with Tailwind CSS
- User submits assessment via form
- Backend processes inputs and returns:
  - Mental disorder prediction (Normal, Bipolar Type-1, Bipolar Type-2, or Depression)
  - LIME-based explanations with key influencing features
  - Feature importance scores
- Activates conversational chatbot powered by OpenAI GPT-3.5-turbo
- Skip option available to bypass assessment and start chatting immediately

---

## 🔧 Configuration & Setup

### Environment Setup
Create a `.env` file in the project root with:
```env
OPENAI_API_KEY=your-openai-api-key-here
SERVER_PORT=5000
FLASK_DEBUG=True
```

**Important:** Obtain your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### Required Data Files
Before running the application, ensure these files exist in the project root:
1. **`catboost_mental_health_model.pkl`** – Pre-trained CatBoost model
   - Generated during notebook execution
   - Used for disorder predictions

2. **`training_data_sample.npy`** – Preprocessed training data sample
   - NumPy array of preprocessed features
   - Used by LIME for generating local explanations
   - Create by saving a subset of your training data during model training:
     ```python
     import numpy as np
     # After preprocessing your training data
     np.save('training_data_sample.npy', X_train_sample)
     ```

**If `training_data_sample.npy` is missing:** The application will fall back to dummy data, which may result in less reliable LIME explanations.

---

## 🔧 How to Run Locally

### Prerequisites
- Python 3.8+
- OpenAI API key
- All dependencies installed

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mehedihasanmir/Conversational-AI-For-Mental-Disorder-Counseling-Using-Machine-Learning-Technique
   cd your-repo
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file with your OpenAI API key:**
   ```env
   OPENAI_API_KEY=your-actual-openai-api-key
   SERVER_PORT=5000
   FLASK_DEBUG=True
   ```

5. **Ensure required model and data files exist:**
   - `catboost_mental_health_model.pkl` (generated from notebook)
   - `training_data_sample.npy` (generated from notebook)
   - See "Configuration & Setup" section above for details

6. **Start the Flask server:**
   ```bash
   python app.py
   ```
   You should see:
   ```
   Model loaded successfully!
   Training data for LIME loaded successfully!
   LIME Explainer initialized successfully with loaded training data
   Running on http://127.0.0.1:5000/
   ```

7. **Visit in your browser:**
   ```
   http://127.0.0.1:5000/
   ```

### Troubleshooting Common Issues

**Issue:** "OPENAI_API_KEY not set in .env file"
- **Solution:** Create `.env` file with valid OpenAI API key

**Issue:** "Model not found" or "Model loaded with error"
- **Solution:** Ensure `catboost_mental_health_model.pkl` exists. Generate it by running the notebook.

**Issue:** "LIME explanations not available"
- **Solution:** Provide `training_data_sample.npy`. The app can work with dummy data but explanations will be less reliable.

---

## 🧠 Counseling Chatbot & Prediction Flow

### Assessment & Prediction Flow
1. **User completes 10-question assessment** covering psychological indicators
2. **Form submission** sends data to `/predict` endpoint
3. **Model prediction** returns:
   - Mental health diagnosis (Normal, Bipolar Type-1, Bipolar Type-2, Depression)
   - Empathetic message tailored to the prediction
   - LIME-based explanation with top 3 influencing features
   - Feature importance scores

### Chatbot Interaction
- **Triggered after prediction** or via "Skip Assessment" button
- **Powered by OpenAI GPT-3.5-turbo** for intelligent, context-aware responses
- **Personalized counseling** based on prediction result
- **Conversation history** maintained for context-aware interactions
- Features:
  - Supportive and empathetic responses
  - Practical mental health advice
  - Encouragement to seek professional help when needed
  - Can be activated without assessment (skip option)

### API Endpoints

#### `/predict` (POST)
Generates mental health prediction with LIME explanations
- **Input:** JSON with 10 features (mood_swing, optimism, sadness, exhausted, authority_respect, euphoric, suicidal_thoughts, sleep_disorder, sexual_activity, concentration)
- **Output:** 
  ```json
  {
    "prediction": "Depression!",
    "explanation": "The key features that influenced this prediction are:<br>- Feature Name: ...",
    "top_features": [
      {"feature": "Feature Name", "importance": 0.1234}
    ]
  }
  ```

#### `/chat` (POST)
Generates chatbot responses using OpenAI API
- **Input:** JSON with conversation history
  ```json
  {
    "messages": [
      {"role": "user", "content": "How can I manage my anxiety?"},
      {"role": "assistant", "content": "..."}
    ]
  }
  ```
- **Output:**
  ```json
  {
    "message": "Chatbot response here..."
  }
  ```

---

## 📊 Model Performance

| Model                   | Accuracy (No Feature Selection) | Accuracy (With Feature Selection) |
|-------------------------|-------------------------------|-----------------------------------|
| CatBoost                | 0.8900                       | 0.9000                            |
| K-Nearest Neighbors     | 0.8083                        | 0.8167                            |
| Support Vector Classifier (SVC) | 0.8917              | 0.8667                            |
| XGBoost                 | 0.8333                        | 0.8750                            |
| Random Forest           | 0.8333                        | 0.8250                            |
| Logistic Regression     | 0.8833                        | 0.8250                            |
| Artificial Neural Network | 0.8000                      | 0.8000                            |

**Model Selection:** CatBoost was chosen for production due to consistent 90% accuracy and superior performance on tabular data.

---

## 🎨 Frontend Features

### Technology Stack
- **Tailwind CSS** – Responsive, modern UI design
- **Vanilla JavaScript** – No framework dependencies for lightweight app
- **Responsive Layout** – Works seamlessly on desktop and mobile devices

### User Interface Components
1. **Assessment Form** (Left panel on desktop)
   - 10 psychological indicator questions
   - Mix of radio buttons, dropdowns, and sliders
   - Clear labels with visual feedback
   - "Submit" button for prediction
   - "Skip Assessment" button to start chatting directly

2. **Prediction Results** (Center)
   - Large, clear diagnosis display
   - Empathetic, personalized message
   - LIME explanation with key influencing features
   - Visual organization of feature importance
   - "Back to Assessment" option to restart

3. **Chatbot Interface** (Right panel / Full width after prediction)
   - Clean message history display
   - User messages aligned right (blue bubbles)
   - Bot messages aligned left (gray bubbles)
   - Input field with send button
   - Auto-scrolling conversation
   - Markdown and code formatting support

### Responsive Design
- **Desktop:** Side-by-side assessment form and chat
- **Tablet:** Stacked layout with adaptive sizing
- **Mobile:** Full-width, optimized touch interactions
- **Smooth transitions** between assessment and chat views

---

## 🔍 Understanding LIME Explanations

**LIME (Local Interpretable Model-agnostic Explanations)** provides transparent, human-understandable explanations for each prediction. Unlike black-box models, LIME shows:

### How It Works
1. **Generates variations** of the input around the predicted instance
2. **Gets predictions** for these variations
3. **Learns a simple, interpretable model** (e.g., linear regression) to approximate the complex model's behavior locally
4. **Identifies top features** that contributed most to the prediction

### Output Example
```
Prediction: Depression!
Key Features Influencing This Prediction:
- Exhaustion: Usually → Increases depression likelihood
- Mood Swings: Yes → Increases depression likelihood  
- Optimism: 3 (Low) → Increases depression likelihood
```

### Why LIME Over SHAP?
- **Model-agnostic:** Works with any model (CatBoost, XGBoost, Neural Networks, etc.)
- **Local explanations:** Focuses on the specific prediction being made
- **Computationally efficient:** Faster than global explanation methods
- **User-friendly:** Easier to understand for non-technical users
- **No retraining required:** Can explain existing models

---

## 📚 Future Improvements

- Add real-time emotion detection (via camera/microphone)
- Secure user history and data with authentication
- Host the application on a cloud server (e.g., Heroku, AWS, Render)
- Extend chatbot to support voice-based interaction
- Integrate additional explainability methods (SHAP, attention mechanisms)
- Add appointment scheduling for licensed therapists
- Implement multi-language support
- Enhanced data visualization for LIME explanations
- Persistent user sessions and history tracking

---

## 📁 Project Structure

```
Conversational-AI-For-Mental-Disorder-Counseling/
│
├── app.py                                    # Flask backend server
├── mental-disorder-classification.ipynb      # Jupyter notebook for model training
├── index.html                                # Frontend HTML interface
├── script.js                                 # Frontend JavaScript logic
├── style.css                                 # Frontend CSS styling
│
├── catboost_mental_health_model.pkl         # Pre-trained CatBoost model
├── training_data_sample.npy                 # Training data for LIME explanations
├── Dataset-Mental-Disorders.csv             # Original dataset
│
├── .env                                      # Environment configuration (API keys)
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
│
├── catboost_info/                           # CatBoost training logs
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   └── time_left.tsv
│
├── myenv/                                    # Virtual environment (optional)
│   └── [Python packages]
│
└── photo/                                    # Demo screenshots
    ├── Home Page.png
    ├── prediction.png
    └── After skipping Question.png
```

### Key Files

- **`app.py`** – Flask REST API with `/predict` and `/chat` endpoints
- **`mental-disorder-classification.ipynb`** – Model training, evaluation, and LIME setup
- **`index.html`** – Main UI with assessment form and chatbot
- **`script.js`** – Frontend logic for form submission, chat, and API calls
- **`style.css`** – Custom styling on top of Tailwind CSS

---

**Mehedi Hasan Mir**    
AI & Data Science Enthusiast  
GitHub: [My-profile](https://github.com/mehedihasanmir)