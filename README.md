# ❤️ Japan Heart Attack Prediction 🏥

Welcome to the **Japan Heart Attack Prediction** project! This project aims to predict the risk of heart attack using machine learning models trained on a dataset from Japan. The project includes data preprocessing, model training, and deployment using Docker, Amazon S3, ECR, EC2, and MongoDB.

project link : [JAPAN HEART ATTACK](https://japanheartattack.streamlit.app/).

---

## 📂 Directory Structure

Here’s the structure of the project:
<details> <summary>Click to expand/collapse directory structure</summary>

```
└── de5hpande-japan_heart_attack/
    ├── Dockerfile
    ├── app.py
    ├── check_mongodb.py
    ├── main.py
    ├── push_data.py
    ├── requirements.txt
    ├── setup.py
    ├── data/
    │   ├── combined_dataset.csv
    │   ├── heart_attack_risk_dataset.csv
    │   ├── japan_dataset.csv
    │   ├── japan_final_ha.csv
    │   ├── japan_heart_attack_dataset.csv
    │   ├── test.ipynb
    │   └── unbalnced.ipynb
    ├── data_schema/
    │   └── schema.yaml
    ├── japan_ha/
    │   ├── __init__.py
    │   ├── __pycache__/
    │   ├── cloud/
    │   │   └── __init__.py
    │   ├── components/
    │   │   ├── __init__.py
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   ├── data_validation.py
    │   │   ├── model_trainer.py
    │   │   └── __pycache__/
    │   ├── constant/
    │   │   ├── __init__.py
    │   │   ├── __pycache__/
    │   │   └── training_pipeline/
    │   │       ├── __init__.py
    │   │       └── __pycache__/
    │   ├── entity/
    │   │   ├── __init__.py
    │   │   ├── artifacts_entity.py
    │   │   ├── config_entity.py
    │   │   └── __pycache__/
    │   ├── exception/
    │   │   ├── __init__.py
    │   │   ├── exception.py
    │   │   └── __pycache__/
    │   ├── logging/
    │   │   ├── __init__.py
    │   │   ├── logger.py
    │   │   └── __pycache__/
    │   ├── pipeline/
    │   │   ├── __init__.py
    │   │   ├── batch_prediction.py
    │   │   ├── training_pipeline.py
    │   │   └── __pycache__/
    │   └── utils/
    │       ├── __init__.py
    │       ├── __pycache__/
    │       ├── main_utils/
    │       │   ├── __init__.py
    │       │   ├── utils.py
    │       │   └── __pycache__/
    │       └── ml_utils/
    │           ├── __init__.py
    │           ├── __pycache__/
    │           ├── metric/
    │           │   ├── __init__.py
    │           │   ├── classification_matrics.py
    │           │   └── __pycache__/
    │           └── model/
    │               ├── __init__.py
    │               ├── estimator.py
    │               └── __pycache__/
    ├── notebook/
    │   ├── EDAandMODELTraining.ipynb
    │   └── japan_heart_attack_dataset.csv
    ├── templates/
    │   └── table.html
    └── valid_data/
        └── test.csv
```
</details>
---

## 🚀 Features

- **Data Ingestion**: Fetch data from MongoDB for training.
- **Data Preprocessing**: Handle missing values, encode categorical features, and scale numerical features.
- **Imbalanced Data Handling**: Use **SMOTE** to balance the dataset.
- **Model Training**: Train machine learning models using the preprocessed data.
- **Model Deployment**: Deploy the model using **Docker**, **Amazon ECR**, and **EC2**.
- **Streamlit App**: A beautiful and interactive web app for predictions.

---

## 🛠️ Technologies Used

- **Python**: Primary programming language.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning models and preprocessing.
- **MongoDB**: Database for storing and retrieving data.
- **Docker**: Containerization for deployment.
- **Amazon S3**: Storage for datasets and model artifacts.
- **Amazon ECR & EC2**: Deployment and hosting.
- **Streamlit**: Web app for user interaction.

---

## 📊 Dataset

The dataset contains health-related features such as:
- **Age**, **Cholesterol Level**, **BMI**, **Heart Rate**, **Blood Pressure**, etc.
- **Categorical Features**: Gender, Smoking History, Diabetes History, etc.

The dataset is stored in MongoDB and preprocessed using **SMOTE** to handle class imbalance.

---

## 🐳 Docker Deployment

The project is containerized using Docker. Here’s how to build and run the Docker image:

```bash
# Build the Docker image
docker build -t japan_heart_attack .

# Run the Docker container
docker run -p 8501:8501 japan_heart_attack
```

---

## ☁️ Cloud Deployment

The model is deployed on **Amazon Web Services (AWS)**:
- **Amazon S3**: Used to store datasets and model artifacts. [Add S3 Link Here]
- **Amazon ECR**: Docker images are pushed to ECR. [Add ECR Link Here]
- **Amazon EC2**: The app is hosted on an EC2 instance. [Add EC2 Link Here]

---

## 📝 How to Run Locally

### **Step 1: Set Up the Environment**

1. Create a virtual environment using `conda`:
   ```bash
   conda create -n venv python=3.8
   conda activate venv/
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### **Step 2: Run the Streamlit App**

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser and go to `http://localhost:8501`.

---

## 📸 Proof of Work

Here are some screenshots of the project in action:

- **Streamlit App**: [Add Screenshot Link Here]
- **Docker Build**: [Add Screenshot Link Here]
- **AWS Deployment**: [Add Screenshot Link Here]

---

## 🙏 Acknowledgments

- Thanks to the creators of the dataset.
- Special thanks to the open-source community for their amazing tools and libraries.

---

Made with ❤️ by [**Rishabh Deshpande**].  
📧 Contact: [rishabhmse@gmail.com]  
🔗 GitHub: [https://github.com/de5hpande]

---