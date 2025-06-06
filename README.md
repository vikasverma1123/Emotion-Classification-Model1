
# 🚀 Tweet Emotion Classification

This repository contains a project that classifies tweets into six emotion categories: **sadness**, **joy**, **love**, **anger**, **fear**, and **surprise**. The project leverages natural language processing and deep learning techniques to analyze tweet content, extracting key features from the text and predicting the underlying emotion.

## 🔍 Project Overview
- **Objective:**  
  Develop a robust classifier to automatically identify the emotion conveyed in tweets.
- **Approach:**  
  The model uses a simple yet effective architecture consisting of an embedding layer and a bidirectional GRU layer. Despite experimenting with more complex architectures, the final model—chosen for its balance between simplicity and performance—achieved a test accuracy of **93.47%** with a test loss of **0.1047**.
- **Insights:**  
  In addition to classification, the project includes a detailed analysis of unique word distributions across emotions, providing insights into the linguistic nuances of social media text.

## 🛠️ Installation & Setup
To get started with the project, clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

**Requirements:**  
- Python 3.x  
- TensorFlow  
- Keras  
- Pandas  
- NumPy  
- Scikit-learn  
- Seaborn  
- Matplotlib

## 💻 Usage
- **Data Preprocessing:**  
  The project includes scripts to clean and preprocess tweet data, including tokenization, padding, and building vocabulary.
- **Model Training:**  
  The core model is defined in the notebook, where you can run the training process using:
  ```python
  history = base_model.fit(X_train_padded, y_train, epochs=4, batch_size=1500, validation_data=(X_test_padded, y_test))
  ```
- **Evaluation & Analysis:**  
  After training, the model is evaluated using metrics such as accuracy and loss. A confusion matrix and detailed word frequency analysis further help to understand the performance and nuances of the classifier.

## 📊 Results
- **Test Loss:** 0.1047  
- **Test Accuracy:** 93.47%  
- **Confusion Matrix Analysis:**  
  The confusion matrix shows strong performance on emotions like sadness and joy, with some expected misclassifications between similar positive (joy vs. love) and negative (anger vs. fear) emotions.

## 📁 Repository Structure
```
tweet-emotion-classification/
│
├── emotion-classification-of-tweets.ipynb  # Main Project file
├── text.csv                                # Data file
└── README.md                               # This file
```

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📫 Contact
For any questions or collaboration opportunities, please contact [John Pospisil](mailto:john@johnpospisil.com).\
Feel free to explore, contribute, and use this project as a foundation for further research in natural language processing and emotion detection!
```
