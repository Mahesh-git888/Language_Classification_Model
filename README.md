# Language_Classification_Model



# README for Language Identification System

## Overview
This repository provides a language identification system that predicts the language of a given text. The system uses a **Random Forest Classifier** trained on textual features such as average word length, stopword counts, script analysis, and more. The model is built using Python libraries such as `scikit-learn`, `NLTK`, `pandas`, and `numpy`. It also includes support for multiple Indian languages.

---

## Features
1. **Feature Extraction**: Extracts linguistic and script-based features for language detection.
2. **Multilingual Support**: Handles text in Indian languages like Hindi, Bengali, Tamil, Telugu, etc., along with English.
3. **Machine Learning Model**: Uses a **Random Forest Classifier** for prediction.
4. **Pre-trained Components**: Includes a saved model (`mahesh_forest.pkl`) and one-hot encoder (`encoder.pkl`).

---

## Requirements
Ensure you have the following installed on your system:

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `matplotlib`

You can install all the necessary libraries using:

```bash
pip install -r requirements.txt
```

---

## Repository Structure
- `training_dataset.csv`: The dataset used to train the model.
- `mahesh_forest.pkl`: The pre-trained Random Forest model.
- `encoder.pkl`: The pre-trained one-hot encoder for the `script` feature.
- `language_identifier.py`: Python script containing the training and prediction functions.
- `requirements.txt`: List of required Python libraries.

---

## How to Run the Project

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone "https://github.com/Mahesh-git888/Language_Classification_Model.git"
cd Language_Classification_Model
```

### 2. Prepare the Environment
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Training the Model (Optional)
If you want to retrain the model:
1. Place your dataset in the same folder and name it `training_dataset.csv`.
2. Run the script:
   ```bash
   python main.py
   ```
3. The script will:
   - Train the model on the dataset.
   - Save the trained model as `mahesh_forest.pkl` and the encoder as `encoder.pkl`.
   - Display evaluation metrics and a confusion matrix.

### 4. Predict Language
To predict the language of a text:
1. Ensure `mahesh_forest.pkl` and `encoder.pkl` are in the same folder as the script.
2. Run the script:
   ```bash
   python language_identifier.py
   ```
3. Enter a text sample when prompted, and the system will predict its language.

---

## Example Usage

1. **Input**: 
   ```
Enter a text sample:എങ്ങനെ
   ```

2. **Output**:
   ```
   The predicted language for the text 'എങ്ങനെ' is: ml
   ```

---

## Technical Details

### Features Used
1. **Average Word Length**: Mean length of words in the text.
2. **Average Sentence Length**: Mean number of words in a sentence.
3. **Stopword Count**: Number of stopwords in the text.
4. **Vowel Count**: Count of vowels based on the language.
5. **Diacritic Frequency**: Frequency of diacritical marks.
6. **Character Trigrams**: Count of trigrams in the text.
7. **POS Ratios**: Ratios of nouns and verbs in the text.
8. **Script Detection**: Identifies the script (Malayalam, telugu, english, etc.,).

### Model Details
- Algorithm: **Random Forest Classifier**
- Evaluation Metrics: **Classification Report**, **Confusion Matrix**

---

## Notes
1. Make sure to download the required NLTK resources when prompted.
2. If retraining, ensure your dataset contains `text` and `language` columns.
3. Use the `training_phase()` function to train the model and `predicting_phase()` to test it.

