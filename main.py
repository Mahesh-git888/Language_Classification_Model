import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk
import matplotlib.pyplot as plt
import unicodedata
import os
from sklearn.preprocessing import OneHotEncoder
import pickle

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Load dataset
dataset_path = 'training_dataset.csv'
data = pd.read_csv(dataset_path)
print("Columns in dataset:", data.columns)

# Ensure 'text' and 'language' columns exist in dataset
if 'text' not in data.columns or 'language' not in data.columns:
    raise KeyError("Please check that 'text' and 'language' columns are present in the dataset.")

# Handle missing or non-string values in the 'text' column
data['text'] = data['text'].astype(str).fillna('')

# Define stopwords and vowels for multiple languages
stopword_sets = {
    'bn': set(stopwords.words('bengali') if 'bengali' in stopwords.fileids() else []),
    'en': set(stopwords.words('english') if 'english' in stopwords.fileids() else []),
    'gu': set(stopwords.words('gujarati') if 'gujarati' in stopwords.fileids() else []),
    'hi': set(stopwords.words('hindi') if 'hindi' in stopwords.fileids() else []),
    'kn': set(stopwords.words('kannada') if 'kannada' in stopwords.fileids() else []),
    'kok': set(['आहे', 'असे', 'आणि', 'पण', 'म्हणजे']),  # Example Konkani stopwords
    'ks': set(['کھ', 'ہے', 'اور', 'لیکن']),  # Example Kashmiri stopwords
    'ml': set(stopwords.words('malayalam') if 'malayalam' in stopwords.fileids() else []),
    'mr': set(stopwords.words('marathi') if 'marathi' in stopwords.fileids() else []),
    'ne': set(['छ', 'वा', 'र', 'तर', 'त', 'मा', 'देखि']),  # Example Nepali stopwords
    'or': set(stopwords.words('oriya') if 'oriya' in stopwords.fileids() else []),
    'pa': set(['ਤੇ', 'ਹੈ', 'ਅਤੇ', 'ਜੋ', 'ਨਹੀਂ']),  # Example Punjabi stopwords
    'sa': set(['च', 'त', 'हि', 'एव', 'वा']),  # Example Sanskrit stopwords
    'ta': set(stopwords.words('tamil') if 'tamil' in stopwords.fileids() else []),
    'te': set(stopwords.words('telugu') if 'telugu' in stopwords.fileids() else []),
    'ur': set(stopwords.words('urdu') if 'urdu' in stopwords.fileids() else []),
}

vowel_sets = {
    'bn': 'অআইঈউঊঋএঐওঔ',
    'en': 'aeiouAEIOU',
    'gu': 'અઆઇઈઉઊઋએઐઓઔ',
    'hi': 'अआइईउऊऋएऐओऔ',
    'kn': 'ಅಆಇಈಉಊಋಎಏಐಒಔ',
    'kok': 'अआइईउऊऋएऐओऔ',
    'ks': 'اےاؤ',
    'ml': 'അആഇഈഉഊഋഎഏഐഒഔ',
    'mr': 'अआइईउऊऋएऐओऔ',
    'ne': 'अआइईउऊऋएऐओऔ',
    'or': 'ଅଆଇଈଉଊଋଏଐଓଔ',
    'pa': 'ਅਆਇਈਉਊਏਐਓਔ',
    'sa': 'अआइईउऊऋएऐओऔ',
    'ta': 'அஆஇஈஉஊஎஏஐஒஓஔ',
    'te': 'అఆఇఈఉఊఎఏఐఒఓఔ',
    'ur': 'اےاؤ',
}

# Feature Extraction Functions
def average_word_length(text):
    words = word_tokenize(text)
    return np.mean([len(word) for word in words]) if words else 0

def average_sentence_length(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0

def calculate_stopword_count(text, language):
    words = word_tokenize(text)
    stopword_set = stopword_sets.get(language, set())
    return sum(1 for word in words if word.lower() in stopword_set)

def calculate_vowel_count(text, language):
    vowels = vowel_sets.get(language, '')
    return sum(1 for char in text if char in vowels)

def calculate_diacritic_frequency(text):
    return sum(1 for char in text if unicodedata.category(char) == 'Mn')

def character_trigram_count(text):
    return len([text[i:i+3] for i in range(len(text) - 2)])

def calculate_prefix_suffix_frequency(words):
    prefixes = [word[:3] for word in words if len(word) >= 3]
    suffixes = [word[-3:] for word in words if len(word) >= 3]
    return len(prefixes), len(suffixes)

def script_based_feature(text):
    if re.search('[\u0980-\u09FF]', text):
        return "bn"
    elif re.search('[\u0A80-\u0AFF]', text):
        return "gu"
    elif re.search('[\u0900-\u097F]', text):
        return "hi"
    elif re.search('[\u0C80-\u0CFF]', text):
        return "kn"
    elif re.search('[\u0900-\u097F]', text) and re.search('[\u0900-\u091F]', text):  # Konkani
        return "kok"
    elif re.search('[\u0600-\u06FF]', text):
        return "ks"
    elif re.search('[\u0D00-\u0D7F]', text):
        return "ml"
    elif re.search('[\u0900-\u097F]', text):
        return "mr"
    elif re.search('[\u0900-\u097F]', text):
        return "ne"
    elif re.search('[\u0B00-\u0B7F]', text):
        return "or"
    elif re.search('[\u0A00-\u0A7F]', text):
        return "pa"
    elif re.search('[\u0900-\u097F]', text):
        return "sa"
    elif re.search('[\u0B80-\u0BFF]', text):
        return "ta"
    elif re.search('[\u0C00-\u0C7F]', text):
        return "te"
    elif re.search('[\u0600-\u06FF]', text):
        return "ur"
    elif re.search('[\u0980-\u09FF]', text):
        return "as"
    elif re.search('[a-zA-Z]', text):
        return "en"
    else:
        return "other"

def calculate_pos_ratios(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    pos_counts = pd.Series([tag for _, tag in tagged])
    noun_ratio = sum(pos_counts.str.startswith('NN')) / len(words) if len(words) else 0
    verb_ratio = sum(pos_counts.str.startswith('VB')) / len(words) if len(words) else 0
    return noun_ratio, verb_ratio

# Extract Features for Each Row
def extract_features(row):
    text = row['text']
    language = row['language']
    words = word_tokenize(text)

    return {
        'avg_word_length': average_word_length(text),
        'avg_sentence_length': average_sentence_length(text),
        'stopword_count': calculate_stopword_count(text, language),
        'vowel_count': calculate_vowel_count(text, language),
        'diacritic_freq': calculate_diacritic_frequency(text),
        'trigram_count': character_trigram_count(text),
        'prefix_freq': calculate_prefix_suffix_frequency(words)[0],
        'suffix_freq': calculate_prefix_suffix_frequency(words)[1],
        'script': script_based_feature(text),
        'noun_ratio': calculate_pos_ratios(text)[0],
        'verb_ratio': calculate_pos_ratios(text)[1]
    }

def training_phase():
        # Apply Feature Extraction
        features_df = pd.DataFrame(data.apply(extract_features, axis=1).tolist())

        # One-Hot Encode the 'script' feature
        encoder = OneHotEncoder(sparse_output=False)
        script_encoded = encoder.fit_transform(features_df[['script']].values.reshape(-1, 1))  # Ensure it's a 2D array

        # Add the encoded features to the features dataframe
        script_df = pd.DataFrame(script_encoded, columns=encoder.categories_[0])
        features_df = pd.concat([features_df, script_df], axis=1)

        # Prepare Data for Model
        X = features_df.drop('script', axis=1)  # Drop the original 'script' column as it's now encoded
        y = data['language']  # Target variable is 'language'

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train) 



        ## For testing with unseen data 


        test_dataset_path = 'test_dataset.csv'
        test_data = pd.read_csv(test_dataset_path)

        # Ensure 'text' and 'language' columns exist in test dataset
        if 'text' not in test_data.columns or 'language' not in test_data.columns:
            raise KeyError("Test dataset must contain 'text' and 'language' columns.")

        # Handle missing or non-string values in the 'text' column
        test_data['text'] = test_data['text'].astype(str).fillna('')

        # Extract features for the test dataset
        features_test_df = pd.DataFrame(test_data.apply(extract_features, axis=1).tolist())


        with open("encoder.pkl", "rb") as encoder_file:
            encoder = pickle.load(encoder_file)

        # One-Hot Encode the 'script' feature in the test dataset
        script_encoded_test = encoder.transform(features_test_df[['script']].values.reshape(-1, 1))  # Use the fitted encoder
        script_df_test = pd.DataFrame(script_encoded_test, columns=encoder.categories_[0])

        # Add the encoded 'script' features to the test feature dataframe
        features_test_df = pd.concat([features_test_df, script_df_test], axis=1)

        # Prepare data for prediction (drop 'script' column as it is now encoded)
        X_test_final = features_test_df.drop('script', axis=1)
        y_test_final = test_data['language']  # Target variable

        # Predict using the trained model
        y_pred_test = model.predict(X_test_final)



        from sklearn.metrics import classification_report
        print("Classification Report on Final Test Data:")
        print(classification_report(y_test_final, y_pred_test))

        # Optionally, show confusion matrix
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm_test = confusion_matrix(y_test_final, y_pred_test, labels=model.classes_)
        ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=model.classes_).plot(cmap='Blues')
        plt.show()

        with open("mahesh_forest.pkl", "wb") as f:

            pickle.dump(model, f)

        with open("encoder.pkl", "wb") as encoder_file:
            pickle.dump(encoder, encoder_file)

        print("Successfully saved the model and encoder")

    

        # Evaluate Model
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot(cmap='Blues')
        plt.show() 



import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize

def preprocess_input(text):
    """Preprocess the input text to extract features."""
    # Use the same feature extraction functions as during training
    words = word_tokenize(text)
    features = {
        'avg_word_length': average_word_length(text),
        'avg_sentence_length': average_sentence_length(text),
        'stopword_count': calculate_stopword_count(text, "en"),  # Default language assumption for stopwords
        'vowel_count': calculate_vowel_count(text, "en"),        # Default language assumption for vowels
        'diacritic_freq': calculate_diacritic_frequency(text),
        'trigram_count': character_trigram_count(text),
        'prefix_freq': calculate_prefix_suffix_frequency(words)[0],
        'suffix_freq': calculate_prefix_suffix_frequency(words)[1],
        'script': script_based_feature(text),
        'noun_ratio': calculate_pos_ratios(text)[0],
        'verb_ratio': calculate_pos_ratios(text)[1]
    }

    # Convert features into a DataFrame
    features_df = pd.DataFrame([features])

    with open("encoder.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    # One-hot encode the 'script' feature
    script_encoded = encoder.transform(features_df[['script']].values.reshape(-1, 1))
    script_df = pd.DataFrame(script_encoded, columns=encoder.categories_[0])


    # Add the encoded features and drop the original 'script' column
    features_df = pd.concat([features_df.drop('script', axis=1), script_df], axis=1)

    return features_df

def predict_language(model, text):
    """Predict the language of the input text using the trained model."""
    # Preprocess the input text
    features = preprocess_input(text)

    # Predict the language
    prediction = model.predict(features)
    return prediction[0]  # Return the predicted class 

def predicting_phase():
    with open("mahesh_forest.pkl", "rb") as f:
        

        load_model = pickle.load(f)
    # Predict an example
    k = input("Enter a text sample:")
    predicted_language = predict_language(load_model, k)
    print(f"The predicted language for the text '{k}' is: {predicted_language}")


if __name__ == "__main__":
    training_phase()
    predicting_phase()


    
