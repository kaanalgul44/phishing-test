"""
Phishing Email Detection System - MVP
Required Libraries:
pip install pandas numpy scikit-learn nltk
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class PhishingDetector:
    """Main class for phishing email detection"""
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the detector with specified model type
        Args:
            model_type: 'naive_bayes', 'logistic_regression', or 'svm'
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.7)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Clean and preprocess email text
        Args:
            text: Raw email text
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Apply stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df, text_column='text', label_column='label'):
        """
        Prepare data for training
        Args:
            df: DataFrame with email data
            text_column: Name of text column
            label_column: Name of label column (0=legitimate, 1=phishing)
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Preprocess all texts
        print("Preprocessing emails...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Split data
        X = df['processed_text']
        y = df[label_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text data
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the selected model
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"Training {self.model_type} model...")
        
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError("Invalid model type")
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        Args:
            X_test: Test features
            y_test: Test labels
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model performance...")
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        print("\n" + "="*50)
        print(f"Model: {self.model_type.upper()}")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Legitimate', 'Phishing']))
        
        return metrics
    
    def predict_email(self, email_text):
        """
        Predict if an email is phishing or legitimate
        Args:
            email_text: Raw email text
        Returns:
            Prediction and confidence score
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess the email
        processed_text = self.preprocess_text(email_text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        return {
            'prediction': 'PHISHING' if prediction == 1 else 'LEGITIMATE',
            'confidence': max(probability) * 100,
            'phishing_probability': probability[1] * 100 if len(probability) > 1 else 0
        }

def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    In production, use a real dataset like:
    - SpamAssassin Public Corpus
    - Enron Email Dataset
    - CSDMC2010 SPAM corpus
    """
    data = {
        'text': [
            # Phishing examples
            "Urgent! Your account will be suspended. Click here immediately to verify your identity and prevent closure.",
            "Congratulations! You've won $1,000,000! Click this link now to claim your prize before it expires!",
            "Your PayPal account has been limited. Please confirm your identity by clicking here within 24 hours.",
            "IRS Tax Refund: You are eligible for a tax refund of $3,458.23. Please submit your bank information here.",
            "Alert: Suspicious activity on your account. Verify your password immediately to prevent unauthorized access.",
            "Your package delivery failed. Click here to reschedule and provide payment information.",
            "Netflix: Your payment method has expired. Update now to continue watching your favorite shows.",
            "Bank of America: Security alert! Unusual sign-in detected. Verify your account now.",
            
            # Legitimate examples
            "Hi John, Following up on our meeting yesterday about the Q3 marketing strategy. Please review the attached document.",
            "Team meeting scheduled for tomorrow at 2 PM in conference room B. Please confirm your attendance.",
            "Your monthly AWS bill for December 2023 is now available. You can view it in your billing dashboard.",
            "Thank you for your recent purchase. Your order #12345 has been shipped and will arrive in 3-5 business days.",
            "Reminder: Your dentist appointment is scheduled for next Tuesday at 10 AM. Reply to confirm.",
            "Weekly newsletter: Check out our latest blog posts about Python programming and machine learning.",
            "Project update: The development team has completed the first phase. Please review the progress report.",
            "Happy birthday! Wishing you a wonderful day filled with joy and celebration. Best regards, Sarah."
        ],
        'label': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 1=phishing, 0=legitimate
    }
    
    return pd.DataFrame(data)

def interactive_prediction(detector):
    """
    Interactive console interface for email prediction
    Args:
        detector: Trained PhishingDetector instance
    """
    print("\n" + "="*60)
    print("PHISHING EMAIL DETECTION - INTERACTIVE MODE")
    print("="*60)
    print("Enter 'quit' to exit\n")
    
    while True:
        print("-" * 60)
        email_text = input("\nEnter email text to analyze (or 'quit' to exit):\n> ")
        
        if email_text.lower() == 'quit':
            print("Exiting interactive mode...")
            break
        
        if len(email_text.strip()) < 10:
            print("Please enter a longer email text for analysis.")
            continue
        
        try:
            result = detector.predict_email(email_text)
            print("\n" + "="*40)
            print(f"🎯 Prediction: {result['prediction']}")
            print(f"📊 Confidence: {result['confidence']:.2f}%")
            print(f"⚠️  Phishing Probability: {result['phishing_probability']:.2f}%")
            
            if result['prediction'] == 'PHISHING':
                print("\n⛔ WARNING: This email appears to be PHISHING!")
                print("Do not click any links or provide personal information.")
            else:
                print("\n✅ This email appears to be LEGITIMATE.")
                print("However, always exercise caution with unexpected emails.")
            print("="*40)
            
        except Exception as e:
            print(f"Error analyzing email: {e}")

def main():
    """
    Main function to run the MVP
    """
    print("="*60)
    print("PHISHING EMAIL DETECTION SYSTEM - MVP")
    print("="*60)
    
    # Step 1: Load or create dataset
    print("\n1. Loading dataset...")
    df = create_sample_dataset()
    print(f"Dataset loaded: {len(df)} emails ({sum(df['label'])} phishing, {len(df)-sum(df['label'])} legitimate)")
    
    # Note: For production, load a real dataset like this:
    # df = pd.read_csv('phishing_dataset.csv')
    
    # Step 2: Initialize detector
    print("\n2. Initializing detector...")
    detector = PhishingDetector(model_type='naive_bayes')
    
    # Step 3: Prepare data
    print("\n3. Preparing data...")
    X_train, X_test, y_train, y_test = detector.prepare_data(df)
    
    # Step 4: Train model
    print("\n4. Training model...")
    detector.train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    print("\n5. Evaluating model...")
    metrics = detector.evaluate_model(X_test, y_test)
    
    # Step 6: Compare multiple models (optional)
    print("\n6. Comparing different models...")
    models = ['naive_bayes', 'logistic_regression', 'svm']
    results = {}
    
    for model_type in models:
        detector_comp = PhishingDetector(model_type=model_type)
        X_train, X_test, y_train, y_test = detector_comp.prepare_data(df)
        detector_comp.train_model(X_train, y_train)
        results[model_type] = detector_comp.evaluate_model(X_test, y_test)
    
    # Step 7: Interactive prediction
    print("\n7. Starting interactive mode...")
    interactive_prediction(detector)

if __name__ == "__main__":
    main()