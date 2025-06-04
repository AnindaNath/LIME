import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries

# TensorFlow and Keras for image classification
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Scikit-learn for text classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# LIME for explainability
from lime import lime_image
from lime.lime_text import LimeTextExplainer

# -----------------------------
# Image Classification Section
# -----------------------------

class ImageClassifier:
    def __init__(self, input_shape=(224, 224, 3)):
        """Initialize the image classifier with a pre-trained ResNet50 base."""
        # Load pre-trained ResNet50 model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        self.model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

    def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
        """Load and preprocess image with error handling."""
        try:
            img = keras_image.load_img(img_path, target_size=target_size)
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return preprocess_input(img_array)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None

    def predict_image(self, img_array):
        """Predict image class probabilities."""
        if img_array is None:
            return None
        return self.model.predict(img_array)

    def explain_prediction(self, img_path, num_samples=1000):
        """Generate LIME explanation for image prediction."""
        try:
            # Load and preprocess image
            img = keras_image.load_img(img_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            
            # Create LIME explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Generate explanation
            explanation = explainer.explain_instance(
                img_array.astype('double'),
                classifier_fn=lambda x: self.model.predict(preprocess_input(x)),
                top_labels=1,
                hide_color=0,
                num_samples=num_samples
            )
            
            # Get explanation mask
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            
            # Create visualization
            self.visualize_explanation(img_array, temp, mask, img_path)
            
            return explanation
            
        except Exception as e:
            print(f"Error generating explanation for {img_path}: {str(e)}")
            return None

    def visualize_explanation(self, original_img, explained_img, mask, img_path):
        """Visualize both original and explained images."""
        plt.figure(figsize=(12, 6))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_img / 255.0)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot explanation
        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(explained_img / 255.0, mask))
        plt.title('LIME Explanation')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize classifier
    classifier = ImageClassifier()
    
    # Get list of images from positive and negative folders
    data_dir = '.'
    for class_name in ['positive', 'negative']:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            print(f"\nProcessing {class_name} images:")
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    print(f"\nAnalyzing: {img_path}")
                    
                    # Load and predict
                    img_array = classifier.load_and_preprocess_image(img_path)
                    if img_array is not None:
                        pred = classifier.predict_image(img_array)
                        print(f"Prediction: {'Positive' if pred[0][0] > 0.5 else 'Negative'} "
                              f"(confidence: {pred[0][0]:.2f})")
                        
                        # Generate and show explanation
                        classifier.explain_prediction(img_path)

if __name__ == "__main__":
    main()

# -----------------------------
# Text Classification Section
# -----------------------------

# Sample dataset
texts = [
    "I feel sad and hopeless.",
    "Life is beautiful and full of joy.",
    "I'm feeling very low and depressed.",
    "Today is a wonderful day!"
]
labels = [1, 0, 1, 0]  # 1: Depressed, 0: Not Depressed

# Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train logistic regression classifier
text_model = LogisticRegression()
text_model.fit(X, labels)

# Define class names
class_names = ['Not Depressed', 'Depressed']

def predict_proba(texts):
    """Predict probabilities for text data."""
    return text_model.predict_proba(vectorizer.transform(texts))

def explain_text_prediction(text_instance):
    """Generate LIME explanation for text prediction."""
    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(
        text_instance,
        predict_proba,
        num_features=6
    )
    explanation.show_in_notebook(text=True)

# Example usage for text
sample_text = "I don't see any reason to keep going."
print(f"Predicted class: {class_names[text_model.predict(vectorizer.transform([sample_text]))[0]]}")
explain_text_prediction(sample_text)
