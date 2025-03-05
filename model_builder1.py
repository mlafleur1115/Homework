from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# Importing the parent: DataPreprocessing class from data_preprocess.py
from src.data_preprocess import DataPreprocessing

class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def nn(self, X_train, X_test, y_train, y_test):
        # Create MLP (Multi-Layer Perceptron) Classifier
        mlp_classifier = MLPClassifier(hidden_layer_sizes=1250, learning_rate_init=0.00001 max_iter=1250, random_state=42)

        # Train the model
        mlp_classifier.fit(X_train, y_train)

        # Test the model
        nn_predicted = mlp_classifier.predict(X_test)

        # Calculate accuracy
        total_accuracy = accuracy_score(y_test, nn_predicted)

        # Get performance
        self.accuracy = total_accuracy

        return mlp_classifier