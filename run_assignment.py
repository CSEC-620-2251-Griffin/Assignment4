import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier

# --- Performance Metrics Helper ---

def calculate_accuracy(y_true, y_pred):
    """
    Calculates the multiclass classification accuracy.
    Accuracy = C / (C + W) where C is correct, W is incorrect.
    
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        
    Returns:
        float: Accuracy score.
    """
    return accuracy_score(y_true, y_pred)

def run_experiment(X_train, X_test, y_train, y_test, model):
    """
    Trains and tests a model, measuring execution time and performance.
    
    Args:
        X_train, X_test, y_train, y_test: Data splits.
        model: The classifier model (DT or RF).
        
    Returns:
        dict: Results dictionary.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    accuracy = calculate_accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    return {
        'model_name': model.__class__.__name__,
        'accuracy': accuracy,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

# --- Plotting Functions ---

def plot_confusion_matrix(cm, labels, title="Normalized Confusion Matrix"):
    """
    Generates and displays a plot of the normalized confusion matrix.
    Saves the plot as 'confusion_matrix.png'.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("--- Plot saved: confusion_matrix.png ---")
    plt.close() # Close the plot to prevent display issues in some environments

def plot_feature_importance(importance_dict, n_top=10, title="Top 10 Feature Importance (Random Forest)"):
    """
    Generates and displays a bar plot of the top N feature importance scores.
    Saves the plot as 'feature_importance.png'.
    """
    # Sort and select top N features
    sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:n_top]
    features = [f"F_{idx}" for idx, _ in sorted_importance]
    scores = [score for _, score in sorted_importance]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(features, scores, color='skyblue')
    ax.set_xlabel("Normalized Gini Importance")
    ax.set_ylabel("Feature Index")
    ax.set_title(title)
    ax.invert_yaxis() # Highest importance at the top
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("--- Plot saved: feature_importance.png ---")
    plt.close() # Close the plot

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Assignment 4: IoT Classification Starter Script ---")
    print("Loading synthetic dataset to simulate IoT network data...")
    
    # Using a synthetic dataset to simulate the process, as the actual data is not available.
    # The actual assignment would involve loading the processed data provided by the instructor.
    X, y = make_classification(
        n_samples=5000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=0, 
        n_classes=8, # Simulating 8 different IoT device classes
        random_state=42
    )
    
    # Use 70/30 split, standard practice.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}\n")
    
    # --- Model Configuration ---
    # Default parameters chosen for a quick, representative run.
    # These should be tuned as required by the assignment (Report Q1)
    DT_PARAMS = {'max_depth': 10, 'min_node': 5}
    RF_PARAMS = {
        'n_trees': 50, 
        'data_frac': 0.8, 
        'feature_subcount': int(np.sqrt(X.shape[1])), # Standard rule of thumb: sqrt(n_features)
        'max_depth': 15, 
        'min_node': 5
    }

    # --- 1. Decision Tree Experiment ---
    dt_classifier = DecisionTreeClassifier(**DT_PARAMS)
    dt_results = run_experiment(X_train, X_test, y_train, y_test, dt_classifier)

    # --- 2. Random Forest Experiment ---
    rf_classifier = RandomForestClassifier(**RF_PARAMS)
    rf_results = run_experiment(X_train, X_test, y_train, y_test, rf_classifier)

    # --- Results Summary (Report Q2) ---
    print("\n" + "="*50)
    print("CLASSIFIER PERFORMANCE COMPARISON (Report Q2)")
    print("="*50)

    for results in [dt_results, rf_results]:
        print(f"\nModel: {results['model_name']}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Training Time (s): {results['fit_time']:.3f}")
        print(f"  Prediction Time (s): {results['predict_time']:.3f}")
        
    print("\n--- Comparative Analysis (Report Q2.i & Q2.ii) ---")
    if rf_results['accuracy'] > dt_results['accuracy']:
        print("The Random Forest model is expected to outperform the single Decision Tree,")
        print("as it leverages ensemble methods (bootstrapping and feature randomness) to")
        print("reduce variance and prevent overfitting (bagging).")
    else:
        print("The single Decision Tree performed better in this run (unlikely but possible).")
        
    print("\nExecution Time Comparison:")
    print("Random Forest fit time is significantly longer because it trains 50 individual trees.")
    print("Prediction time is also longer, as it requires prediction from 50 trees and a voting process.")

    # --- Plot Generation (Report Q2.b and Q1.b) ---
    
    # Confusion Matrix Plot (Report Q2.b)
    labels = np.unique(y) # Class labels (0 to 7 in this synthetic example)
    plot_confusion_matrix(rf_results['confusion_matrix'], labels=labels, title="RF Normalized Confusion Matrix (Report Q2.b)")
    
    # Feature Importance Plot (Report Q1.b)
    plot_feature_importance(rf_classifier.get_feature_importance(), n_top=10, title="RF Top 10 Feature Importance (Report Q1.b)")

    # --- Node Importance and Gini Info (Report Q4) ---
    print("\n" + "="*50)
    print("NODE IMPORTANCE INFORMATION (Report Q4)")
    print("="*50)
    # Find and print the information for one non-trivial node in the *first* tree of the forest
    # This prints the required values for Gini computation and Node Importance calculation.
    rf_classifier.trees[0].print_node_info()

    # Hint for Report Q2.d:
    print("\nHint for Report Q2.d: Review 'confusion_matrix.png' for off-diagonal elements (misclassifications).")
    print("Use the IoT device names from 'list_of_devices.txt' (not provided) to explain WHY (e.g., two devices have similar traffic patterns).")
    
    print("\nExecution complete. Check for the generated .png files for your report.")