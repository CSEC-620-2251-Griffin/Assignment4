import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from itertools import product
from classify import load_iot_dataset

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

# --- Hyperparameter Tuning Function ---
def tune_model(model_cls, param_grid, X_train, y_train, X_val, y_val, model_name="model"):
    history = []  # <-- list of {"params":..., "acc":...}

    keys = list(param_grid.keys())
    best_params = None
    best_acc = -1

    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))

        model = model_cls(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = calculate_accuracy(y_val, y_pred)

        # record
        history.append({"params": params, "acc": acc})

        print(f"{model_name} params={params} -> val acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = params

    print(f"\nBEST {model_name}: {best_params}  ACC={best_acc:.4f}\n")

    return best_params, history


# --- Hyperparameter Impact Plotting Function, Exclusively for Q1 ---
def plot_hyperparameter_impact(history, model_name="Model"):
    """
    history = list of {"params": {...}, "acc": float}
    Saves PNG: hyperparameter_impact_<modelname>.png
    """

    # Convert history to dataframe
    rows = []
    for entry in history:
        row = entry["params"].copy()
        row["accuracy"] = entry["acc"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("accuracy", ascending=False)

    # Build labels for each bar (multiline)
    labels = df_sorted.apply(
        lambda r: "\n".join([f"{k}={r[k]}" for k in r.index if k != "accuracy"]),
        axis=1
    )

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df_sorted)), df_sorted["accuracy"])
    plt.xticks(range(len(df_sorted)), labels, rotation=70, ha="right")

    plt.title(f"Hyperparameter Impact on Validation Accuracy ({model_name})")
    plt.ylabel("Validation Accuracy")
    plt.tight_layout()

    # --- SAVE PNG ---
    filename = f"hyperparameter_impact_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    print(f"[Saved] {filename}")

    # Show it
    plt.show()

# --- Experiment Execution Function ---
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
    print("--- Loading IoT Dataset ---")

    DATA_ROOT = "./iot_data"   # path to folder containing the dataset files

    X, y, label_encoder = load_iot_dataset(DATA_ROOT)

    # Train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20,
        random_state=42, stratify=y_train_full
    )

    print("Loaded classes:", len(label_encoder.classes_))
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    
    # Hyperparameter tuning can be done here if desired
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Validation split for hyperparameter tuning
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
    print(f"Training Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}\n")
    
    # --- Model Configuration ---
    # These are set via hyperparameter tuning
    """ Uncomment to perform hyperparameter tuning for Decision Tree
    dt_param_grid = {
        "max_depth": [5, 10, 15],
        "min_node": [2, 5, 10]}

    best_dt_params, dt_history = tune_model(
        DecisionTreeClassifier,
        dt_param_grid,
        X_train, y_train,
        X_val, y_val,
        model_name="Decision Tree")
    """

    # Comment out if uncommenting tuning above
    best_dt_params = {"max_depth": 10, "min_node": 5}

    # Random Forest search space
    n_features = X.shape[1]

    """ Uncomment to perform hyperparameter tuning for Random Forest
    rf_param_grid = {
        "n_trees": [20, 50, 100],
        "data_frac": [0.6, 0.8],
        "feature_subcount": [int(np.sqrt(n_features)), n_features // 2],
        "max_depth": [10, 15, 20],
        "min_node": [2, 5, 10]}
    

    best_rf_params, rf_history = tune_model(
        RandomForestClassifier,
        rf_param_grid,
        X_train, y_train,
        X_val, y_val,
        model_name="Random Forest")
    """

    # Comment out if uncommenting tuning above
    best_rf_params = {
        'n_trees': 100,
        'data_frac': 0.6,
        'feature_subcount': int(np.sqrt(n_features)),
        'max_depth': 20,
        'min_node': 2
    }

    # --- 1. Decision Tree Experiment ---
    dt_classifier = DecisionTreeClassifier(**best_dt_params)
    dt_results = run_experiment(X_train, X_test, y_train, y_test, dt_classifier)

    # --- 2. Random Forest Experiment ---
    rf_classifier = RandomForestClassifier(**best_rf_params)
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
    labels = label_encoder.classes_
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
    
    """ uncomment to generate hyperparameter impact plots
    plot_hyperparameter_impact(dt_history, "Decision Tree")
    plot_hyperparameter_impact(rf_history, "Random Forest")
    """

    print("\nExecution complete. Check for the generated .png files for your report.")
