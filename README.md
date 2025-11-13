# Assignment 4: Custom IoT Classification (Decision Tree & Random Forest)

This repository contains a custom implementation of the Decision Tree (DT) and Random Forest (RF) classifiers from scratch, as required for CSEC 520/620 Assignment 4.

The implementation avoids using scikit-learn's built-in DT/RF classifiers or Gini impurity computation, adhering to the assignment's core task of custom implementation.

## Project Structure

* `decision_tree.py`: Contains the `Node` class and the `DecisionTreeClassifier` implementation (Gini impurity, splitting, tree building).

* `random_forest.py`: Implements the `RandomForestClassifier` ensemble method (bootstrapping, feature subsetting, voting).

* `run_assignment.py`: The main script to load the data, configure hyperparameters, train and test the models, and generate the required analysis output.

* `requirements.txt`: Python package dependencies.

**NOTE:** The actual IoT dataset (`iot_data` folder) and the original `classify.py` file mentioned in the assignment PDF were not provided. `run_assignment.py` uses a **synthetic multiclass dataset** (`sklearn.datasets.make_classification`) to demonstrate the functionality and structure. **You must replace the data loading logic in `run_assignment.py` with the actual data loading from your provided assignment files.**

## Setup Instructions

1. **Environment Setup:** Ensure you have Python 3 installed.

2. **Install Dependencies:** Install the required packages using pip:

pip install -r requirements.txt

## How to Run the Code

Run the main execution script from your terminal:

python run_assignment.py

### Expected Output

The script will:

1. Load the synthetic dataset.

2. Train the custom `DecisionTreeClassifier`.

3. Train the custom `RandomForestClassifier`.

4. Print the performance metrics, execution times, and a normalized confusion matrix (as required for Report Q2).

5. Print the specific Gini Impurity and Node Importance calculation information from a non-trivial node in the first tree (as required for Report Q4).

## Hyperparameter Tuning

The parameters for both models are defined in `run_assignment.py` under `DT_PARAMS` and `RF_PARAMS`. To complete the Report Q1 requirement, you will need to modify these parameters and run experiments to observe the impact on performance.

Example parameters in run_assignment.py
DT_PARAMS = {'max_depth': 10, 'min_node': 5} RF_PARAMS = { 'n_trees': 50, 'data_frac': 0.8, 'feature_subcount': int(np.sqrt(X.shape[1])), 'max_depth': 15, 'min_node': 5 }