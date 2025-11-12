import numpy as np
from collections import Counter
from decision_tree import DecisionTreeClassifier
import pandas as pd # Used for efficient sampling

# --- Random Forest Classifier ---

class RandomForestClassifier:
    """
    Custom implementation of a Random Forest Classifier using custom Decision Trees.

    Parameters:
        n_trees (int): The number of decision trees in the forest.
        data_frac (float): The percentage of the dataset to use when building each tree (bootstrapping).
        feature_subcount (int): The number of features to sample when performing splits during tree building.
        max_depth (int): The maximum depth for each individual tree.
        min_node (int): The minimum number of samples allowed per leaf node for each tree.
    """
    def __init__(self, n_trees=100, data_frac=0.8, feature_subcount=None, max_depth=10, min_node=5):
        self.n_trees = n_trees
        self.data_frac = data_frac
        self.feature_subcount = feature_subcount
        self.max_depth = max_depth
        self.min_node = min_node
        self.trees = []
        self.feature_importance = {} # Aggregated importance


    def _sample_data(self, X, y):
        """
        Creates a random subset of the data with replacement (bootstrapping).
        
        Args:
            X (np.array): Full feature matrix.
            y (np.array): Full label array.
            
        Returns:
            tuple: (X_subset, y_subset)
        """
        n_samples = len(X)
        # Calculate the size of the subset
        subset_size = int(self.data_frac * n_samples)
        
        # Randomly sample indices with replacement
        indices = np.random.choice(n_samples, size=subset_size, replace=True)
        
        # Select the samples based on the indices
        X_subset = X[indices]
        y_subset = y[indices]
        
        return X_subset, y_subset


    def fit(self, X, y):
        """
        Builds n_trees decision trees, each trained on a bootstrapped sample
        and utilizing feature subsetting at each split.
        
        Args:
            X (np.array): Training feature matrix.
            y (np.array): Training labels.
        """
        self.trees = []
        self.feature_importance = {i: 0.0 for i in range(X.shape[1])}
        
        print(f"Building {self.n_trees} trees...")

        for i in range(self.n_trees):
            # 1. Build a subset of the data (bootstrapping)
            X_sample, y_sample = self._sample_data(X, y)
            
            # 2. Initialize a Decision Tree with ensemble parameters
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                min_node=self.min_node, 
                feature_subcount=self.feature_subcount
            )
            
            # 3. Train the tree (feature subsetting is handled inside the DT's _build_tree)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            # Aggregate feature importance
            for idx, importance in tree.get_feature_importance().items():
                self.feature_importance[idx] += importance
                
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{self.n_trees} trees.")
                
        print("Random Forest training complete.")


    def predict(self, X):
        """
        Predicts the class labels for a dataset X by voting from all trees.
        
        Args:
            X (np.array): Feature matrix to predict on.
            
        Returns:
            np.array: Array of predicted class labels.
        """
        # Get predictions from every single tree for all samples
        # shape: (n_trees, n_samples)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to get shape (n_samples, n_trees)
        tree_predictions = tree_predictions.T
        
        final_predictions = np.array([
            # For each sample, find the most common prediction (the mode/vote)
            Counter(sample_predictions).most_common(1)[0][0]
            for sample_predictions in tree_predictions
        ])
        
        return final_predictions
    
    
    def get_feature_importance(self):
        """
        Returns the aggregated raw Gini importance for all features across the forest.
        """
        # Normalize aggregated importance by the total importance sum
        total_importance = sum(self.feature_importance.values())
        if total_importance == 0:
            return {}
            
        normalized_importance = {
            idx: value / total_importance 
            for idx, value in self.feature_importance.items()
        }
        return normalized_importance