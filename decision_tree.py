import numpy as np
from collections import Counter

# --- Node Class ---

class Node:
    """
    Represents a single node in the Decision Tree.

    Attributes:
        feature_index (int or None): Index of the feature used for splitting at this node.
        threshold (float or None): Value of the feature used for splitting at this node.
        left (Node or None): Left child node (samples <= threshold).
        right (Node or None): Right child node (samples > threshold).
        value (int or None): The prediction class if this is a leaf node (mode of labels).
        samples (int): Number of samples that reached this node.
        gini_impurity (float): Gini impurity of the data at this node.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, samples=0, gini_impurity=0.0):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        # Leaf node prediction
        self.value = value
        # Metadata for analysis (e.g., node importance calculation)
        self.samples = samples
        self.gini_impurity = gini_impurity

# --- Decision Tree Classifier ---

class DecisionTreeClassifier:
    """
    Custom implementation of a Decision Tree Classifier using Gini impurity.

    Parameters:
        max_depth (int): The maximum depth the tree can grow to.
        min_node (int): The minimum number of samples required to split an internal node.
        feature_subcount (int or None): Number of features to randomly sample at each split
                                        (used by Random Forest, None for standard DT).
    """
    def __init__(self, max_depth=10, min_node=5, feature_subcount=None):
        self.max_depth = max_depth
        self.min_node = min_node
        self.feature_subcount = feature_subcount
        self.root = None
        # Stores the total Gini decrease (importance) for each feature index
        self.feature_importance = {}


    def _calculate_gini(self, y):
        """
        Computes the Gini impurity for a set of labels (y).
        Gini = 1 - sum(p_k^2) for k=1 to C classes.
        
        Args:
            y (np.array): Array of class labels.
            
        Returns:
            float: Gini impurity score.
        """
        n_samples = len(y)
        if n_samples == 0:
            return 0.0
            
        # Count occurrences of each class
        label_counts = Counter(y)
        gini = 1.0
        
        # Calculate sum of squared probabilities
        for label in label_counts:
            p_k = label_counts[label] / n_samples
            gini -= p_k**2
            
        return gini


    def _split_data(self, X, y, feature_index, threshold):
        """
        Splits the dataset (X, y) into two groups based on a feature and a threshold.
        
        Args:
            X (np.array): Feature matrix.
            y (np.array): Label array.
            feature_index (int): Index of the feature to split on.
            threshold (float): Value to split the feature on.
            
        Returns:
            tuple: (X_left, y_left, X_right, y_right)
        """
        # Boolean masks for left (<= threshold) and right (> threshold) groups
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        return X_left, y_left, X_right, y_right


    def _find_best_split(self, X, y, feature_indices):
        """
        Finds the best feature and threshold to split the data (X, y) based on Gini impurity.
        
        Args:
            X (np.array): Feature matrix.
            y (np.array): Label array.
            feature_indices (np.array): Subset of feature indices to check (for RF).
            
        Returns:
            tuple: (best_gini, best_feature_index, best_threshold)
        """
        n_samples, n_features = X.shape
        if n_samples < 2:
            return float('inf'), None, None
            
        # Gini of the current parent node
        parent_gini = self._calculate_gini(y)
        best_gini = parent_gini
        best_feature_index = None
        best_threshold = None
        
        # Iterate over the subset of features (or all features for standard DT)
        for feature_index in feature_indices:
            unique_values = np.unique(X[:, feature_index])
            
            # Use midpoints between unique values as potential thresholds
            # This is more efficient than checking every single data point
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i+1]) / 2
                
                # Split the data
                _, y_left, _, y_right = self._split_data(X, y, feature_index, threshold)
                
                n_left, n_right = len(y_left), len(y_right)
                n_total = n_left + n_right
                
                # Stopping Condition: Optimal split results in a group with no samples
                if n_left == 0 or n_right == 0:
                    continue
                    
                # Calculate weighted Gini impurity for the split
                gini_left = self._calculate_gini(y_left)
                gini_right = self._calculate_gini(y_right)
                
                weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
                
                # Check for the best split (lowest weighted Gini)
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        # Stopping Condition: The optimal split has a worse Gini impurity than the parent node,
        # in which case use the parent node (handled by checking if best_gini is still parent_gini)
        # Note: best_gini will be less than or equal to parent_gini. If it is equal, we return None
        # for split parameters, which the _build_tree function interprets as a leaf node.
        
        return best_gini, best_feature_index, best_threshold, parent_gini


    def _create_leaf_value(self, y):
        """
        Determines the predicted class for a leaf node (the mode of the sample labels).
        
        Args:
            y (np.array): Array of class labels.
            
        Returns:
            int: The most common class label.
        """
        # Count occurrences of each class
        most_common = Counter(y).most_common(1)
        # Return the label of the most common class
        return most_common[0][0]


    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.
        
        Args:
            X (np.array): Feature matrix for the current node.
            y (np.array): Label array for the current node.
            depth (int): Current depth of the node.
            
        Returns:
            Node: The root node of the constructed (sub)tree.
        """
        n_samples, n_features = X.shape
        current_gini = self._calculate_gini(y)
        
        # --- Leaf Node Conditions ---
        
        # 1. Stopping Condition: Homogeneous node (Gini is 0)
        # 2. Stopping Condition: The branch has reached the max_depth
        # 3. Stopping Condition: The number of samples in the group to split is less than the min_node
        if current_gini == 0.0 or depth >= self.max_depth or n_samples < self.min_node:
            leaf_value = self._create_leaf_value(y)
            return Node(value=leaf_value, samples=n_samples, gini_impurity=current_gini)
            
        # --- Splitting Logic ---
        
        # 1. Feature Subsetting for Random Forest
        if self.feature_subcount is not None and self.feature_subcount <= n_features:
            # Randomly sample 'feature_subcount' features with replacement
            feature_indices = np.random.choice(n_features, self.feature_subcount, replace=False)
        else:
            # Use all features for standard DT or if feature_subcount is invalid
            feature_indices = np.arange(n_features)

        # 2. Find the best split
        best_gini, best_feature_index, best_threshold, parent_gini = self._find_best_split(X, y, feature_indices)

        # 3. Stopping Condition: The optimal split has a worse Gini impurity than the parent node
        # (This is implicitly handled if best_feature_index is None, meaning no split was found
        # that decreased the Gini, or best_gini == parent_gini)
        if best_feature_index is None or best_gini >= parent_gini:
            leaf_value = self._create_leaf_value(y)
            return Node(value=leaf_value, samples=n_samples, gini_impurity=current_gini)
            
        # --- Create Internal Node and Recurse ---
        
        # Record the Gini decrease for feature importance calculation
        # Node Importance = N_parent * (Gini_parent - N_left/N_parent * Gini_left - N_right/N_parent * Gini_right)
        # Since we only stored best_gini (the weighted child Gini), we use:
        gini_decrease = parent_gini - best_gini
        self.feature_importance[best_feature_index] = self.feature_importance.get(best_feature_index, 0) + n_samples * gini_decrease
        
        # Split the data based on the best split found
        X_left, y_left, X_right, y_right = self._split_data(X, y, best_feature_index, best_threshold)
        
        # Create the node and recursively build children
        node = Node(
            feature_index=best_feature_index,
            threshold=best_threshold,
            left=self._build_tree(X_left, y_left, depth + 1),
            right=self._build_tree(X_right, y_right, depth + 1),
            samples=n_samples,
            gini_impurity=current_gini
        )
        return node


    def fit(self, X, y):
        """
        Trains the Decision Tree on the provided data.
        
        Args:
            X (np.array): Training feature matrix.
            y (np.array): Training labels.
        """
        # Clear previous feature importance map
        self.feature_importance = {i: 0.0 for i in range(X.shape[1])}
        self.root = self._build_tree(X, y)


    def _traverse_tree(self, x, node):
        """
        Traverses the tree to find a leaf node and get the prediction.
        
        Args:
            x (np.array): Single test sample (feature vector).
            node (Node): Current node in the traversal.
            
        Returns:
            int: Predicted class label.
        """
        # Base case: If it's a leaf node, return its value
        if node.value is not None:
            return node.value
            
        # Decide which path to take
        feature_value = x[node.feature_index]
        
        if feature_value <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


    def predict(self, X):
        """
        Predicts the class labels for a dataset X.
        
        Args:
            X (np.array): Feature matrix to predict on.
            
        Returns:
            np.array: Array of predicted class labels.
        """
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
        
    
    def get_feature_importance(self):
        """
        Returns the raw Gini importance (total Gini decrease) for each feature.
        """
        return self.feature_importance

    # Helper method for the report requirement (Section 4)
    def print_node_info(self, node=None, depth=0):
        """
        Finds and prints information for a non-trivial (non-leaf) node for analysis.
        This is a depth-first search for the first non-leaf node after the root.
        """
        if node is None:
            node = self.root

        if node.value is None and depth > 0: # Found a non-trivial split node
            print("\n--- Non-Trivial Node Information (for Report Q4) ---")
            print(f"Node Depth: {depth}")
            print(f"Total Samples at Node: {node.samples}")
            print(f"Parent Gini Impurity: {node.gini_impurity:.4f}")
            print(f"Splitting Feature Index: {node.feature_index}")
            print(f"Splitting Threshold: {node.threshold:.4f}")
            
            # Since we don't have the original data at this node, we calculate the post-split Gini
            # of the children using the saved node structure.
            n_left = node.left.samples
            gini_left = node.left.gini_impurity
            
            n_right = node.right.samples
            gini_right = node.right.gini_impurity
            
            n_total = node.samples
            weighted_child_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
            
            # Calculate Node Importance (Gini Decrease)
            # Node Importance = Samples_at_Node * (Gini_Parent - Weighted_Child_Gini)
            node_importance = n_total * (node.gini_impurity - weighted_child_gini)
            
            print(f"Left Child Samples: {n_left} (Gini: {gini_left:.4f})")
            print(f"Right Child Samples: {n_right} (Gini: {gini_right:.4f})")
            print(f"Weighted Child Gini (Best Split Gini): {weighted_child_gini:.4f}")
            print(f"Calculated Node Importance: {node_importance:.6f}")
            print("--------------------------------------------------")
            return True
            
        if node.left and self.print_node_info(node.left, depth + 1):
            return True
        if node.right and self.print_node_info(node.right, depth + 1):
            return True
            
        return False