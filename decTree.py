import numpy as np
from collections import Counter
class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature  # the feature index used for the split
        self.threshold = threshold  # the threshold value for splitting
        self.left = left    # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Values if it's a leaf node

    # Checks if a node is a leaf
    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth = None, min_sample_split = 2, min_sample_leaf = 1, min_impurity_decrease = 0):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
    
    # X is a matrix where each row is a samole and each row is a feature
    # y is target labels
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth = 0):
        # .shape returns #of rows and #of columns in a DataFrame (NumPy or pandas)
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # Base case: stop splitting if all labels are the same or max depth is reached!!!
        if len(unique_labels) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            # Counter(y) returns a list of tuples. 
            # Counter(y).most_common(1) returns a list with one tuple
            # Counter(y).most_common(1)[0][0] returns the first element in the first tuple in a list
            return Node(value = Counter(y).most_common(1)[0][0])
        
        # check if there are enough samples to split
        if num_samples < self.min_sample_split:
            return Node(value = Counter(y).most_common(1)[0][0])
        
        # Finds the best feature and threshold to split on
        best_feature, best_threshold = self._best_split(X, y, num_features)

        # If no good split is found, return a leaf node with the majority class. It's a part of Base case!!!
        if best_feature is None:
            return Node(value = Counter(y).most_common(1)[0][0])

        # X[:, best_feature] selects all rows from best_feature column
        # X[:, best_feature] < best_threshold returns a boolean mask [ True True False False ... ]
        left_index = X[:, best_feature] <= best_threshold
        # ~ inverts the boolean mask. (bitwise NOT)
        right_index = ~left_index
        # recursivly calls _grow_tree().
        # Selects rows where left_index is True
        left_child = self._grow_tree(X[left_index], y[left_index], depth + 1)
        # Selects rows where right_index is True
        right_child = self._grow_tree(X[right_index], y[right_index], depth + 1)

        # Check if there is at lelast one sample in a leaf
        if len(y[left_index]) < self.min_sample_leaf or len(y[right_index]) < self.min_sample_leaf:
            return Node(value = Counter(y).most_common(1)[0][0])
        
        # Check if the impurity decrease is sufficient
        if self._information_gain(y, X[:,best_feature], best_threshold) < self.min_impurity_decrease:
            return Node(value = Counter(y).most_common(1)[0][0])

        return Node(feature = best_feature, threshold = best_threshold, left = left_child, right = right_child)
    
    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_feature = feature
        return best_feature, best_threshold
    
    def _information_gain(self, y, X_column, threshold):
        parent_enropy = self._entropy(y)
        left_y, right_y = y[X_column <= threshold], y[X_column > threshold]
        n, n_left, n_right = len(y), len(left_y), len(right_y)
        if n_left == 0 or n_right == 0:
            return 0
        
        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)
        return parent_enropy - child_entropy
        

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / np.sum(counts)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        if x[node.feature] > node.threshold:
            return self._traverse_tree(x, node.right)
        
if __name__=="__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import pandas as pd

    data = load_breast_cancer()

    # _______ This part is unnecessary, and it's present just for learning purposes. The load_breast_cancer is already preprocessed.
    # converts dataset into a DataFrame for easier manipulation
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print(df.head())

    # convert target to DataFrame
    df['target'] = data.target

    # Handling missing values (if any)
    # filling missing values with mean column values
    df.fillna(df.mean(), inplace=True)

    # Converts categorical labels into numerical values
    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["target"])

    # Feature scaling
    scaler = StandardScaler()
    # df.iloc[:,:-1] selects all rows and all columns except the last one (The last one usualy represents labels)
    df.iloc[:,:-1] = scaler.fit_transform(df.iloc[:,:-1])

    print(df.head())

    # ___________________________________________________________
    #X, y = data.data, data.target

    X = df.drop(columns='target', axis =1).to_numpy()
    y = df['target'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #tree = DecisionTree()
    tree = DecisionTree(max_depth=3)
    #tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    prediction = tree.predict(X_test)
    print("Predicted labels:")
    print(prediction)
    print("Actual labels:")
    print(y_test)

    accuracy = np.mean(prediction == y_test)

    print(f"Accuracy score: {accuracy: .2f}")

    precision_test = precision_score(y_test, prediction)
    print(f"Precision score: {precision_test: .2f}")

    recall_test = recall_score(y_test, prediction)
    print(f"Recall score: {precision_test: .2f}")

    f1_score_test = f1_score(y_test, prediction)
    print(f"F1 score: {precision_test: .2f}")
