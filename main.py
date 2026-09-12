import math
from collections import Counter

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)

def knn_classify(training_data, new_point, k=3):
    """
    Classifies a new data point using the K-Nearest Neighbors algorithm.
    training_data: List of (features, label) tuples.
    new_point: List of features for the point to classify.
    k: Number of nearest neighbors to consider.
    """
    distances = []
    # Calculate distance from new_point to every point in training_data
    # This step simulates the "learning from data" by comparing new input
    # to all known examples.
    for features, label in training_data:
        dist = euclidean_distance(features, new_point)
        distances.append((dist, label))

    # Sort distances and get the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    # Extract labels of the k nearest neighbors
    neighbor_labels = [label for dist, label in k_nearest_neighbors]

    # Find the most common label among the neighbors
    # This is how the "prediction" is made based on the learned patterns.
    most_common = Counter(neighbor_labels).most_common(1)
    return most_common[0][0]

if __name__ == "__main__":
    # --- Training Data ---
    # This represents the "experience" or "data" the machine learns from.
    # Each entry is (features, label).
    # Features could be anything, e.g., (sweetness, crunchiness) for fruits,
    # or (length, width) for objects. We use simple numerical features.
    training_data = [
        ([7, 3], 'Apple'),
        ([8, 4], 'Apple'),
        ([6, 2], 'Apple'),
        ([2, 8], 'Orange'),
        ([3, 7], 'Orange'),
        ([1, 9], 'Orange'),
        ([7, 2], 'Apple'),
        ([2, 9], 'Orange'),
        ([5, 4], 'Apple'), # More diverse data
        ([4, 6], 'Orange'),
    ]
    print("--- Machine Learning Example: K-Nearest Neighbors ---")
    print("Training Data (Features, Label):")
    for features, label in training_data:
        print(f"  {features} -> {label}")
    print("\n")

    # --- New Data Point to Classify ---
    # This is an "unseen" piece of data that the machine needs to make a prediction for.
    new_fruit = [5, 5]
    print(f"New data point to classify: {new_fruit}")

    # --- Classification ---
    # The machine "learns" by comparing this new point to its training data
    # and identifying patterns (nearest neighbors).
    predicted_label = knn_classify(training_data, new_fruit, k=3)
    print(f"Predicted label for {new_fruit} (k=3): {predicted_label}")

    new_fruit_2 = [1, 1]
    predicted_label_2 = knn_classify(training_data, new_fruit_2, k=3)
    print(f"Predicted label for {new_fruit_2} (k=3): {predicted_label_2}")

    new_fruit_3 = [9, 9]
    predicted_label_3 = knn_classify(training_data, new_fruit_3, k=3)
    print(f"Predicted label for {new_fruit_3} (k=3): {predicted_label_3}")

    print("\nThis simple example demonstrates how a machine can 'learn' from existing data (training_data)")
    print("to classify new, unseen data points without being explicitly programmed with rules like 'if feature1 > X and feature2 < Y then Apple'.")
    print("Instead, it infers the classification based on similarity to known examples.")
