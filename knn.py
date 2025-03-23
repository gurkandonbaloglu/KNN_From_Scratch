
import numpy as np

# Define different distance metrics
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)))

def minkowski_distance_p3(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)) ** 3) ** (1/3)    

# Core KNN prediction logic
def knn_predict(training_data, training_labels, test_point, k, distance_func):
    # Calculate distances from test point to all training points
    distances = []
    for i in range(len(training_data)):
        dist = distance_func(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    
    def get_distance(item):
        return item[0]
    # Sort by distance
    distances.sort(key=get_distance)

    # Get labels of k nearest neighbors
    k_nearest = distances[:k]
    k_labels = []
    for distance, label in k_nearest:
        k_labels.append(label)

    # Count frequency of each label among neighbors
    label_counts = {}
    for label in k_labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # Return the most frequent label
    most_coomon_label = max(label_counts, key=label_counts.get)
    return most_coomon_label

# Evaluate prediction accuracy
def evaluate_accuracy(X_train, Y_train, X_test, Y_test, k, distance_func):
    predictions = []
    for x in X_test:
        pred = knn_predict(X_train, Y_train, x, k, distance_func)
        predictions.append(pred)
    accuracy = np.sum(predictions == Y_test) / len(Y_test)
    return accuracy





    



