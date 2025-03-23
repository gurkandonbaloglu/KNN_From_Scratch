import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path, columns):
    data_frame = pd.read_csv(file_path, header=None, names=columns)
    X = data_frame.iloc[:, 1:].values
    Y = data_frame.iloc[:, 0].values
    return X, Y

def normalize_data(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized

def data_split(X, Y, split_ratio=0.8, seed=42):
    indices = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    Y_train, Y_test = Y[indices[:split_index]], Y[indices[split_index:]]
    return X_train, X_test, Y_train, Y_test

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)))

def minkowski_distance_p3(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)) ** 3) ** (1/3)

def knn_predict(training_data, training_labels, test_point, k, distance_func):
    distances = []
    for i in range(len(training_data)):
        dist = distance_func(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    
    def get_distance(item):
        return item[0]
    
    distances.sort(key=get_distance)

    k_nearest = distances[:k]
    k_labels = []
    for distance, label in k_nearest:
        k_labels.append(label)

    label_counts = {}
    for label in k_labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    most_coomon_label = max(label_counts, key=label_counts.get)
    return most_coomon_label

def evaluate_accuracy(X_train, Y_train, X_test, Y_test, k, distance_func):
    predictions = []
    for x in X_test:
        pred = knn_predict(X_train, Y_train, x, k, distance_func)
        predictions.append(pred)
    accuracy = np.sum(predictions == Y_test) / len(Y_test)
    return accuracy

def plot_accuracy_vs_k(X_train, Y_train, X_test, Y_test, distance_func, k_values):
    accuracy_matrix = []
    for k_value in k_values:
        acc = evaluate_accuracy(X_train, Y_train, X_test, Y_test, k_value, distance_func)
        accuracy_matrix.append(acc)
    
    distance_name = distance_func.__name__.replace("_", " ").title()
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracy_matrix, marker='o')
    plt.title(f'Accuracy vs K ({distance_name})')
    plt.xlabel('K Values')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

def confusing_matrix_and_Classification_Report(Y_test, predictions, classes, k):
    class_n = sorted(list(set(classes)))
    class_to_index = {label: index for index, label in enumerate(class_n)}

    conf_matrix = np.zeros((len(class_n), len(class_n)), dtype=int)
    for actual, predicted in zip(Y_test, predictions):
        i = class_to_index[actual]
        j = class_to_index[predicted]
        conf_matrix[i][j] += 1

    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_n, yticklabels=class_n)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(f'Confusion Matrix (k = {k})')
    plt.show()

    print("Class\tPrecision\tRecall\t\tF1-Score\tSupport")

    for i, label in enumerate(class_n):
        TP = conf_matrix[i][i]
        FP = sum(conf_matrix[:, i]) - TP
        FN = sum(conf_matrix[i, :]) - TP
        support = sum(conf_matrix[i, :])

        precision_den = TP + FP
        recall_den = TP + FN

        precision = TP / precision_den
        recall = TP / recall_den

        f1_den = precision + recall
        f1 = 2 * precision * recall / f1_den if f1_den != 0 else 0.0

        print(f"{label}\t{precision:.2f}\t\t{recall:.2f}\t\t{f1:.2f}\t\t{support}")

if __name__ == "__main__":

    file_path = 'wine\\wine.data'
    columns = ['Class', 'Alcohol', 'Malic_Asid', 'Ash', 'Alcalinity_of_Ash', 'Magnesium',
               'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins',
               'Color_Intensity', 'Hue', 'OD280/OD315', 'Proline']
    
    X, Y = load_data(file_path, columns)
    X = normalize_data(X)
    X_train, X_test, Y_train, Y_test = data_split(X, Y)

    k = 5
    for dist_func in [euclidean_distance, manhattan_distance, minkowski_distance_p3]:
        accuracy = evaluate_accuracy(X_train, Y_train, X_test, Y_test, k, dist_func)
        print(f"Accuracy with {dist_func.__name__.replace('_', ' ').title()}: {accuracy * 100:.2f}%")

    plot_accuracy_vs_k(X_train, Y_train, X_test, Y_test, euclidean_distance, range(1,17))
    predictions = [knn_predict(X_train, Y_train, x, k, euclidean_distance) for x in X_test]
    confusing_matrix_and_Classification_Report(Y_test, predictions, Y, k)  


    



