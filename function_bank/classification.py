from distance_calculation import *

def k_nearest_neighbor_classification(
    train_points: list,
    train_labels: list,
    query_point: tuple,
    k: int,
    distance_function=euclidean
) -> any:
    """
    Classify a point using the K-Nearest Neighbors algorithm.

    Parameters:
        train_points (list of tuples): Training data points.
        train_labels (list): Class labels corresponding to each training point.
        query_point (tuple): The point to classify.
        k (int): Number of neighbors to consider.
        distance_function (function): Distance function to use.

    Returns:
        The predicted class label.
    """

    if len(train_points) != len(train_labels):
        raise ValueError("train_points and train_labels must have same length")

    if k <= 0:
        raise ValueError("k must be positive")

    # Calculate distances to all training points
    distances = []
    for point, label in zip(train_points, train_labels):
        distance = distance_function(query_point, point)
        distances.append((distance, label))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get k nearest neighbors
    nearest_neighbors = distances[:k]

    # Count votes
    votes = {}
    for _, label in nearest_neighbors:
        votes[label] = votes.get(label, 0) + 1

    # Return label with most votes
    return max(votes, key=votes.get)


def classification_metrics(true_labels: list, predicted_labels: list):
    """
    Compute precision, recall, and F1 for each class,
    plus micro and macro averaged F1.
    
    Parameters:
        true_labels (list): Ground truth class labels.
        predicted_labels (list): Predicted class labels.
    
    Returns:
        tuple: (per_class_results, micro_f1, macro_f1)
            - per_class_results (dict): Metrics for each class
            - micro_f1 (float): Micro-averaged F1 score  
            - macro_f1 (float): Macro-averaged F1 score
    """
    
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have same length")

    classes = set(true_labels)
    results = {}

    total_TP = 0
    total_FP = 0
    total_FN = 0

    for c in classes:

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for t, p in zip(true_labels, predicted_labels):

            if t == c and p == c:
                TP += 1
            elif t != c and p == c:
                FP += 1
            elif t == c and p != c:
                FN += 1
            else:
                TN += 1

        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0

        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        results[c] = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Micro average - aggregate then compute
    if total_TP + total_FP > 0:
        micro_precision = total_TP / (total_TP + total_FP)
    else:
        micro_precision = 0
        
    if total_TP + total_FN > 0:
        micro_recall = total_TP / (total_TP + total_FN)
    else:
        micro_recall = 0
        
    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = 0

    # Macro average - average of individual F1 scores
    macro_f1 = sum(results[c]["f1"] for c in classes) / len(classes)

    return results, micro_f1, macro_f1