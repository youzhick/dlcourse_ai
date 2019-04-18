def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''

    # TODO: implement metrics!
    TP = prediction[(prediction == True)  & (ground_truth == True) ].shape[0]
    FP = prediction[(prediction == True)  & (ground_truth == False)].shape[0]
    TN = prediction[(prediction == False) & (ground_truth == False)].shape[0]
    FN = prediction[(prediction == False) & (ground_truth == True) ].shape[0]
    
    precision = TP/(TP + FP) if TP + FP > 0 else 0
    recall = TP/(TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN)/ground_truth.shape[0]
    f1 = 2*precision*recall/(precision + recall) if precision + recall > 0 else 0

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    
    total = prediction.shape[0]
    correct = prediction[prediction == ground_truth].shape[0]
    
    return correct/total
