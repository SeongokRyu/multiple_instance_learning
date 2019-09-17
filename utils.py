import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

def np_sigmoid(x):
    return 1./(1.+np.exp(-x))

def calculate_accuracy_auroc(y_truth, y_pred):    
    auroc = 0.0
    try:
        auroc = roc_auc_score(y_truth, y_pred)
    except:
        auroc = 0.0    

    y_truth = np.around(y_truth)
    y_pred = np.around(y_pred).astype(int)

    accuracy = accuracy_score(y_truth, y_pred)
    precision = precision_score(y_truth, y_pred)
    recall = recall_score(y_truth, y_pred)
    f1_score = 2*(precision*recall)/(precision+recall)
    return accuracy, auroc, precision, recall, f1_score

def print_metrics(y_truth, y_pred):
    accuracy, auroc, precision, recall, f1_score = \
        calculate_accuracy_auroc(y_truth, y_pred)
    print ("Accuracy:", round(accuracy, 5), 
           "AUROC:", round(auroc, 5),
           "Precision:", round(precision, 5),
           "Recall:", round(recall, 5),
           "F1-score:", round(f1_score, 5))
    return        
