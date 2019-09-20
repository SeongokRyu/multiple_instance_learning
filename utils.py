import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

def get_mnist_data():
    path = "/home/wykgroup/seongok/ml_study/mnist_mlp/mnist/"
    mnist = input_data.read_data_sets(path, one_hot=False)
    train_set = mnist.train
    valid_set = mnist.validation
    test_set = mnist.test
    return train_set, valid_set, test_set

def instances_to_bags(ds, n_inst, target, n_bags=None, p=0.5, seed=123):
    x = np.asarray([img.reshape(28,28) for img in ds.images])
    y = ds.labels

    n_total = x.shape[0]
    if n_bags is None:
        n_bags = n_total//n_inst

    rand_state = np.random.RandomState(seed)
    indices = rand_state.randint(0, n_total, n_total)
    x = x[indices]
    y = y[indices]

    target_idx = []
    off_target_idx = []
    for i in indices:        
        if y[i] == target:
            target_idx.append(i)
        else:     
            off_target_idx.append(i)

    x_on = x[target_idx]
    x_off = x[off_target_idx]

    x_bag = []
    y_bag = []
    for i in range(n_bags):
        xi = x_off[i*(n_inst-1):(i+1)*(n_inst-1)]
        pi = rand_state.uniform(0.0,1.0)

        label=0.0
        if pi > (1.0-p):
           label=1.0
           xi = np.concatenate([xi, np.asarray([x_on[i]])], axis=0)
        else:   
           xi = np.concatenate([xi, np.asarray([x_off[-i]])], axis=0)
           

        x_bag.append(xi)
        y_bag.append(label)
    return np.asarray(x_bag), np.asarray(y_bag)
