# project_utils.py

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Function to reshape record to 110 x 110 matrix
def make_pic_mat(pic_record):
    return np.array(pic_record).reshape(110, 110)

# Print a formatted confusion matrix
def print_confumat(confumat):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confumat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confumat.shape[0]):
        for j in range(confumat.shape[1]):
            ax.text(x=j, y=i, s=confumat[i, j], va="center", ha="center")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.show()

# Model running function for ML alogrithms
def run_ml_models(models, X_train, y_train, X_test, y_test, folds, score_metric, random_seed, n_jobs=1):
    # Set up results dictionary
    results = {"model": [], "mean_train_acc": [], "std_train_acc": [], "test_acc": [],
               "precision": [], "recall": [], "f1_score": [], "full_cv_results": [],
               "confusion_matrix": [], "classification_report": []}

    for name, model in models:
        kfold = KFold(n_splits=folds, random_state=random_seed)
        cv_results = cross_val_score(estimator=model, X=X_train, y=y_train,
                                     cv=kfold, scoring=score_metric, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results["model"].append(name)
        results["mean_train_acc"].append(np.mean(cv_results))
        results["std_train_acc"].append(np.std(cv_results))
        results["test_acc"].append(accuracy_score(y_test, preds))
        results["precision"].append(precision_score(y_test, preds))
        results["recall"].append(recall_score(y_test, preds))
        results["f1_score"].append(f1_score(y_test, preds))
        results["full_cv_results"].append(cv_results)
        results["confusion_matrix"].append(confusion_matrix(y_test, preds))
        results["classification_report"].append(classification_report(y_test, preds))
        print("{}\tmean acc: {:.4f} +/- {:.4f}".format(name, np.mean(cv_results), np.std(cv_results)))

    return results

# Model running function for NN alogrithms
def run_nn_model(model_tup, X_train, y_train, X_test, y_test, epochs, batch_size, validation_split):
    # Set up results dictionary
    results = {"model": [], "train_acc": [], "test_acc": [], "precision": [], "recall": [],
               "f1_score": [], "confusion_matrix": [], "classification_report": []}

    name = model_tup[0]
    model = model_tup[1]
    model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=1)

    scores = model.evaluate(X_train, y_train, verbose=0)
    preds = model.predict_classes(X_test)

    results["model"].append(name)
    results["train_acc"].append(scores[1])
    results["test_acc"].append(accuracy_score(y_test, preds))
    results["precision"].append(precision_score(y_test, preds))
    results["recall"].append(recall_score(y_test, preds))
    results["f1_score"].append(f1_score(y_test, preds))
    results["confusion_matrix"].append(confusion_matrix(y_test, preds))
    results["classification_report"].append(classification_report(y_test, preds))
    print("\n{}: train acc: {:.4f}, test acc: {:.4f}".format(name, scores[1], accuracy_score(y_test, preds)))

    return results

# Prints formatted results from run_ml_models
def print_ml_results(result_dic, idx):
    print("###### Results ######")
    print("Model: {}".format(result_dic["model"][idx]))
    print("Mean Training Accuracy: {:.3f}".format(result_dic["mean_train_acc"][idx]))
    print("Std Training Accuracy: {:.3f}".format(result_dic["std_train_acc"][idx]))
    print("Test Accuracy: {:.3f}".format(result_dic["test_acc"][idx]))
    print("Precision: {:.3f}".format(result_dic["precision"][idx]))
    print("Recall: {:.3f}".format(result_dic["recall"][idx]))
    print("F1 Score: {:.3f}".format(result_dic["f1_score"][idx]))
    print_confumat(result_dic["confusion_matrix"][idx])

# Prints formatted results from run_nn_model
def print_nn_results(result_dic, idx):
    print("###### Results ######")
    print("Model: {}".format(result_dic["model"][idx]))
    print("Training Accuracy: {:.3f}".format(result_dic["train_acc"][idx]))
    print("Test Accuracy: {:.3f}".format(result_dic["test_acc"][idx]))
    print("Precision: {:.3f}".format(result_dic["precision"][idx]))
    print("Recall: {:.3f}".format(result_dic["recall"][idx]))
    print("F1 Score: {:.3f}".format(result_dic["f1_score"][idx]))
    print_confumat(result_dic["confusion_matrix"][idx])

