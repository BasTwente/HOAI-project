# This script is used to train the SVC to classify hands and test it.

import pandas as pd
import numpy as np
import ast

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, precision_score
import matplotlib.pyplot as plt
from joblib import dump, load



classification_model_save_path = "models/saved_model.joblib"

# Whether to retrain the model or not
train = True

def retrain_model():
    # Read data
    df = pd.read_csv("data/combined_data.csv")

    # comment this part if we want to use "Nothing" gesture
    for x in df.index:
        if df.loc[x, "Gesture"] == "Nothing":
            df.drop(x, inplace=True)

    # The landmarks need to be flattened into single arrays first
    landmarks = df["Landmarks"].apply(ast.literal_eval).to_list()
    flattened_data = [np.array(entry).flatten() for entry in landmarks]

    # Y contains the gesture label
    y = df["Gesture"]

    # Split the data
    train_X, test_X, train_y, test_y = train_test_split(flattened_data, y, test_size=0.2)


    print("Data has been loaded and split.")
    #
    # SVC
    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.1, 10], 'gamma': [0.1, 1, 10]}
    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, parameters)

    # Fit model
    clf.fit(train_X, train_y)
    dump(clf, classification_model_save_path)
    clf = load("models/saved_model.joblib")
    print("Model training has been completed. Saved as" + classification_model_save_path)
    print("Metrics:")
    test_predictions = clf.predict(test_X)
    cm = confusion_matrix(test_y, test_predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
    disp.plot()
    print("accuracy", clf.score(test_X, test_y))
    test_probac = clf.predict_proba(test_X)

    fpr, tpr, thresholds = roc_curve(test_y, test_probac[:, clf.classes_ == "Open"], pos_label="Open")
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()
    print(clf.best_params_)
    print("precision:", precision_score(test_y, test_predictions,average="micro"))



if train:
    retrain_model()

