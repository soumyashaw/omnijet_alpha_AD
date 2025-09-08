# imports
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

def get_anomaly_scores(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    print("ROC AUC Score:", auc)

    # SIC
    return auc
    