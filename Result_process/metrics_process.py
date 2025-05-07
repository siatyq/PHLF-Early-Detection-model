from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score,roc_curve, auc
import numpy as np

def metrics_with_youden(y_true, pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, pred_prob)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    y_preds = (np.array(pred_prob) > optimal_threshold).astype(int)
    acc = accuracy_score(y_true=y_true, y_pred=y_preds)
    recall = recall_score(y_true=y_true, y_pred=y_preds)
    precision = precision_score(y_true=y_true, y_pred=y_preds, labels=[0])
    f1 = f1_score(y_true=y_true, y_pred=y_preds)
    kappa = cohen_kappa_score(y1=y_true, y2=y_preds)
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_preds)
    con = confusion_matrix(y_true, y_preds)

    TN = con[0, 0]
    FP = con[0, 1]
    FN = con[1, 0]
    TP = con[1, 1]

    if (TP + FP) > 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 0  # 或者可以设置为 NaN，取决于你的需求

    if (TN + FN) > 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 0
    specificity = TN / (TN + FP)

    # Compute AUC and DeLong test p-value
    auc = roc_auc_score(y_true=y_true, y_score=pred_prob)
    print(con)
    print('Optimal Threshold:', '{:.3f}'.format(optimal_threshold))

    print('AUC','{:.3f}'.format(auc), 'acc:', '{:.3f}'.format(acc), 'f1:', '{:.3f}'.format(f1),'sensitivity', '{:.3f}'.format(recall),
          'specificity:', '{:.3f}'.format(specificity),  'PPV:', '{:.3f}'.format(PPV), 'NPV:', '{:.3f}'.format(NPV))

    print('{:.3f}'.format(auc), '{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall), \
          '{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV), '{:.3f}'.format(optimal_threshold))

    return ['{:.3f}'.format(auc), '{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall), \
          '{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV), '{:.3f}'.format(optimal_threshold)], con

def compute_metrics_per_class(y_true, y_pred, n_classes):
    """
    Compute specificity, PPV, and NPV for each class in a multi-class setting.

    Parameters:
        y_true (array-like): True labels (shape: [n_samples,]).
        y_pred (array-like): Predicted labels (shape: [n_samples,]).
        n_classes (int): Number of classes.

    Returns:
        specificity (list): Specificity for each class.
        ppv (list): Positive Predictive Value (PPV) for each class.
        npv (list): Negative Predictive Value (NPV) for each class.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    specificity = []
    ppv = []
    npv = []

    for class_idx in range(n_classes):
        # Extract true positives, false positives, true negatives, false negatives
        TP = cm[class_idx, class_idx]
        FP = cm[:, class_idx].sum() - TP
        FN = cm[class_idx, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        # Calculate metrics for the current class
        if (TN + FP) > 0:
            specificity.append(TN / (TN + FP))
        else:
            specificity.append(0)  # Avoid division by zero

        if (TP + FP) > 0:
            ppv.append(TP / (TP + FP))
        else:
            ppv.append(0)  # Avoid division by zero

        if (TN + FN) > 0:
            npv.append(TN / (TN + FN))
        else:
            npv.append(0)  # Avoid division by zero

    return specificity, ppv, npv

def metrics_multiclass(y_true, pred_prob, average='macro'):
    """
    Compute metrics for multi-class classification without using Youden's index.

    Parameters:
        y_true (array-like): True labels (shape: [n_samples,]).
        pred_prob (array-like): Predicted probabilities (shape: [n_samples, n_classes]).
        average (str): Type of averaging for multi-class metrics ('macro', 'micro', 'weighted').

    Returns:
        Prints metrics and confusion matrix for multi-class classification.
    """
    n_classes = pred_prob.shape[1]

    # Assign predicted class based on the highest predicted probability
    y_pred_classes = np.argmax(pred_prob, axis=1)

    # Compute overall metrics
    acc = accuracy_score(y_true=y_true, y_pred=y_pred_classes)
    recall = recall_score(y_true=y_true, y_pred=y_pred_classes, average=average, zero_division=0)
    # precision = precision_score(y_true=y_true, y_pred=y_pred_classes, average=average)
    f1 = f1_score(y_true=y_true, y_pred=y_pred_classes, average=average, zero_division=0)
    # kappa = cohen_kappa_score(y1=y_true, y2=y_pred_classes)
    # mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred_classes)
    con = confusion_matrix(y_true, y_pred_classes)

    # Compute AUC for each class using One-vs-Rest
    y_true_onehot = np.eye(n_classes)[y_true]  # Convert y_true to one-hot encoding
    auc_scores = []
    for class_idx in range(n_classes):
        auc = roc_auc_score(y_true_onehot[:, class_idx], pred_prob[:, class_idx])
        auc_scores.append(auc)
    auc_avg = np.mean(auc_scores)

    # Compute specificity, PPV, and NPV per class
    specificity, ppv, npv = compute_metrics_per_class(y_true, y_pred_classes, n_classes)
    specificity_avg = np.mean(specificity)
    ppv_avg = np.mean(ppv)
    npv_avg = np.mean(npv)

    # print(con)
    #
    # print('AUC','{:.3f}'.format(auc), 'acc:', '{:.3f}'.format(acc), 'f1:', '{:.3f}'.format(f1),'sensitivity', '{:.3f}'.format(recall),
    #       'specificity:', '{:.3f}'.format(specificity_avg),  'PPV:', '{:.3f}'.format(ppv_avg), 'NPV:', '{:.3f}'.format(npv_avg))
    #
    # print('{:.3f}'.format(auc), '{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall), \
    #       '{:.3f}'.format(specificity_avg), '{:.3f}'.format(ppv_avg), '{:.3f}'.format(npv_avg))
    return ['{:.3f}'.format(auc_avg), '{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall), \
          '{:.3f}'.format(specificity_avg), '{:.3f}'.format(ppv_avg), '{:.3f}'.format(npv_avg)], con

def AUC_resutl_with_CI (y_true, y_pred):
    dic = {}
    dic['fpr'], dic['tpr'], _ = roc_curve(y_true, y_pred)
    dic['roc_auc'] = auc(dic['fpr'], dic['tpr'])

    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_score = []
    bootstrapped_tprs = []

    mean_fpr = np.linspace(0, 1, 100)  # Fixed grid for interpolation

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        fpr_bs, tpr_bs, _ = roc_curve(y_true[indices], y_pred[indices])
        bootstrapped_score.append(score)
        bootstrapped_tprs.append(np.interp(mean_fpr, fpr_bs, tpr_bs))  # Interpolate TPR on fixed grid

    sorted_scores = np.array(bootstrapped_score)
    bootstrapped_tprs = np.array(bootstrapped_tprs)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(len(sorted_scores) * 0.025)]
    confidence_upper = sorted_scores[int(len(sorted_scores) * 0.975)]
    tprs_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
    tprs_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)

    dic['95% CI'] = ('( {}, {})'.format(round(confidence_lower, 3), round(confidence_upper, 3)))
    dic['tprs_lower'] = tprs_lower
    dic['tprs_upper'] = tprs_upper
    dic['mean_fpr'] = mean_fpr
    return dic

