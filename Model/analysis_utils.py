import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    cohen_kappa_score, matthews_corrcoef, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

def get_final_result(threshold, y_test, preds):
    y_preds = (preds > threshold).astype(int)
    auc = roc_auc_score(y_true=y_test, y_score=preds, )
    # y_preds = preds.argmax(1) #argmax取出preds元素最大值所对应的索引,1代表维度，是指在第二维里取最大值的索引
    acc = accuracy_score(y_true=y_test, y_pred=y_preds)
    recall = recall_score(y_true=y_test, y_pred=y_preds, )
    precision = precision_score(y_true=y_test, y_pred=y_preds,  labels=[0])
    f1 = f1_score(y_true=y_test, y_pred=y_preds, )
    kappa = cohen_kappa_score(y1=y_test, y2=y_preds, )
    mcc = matthews_corrcoef(y_true=y_test, y_pred=y_preds)
    con = confusion_matrix(y_test, y_preds)
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
    print(con)
    print('acc:', '{:.3f}'.format(acc), 'f1:', '{:.3f}'.format(f1), 'auc:', '{:.3f}'.format(auc),
          'kappa:', '{:.3f}'.format(kappa), 'MCC:', '{:.3f}'.format(mcc), 'recall:', '{:.3f}'.format(recall),
          'specificity:', '{:.3f}'.format(specificity), 'PPV:', '{:.3f}'.format(PPV), 'NPV:', '{:.3f}'.format(NPV))
    print('{:.3f}'.format(auc), '{:.3f}'.format(acc), '{:.3f}'.format(f1), '{:.3f}'.format(recall),
          '{:.3f}'.format(specificity), '{:.3f}'.format(PPV), '{:.3f}'.format(NPV), '\n')
    return y_preds

def X_y_split(df, lal):
    X = df.drop([lal], axis=1)
    y = df[lal]
    return X, y

def confusion_matrix_plots(label, y_preds, path):
    con = confusion_matrix(label, y_preds)
    labels = ['non-PHLF', 'PHLF']
    plt.figure(figsize=(13, 10))
    plt.rcParams['font.size'] = 32
    sns.heatmap(con, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    # plt.xlabel('Predicted', fontsize=22)
    # plt.ylabel('Truth', fontsize=22)
    plt.savefig(path)
    # plt.show()

# def confusion_matrix_plots_User_Exm(label, y_preds, path):
#     con = confusion_matrix(label, y_preds)
#     labels = ['non-PHLF', 'PHLF']
#     plt.figure(figsize=(13, 10))
#     plt.rcParams['font.size'] = 32
#     sns.heatmap(con, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
#     # plt.xlabel('Predicted', fontsize=22)
#     # plt.ylabel('Truth', fontsize=22)
#     plt.savefig(path)


def dict_result_obtain(threshold, pred, label, path):
    result = get_final_result(threshold, y_test=label, preds=pred)
    label_list = [float(x) for x in list(label)]
    y_pred_list = [float(x) for x in list(pred)]
    result_list = [float(x) for x in list(result)]
    dict_result = {
        'real': label_list,
        'pred_pro': result_list,
        'pred': y_pred_list,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dict_result, f)

def error_samples (data, ori_data,y_pres, y_data):
    error_data = data[y_pres != y_data]
    ori_erro_data = ori_data.iloc[error_data.index, :]
    value_counts = ori_erro_data['PHLFG'].value_counts()
    result = [(str(index), str(value)) for index, value in value_counts.items()]
    # print('error','\n',ori_erro_data['PHLFG'].value_counts())
    for index, value in value_counts.items():
        print(f'{index}: {value}')
    # print(ori_erro_data['PHLFG'].value_counts())
    # type_list = list(ori_erro_data['PHLFG'].value_counts())
    return error_data, ori_erro_data, result

def error_samples_name (data, ori_data,y_pres, y_data, name):
    error_data = data[y_pres != y_data]
    ori_erro_data = ori_data.iloc[error_data.index, :]
    value_counts = ori_erro_data['PHLFG'].value_counts()
    result = [(str(index), str(value)) for index, value in value_counts.items()]
    print(name)
    print('error','\n',ori_erro_data['PHLFG'].value_counts())
    # type_list = list(ori_erro_data['PHLFG'].value_counts())
    return error_data, ori_erro_data, result

def correct_samples (data, ori_data,y_pres, y_data):
    error_data = data[y_pres == y_data]
    ori_erro_data = ori_data.iloc[error_data.index, :]
    value_counts = ori_erro_data['PHLFG'].value_counts()
    result = [(str(index), str(value)) for index, value in value_counts.items()]
    print('correct','\n',ori_erro_data['PHLFG'].value_counts())
    # type_list = list(ori_erro_data['PHLFG'].value_counts())
    return error_data, ori_erro_data, result

def test_2_oridata (data, ori_data):
    '''
    根据test data 的索引得到对应的original data
    '''
    ori_data_test = ori_data.loc[data.iloc[:, 0], :]

    return ori_data_test