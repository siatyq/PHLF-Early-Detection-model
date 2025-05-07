import Model
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score, confusion_matrix,\
    cohen_kappa_score, matthews_corrcoef, precision_score
import numpy as np

def get_final_result(preds,label,):
    y_preds = np.zeros_like(preds)
    for idx, val in enumerate(preds):
        if val > 0.5:
            y_preds[idx] = 1
    auc = roc_auc_score(y_true=label, y_score=preds)
    # y_preds = preds.argmax(1) #argmax取出preds元素最大值所对应的索引,1代表维度，是指在第二维里取最大值的索引
    acc = accuracy_score(y_true=label, y_pred=y_preds)
    recall = recall_score(y_true=label, y_pred=y_preds)
    precision = precision_score(y_true=label, y_pred=y_preds, labels=[0])
    f1 = f1_score(y_true=label, y_pred=y_preds, )
    kappa = cohen_kappa_score(y1=label, y2=y_preds, )
    mcc = matthews_corrcoef(y_true=label, y_pred= y_preds)
    con = confusion_matrix(label, y_preds)
    print(con)
    print('acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc), 'recall:','{:.3f}'.format(recall))
    return ['acc:','{:.3f}'.format(acc),'f1:','{:.3f}'.format(f1),'auc:','{:.3f}'.format(auc),
            'kappa:','{:.3f}'.format(kappa),'MCC:','{:.3f}'.format(mcc), 'recall:','{:.3f}'.format(recall),], con

cate_cols = []
num_cols = []
bin_cols = []

train_dataset_path = ''
test_dataset_path = ''

df_alin_train = pd.read_csv(train_dataset_path)
df_alin_test = pd.read_csv(test_dataset_path)

df_alin_train = df_alin_train.iloc[:,1:]
df_alin_test = df_alin_test.iloc[:,1:]

df_train, df_valid = train_test_split(df_alin_train, test_size=0.2,random_state=42)
train_set = [df_train.iloc[:,1:], df_train.iloc[:,0]]
valid_set = [df_valid.iloc[:,1:], df_valid.iloc[:,0]]

training_arguments = {
    'batch_size':16,
    'lr':1e-4,
    'weight_decay':1e-5,
    'patience' :5,
    'eval_batch_size':256,
    'num_epoch' :200,
    'imb_weight': 2.5
}

PATH = ''
# best_params = None
best_score = -float('inf')  # 我们是在最大化某个指标
count = 0


model = Model.build_classifier(cate_cols, num_cols, bin_cols, imb_weight=training_arguments['imb_weight'])
model = model.to('cuda')
Model.train(model, train_set, valid_set, **training_arguments)

prob_int = Model.predict(model, df_alin_test.iloc[:, 1:], df_alin_test.iloc[:, 0])
result_int, cm_int = get_final_result(prob_int, df_alin_test.iloc[:, 0])

model.save('./checkpoint/'+PATH + str(count))




