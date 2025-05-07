import Model
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from Model.evaluator import predict_multi_task
from Result_process.metrics_process import metrics_with_youden, metrics_multiclass

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
    'batch_size':8,
    'lr':0.0005,
    'weight_decay':1e-05,
    'patience' :5,
    'eval_batch_size':128,
    'num_epoch' :200,
    'imb_weight': 2
}

PATH = ''
def evaluate_params(params):

    model = Model.build_classifier_multi_task(cate_cols, num_cols, bin_cols, imb_weight=params['imb_weight'])
    model = model.to('cuda')
    Model.train_multi_task(model, train_set, valid_set, **params)
    # print(params)

    y_test = pd.concat([df_valid.iloc[:, 1], df_valid.iloc[:, 0]], axis=1)
    prob_int_parent, prob_int_child = predict_multi_task(model, df_valid.iloc[:, 2:], y_test)

    result_int_parent, parent_con = metrics_with_youden(df_valid.iloc[:, 1], prob_int_parent[:, 1])
    print('\n')
    result_int_child, child_con = metrics_multiclass(df_valid.iloc[:, 0], prob_int_child)

    return result_int_parent, result_int_child,parent_con, child_con, model


PATH = ''

result_int_parent, result_int_child, parent_con, child_con, model = evaluate_params(training_arguments)
model.save('./checkpoint/' + PATH )


