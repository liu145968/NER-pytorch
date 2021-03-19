import numpy as np
import pandas as pd
from tqdm import tqdm
import torch,pickle,time
from Config import X_test_PATH,y_test_PATH,P_test_PATH,USE_POS,word2id_PATH,tag2id_PATH,model_storage_PATH,\
    BATCH_SIZE,EPOCH,LR,EMBED_SIZE,HIDDEN_SIZE,DROPOUT_RATE,n_transformer_layers,n_heads,DEVICE,MODEL_type,USE_pretrained_vector,\
    train_data_PATH,TAG_O_CHAR

def evaluate(tag_seq,pred_seq,tag2id,print_detailed=False):
    print("Compute Confusion Matrix...")
    confusion_matrix = np.zeros((len(tag2id), len(tag2id)), dtype=float)
    for yt, yp in zip(tag_seq, pred_seq):
        confusion_matrix[yt][yp] += 1
    score_dict = {}
    EPS = 1e-8
    tags_sum = 0
    print("evaluate...")
    for i in tqdm(range(len(confusion_matrix))):
        amount_true = np.sum(confusion_matrix[i, :])
        amount_pred = np.sum(confusion_matrix[:, i])
        precision = confusion_matrix[i][i] / (amount_pred + EPS)
        recall = confusion_matrix[i][i] / (amount_true + EPS)
        F1 = (2 * precision * recall) / (precision + recall + EPS)
        score_dict[list(tag2id.keys())[i]] = [precision, recall, F1, amount_true]
        tags_sum += amount_true

    tempP, tempR, tempF1 = 0, 0, 0
    for k,v in score_dict.items():
        tempP += v[0] * v[3]
        tempR += v[1] * v[3]
        tempF1 += v[2] * v[3]
    wei_averageP = tempP / tags_sum
    wei_averageR = tempR / tags_sum
    wei_averageF1 = tempF1 / tags_sum
    if TAG_O_CHAR not in score_dict:
        average_dict = {"weighted avg": [wei_averageP, wei_averageR, wei_averageF1, tags_sum]}
    else:
        wei_averageeOP = (tempP - (score_dict[TAG_O_CHAR][0] * score_dict[TAG_O_CHAR][3])) / (tags_sum - score_dict[TAG_O_CHAR][3])
        wei_averageeOR = (tempR - (score_dict[TAG_O_CHAR][1] * score_dict[TAG_O_CHAR][3])) / (tags_sum - score_dict[TAG_O_CHAR][3])
        wei_averageeOF1 = (tempF1 - (score_dict[TAG_O_CHAR][2] * score_dict[TAG_O_CHAR][3])) / (tags_sum - score_dict[TAG_O_CHAR][3])

        average_dict = {"weighted avg": [wei_averageP, wei_averageR, wei_averageF1, tags_sum],
                        "weighted avg(excluding {})".format(TAG_O_CHAR): [wei_averageeOP, wei_averageeOR, wei_averageeOF1, tags_sum - score_dict[TAG_O_CHAR][3]]}


    # print(wei_averageeOF1)
    if print_detailed:
        res_pd=pd.DataFrame(np.zeros((len(score_dict)+len(average_dict)+1,4)))
        res_pd.columns=["Precision", "Recall", "F1", "Amount"]
        res_pd.index=list(score_dict.keys())+[" "]+list(average_dict.keys())
        res_pd.loc[" ",:]=[" "," "," "," "]
        for k,v in score_dict.items():
            res_pd.loc[k, :] = v
        for k,v in average_dict.items():
            res_pd.loc[k, :] = v
        print(res_pd)

    if TAG_O_CHAR not in score_dict:
        return wei_averageF1
    return wei_averageeOF1


def predict(model,xtest,device,Pos=None):
    model.eval()
    model=model.to(device)
    y_pred=[]
    with torch.no_grad():
        print("predict...")
        for i in tqdm(range(len(xtest))):
            X=torch.tensor(xtest[i]).reshape(1,-1).to(device)
            P=torch.tensor(Pos[i]).reshape(1,-1).to(device) if Pos else None

            y_pred.extend(model.predict(X,P=P))
    return y_pred


def pickleload(path):
    with open(path,"rb") as f:
        data=pickle.load(f)
    return data

X_test=pickleload(X_test_PATH)
y_test=pickleload(y_test_PATH)
word2id=pickleload(word2id_PATH)
tag2id=pickleload(tag2id_PATH)
P_test=pickleload(P_test_PATH) if USE_POS else None



y_true=[]
for i in y_test:
    y_true.extend(i)

model = torch.load(model_storage_PATH)
st=time.time()
y_pred=predict(model,X_test,device=DEVICE,Pos=P_test)
predict_time=time.time()-st

F1=evaluate(y_true,y_pred,tag2id,print_detailed=True)

evaluate_time=time.time()-st-predict_time

print("\npredict time: {:.3f}s, evaluate time: {:.3f}s, F1 score: {}".format(predict_time,evaluate_time,F1))

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

