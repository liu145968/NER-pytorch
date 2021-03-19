import numpy as np
import math,random,pickle,os

# Load Parameters
from Config import train_data_PATH,valid_data_PATH,test_data_PATH,model_storage_PATH,\
    word2id_PATH,tag2id_PATH,pos2id_PATH,USE_POS,word_vector_PATH,USE_pretrained_vector,\
    train_PATH,valid_PATH,test_PATH,X_test_PATH,y_test_PATH,P_test_PATH,\
    BATCH_SIZE,EMBED_SIZE,PAD,POS_X_CHAR,TAG_O_CHAR,\
    X_train_len_PATH,X_valid_len_PATH,X_test_len_PATH,SPLIT_SIGN



def mkdir(path):
    if os.path.exists(path.split("/")[0]):
        return
    else:
        os.mkdir(path.split("/")[0])

def data_encode(data,dic):
    return [[dic[s] for s in sent]for sent in data]


def gen_dictanddata(data_path,word2id,tag2id,split_sign="\t",pos2id={}):
    word_start_id=len(word2id)
    tag_start_id=len(tag2id)
    pos_start_id=len(pos2id)
    W, P, t = [], [], []
    W_sent,P_sent,t_sent=[],[],[]
    for line in open(data_path,encoding="utf-8"):
        if line!="\n":
            line_ls=line.strip("\n").split(split_sign)
            W_sent.append(line_ls[0])
            t_sent.append(line_ls[-1])
            P_sent.append(line_ls[1])
            if line_ls[0] not in word2id:
                word2id[line_ls[0]]=word_start_id
                word_start_id+=1
            if line_ls[-1] not in tag2id:
                tag2id[line_ls[-1]]=tag_start_id
                tag_start_id+=1
            if line_ls[1] not in pos2id:
                pos2id[line_ls[1]]=pos_start_id
                pos_start_id+=1
        else:
            if W_sent:
                W.append(W_sent)
                t.append(t_sent)
                P.append(P_sent)
                W_sent, P_sent, t_sent = [], [], []
    return word2id,tag2id,W,t,pos2id,P




def creat_train_data(X,y,batch_size,tag_O_id,shuffle=False,use_pos=False,pos=None,pos_x_id=None):
    lengths=[len(sen) for sen in X]
    len_sorted,sorted_id=np.sort(lengths),np.argsort(lengths)
    X,y=np.array(X),np.array(y)
    X_sorted,y_sorted=X[sorted_id],y[sorted_id]
    X_out,y_out,X_batch_len=[],[],[]
    if use_pos:
        P=np.array(pos)
        P_sorted=P[sorted_id]
        P_out=[]
    for i in range(math.ceil(len(lengths)/batch_size)):
        start=i*batch_size
        end=start+batch_size if start+batch_size<=len(lengths) else len(lengths)
        X_temp,y_temp=X_sorted[start:end],y_sorted[start:end]
        max_len=max(len_sorted[start:end])
        X_t_array,y_t_array=np.zeros((end-start,max_len),dtype=int),np.ones((end-start,max_len),dtype=int)*tag_O_id
        if use_pos:
            P_temp=P_sorted[start:end]
            P_t_array=np.ones((end-start,max_len),dtype=int)*pos_x_id
        for i in range(len(X_t_array)):
            X_t_array[i,0:len(X_temp[i])]=X_temp[i]
            y_t_array[i,0:len(y_temp[i])]=y_temp[i]
            if use_pos:
                P_t_array[i,0:len(P_temp[i])]=P_temp[i]
        X_out.append(X_t_array)
        y_out.append(y_t_array)
        if use_pos:
            P_out.append(P_t_array)
        X_batch_len.append(len_sorted[start:end])
    if shuffle:
        random.seed(10)
        random.shuffle(X_out)
        random.seed(10)
        random.shuffle(y_out)
        random.seed(10)
        random.shuffle(X_batch_len)
        if use_pos:
            random.seed(10)
            random.shuffle(P_out)
    if use_pos:
        return X_out,P_out,y_out,X_batch_len
    else:
        return X_out,[],y_out,X_batch_len

def read_save_wordvector(vector_load_path,embed_size,word2id):
    word_vector=np.random.randn(len(word2id.keys()),embed_size)
    wid_vec_dic={}
    for line in open(vector_load_path,encoding="utf-8"):
        if line[0]==" ":
            v=line[2:].strip("\n").split(" ")
            w=" "
        else:
            line=line.strip("\n").split(" ")
            w,v=line[0],line[1:]
        if w not in word2id:
            continue
        v=list(map(float,v))
        wid_vec_dic[word2id[w]]=v
    for i in range(word_vector.shape[0]):
        if i not in wid_vec_dic:
            continue
        word_vector[i,:]=np.array(wid_vec_dic[i],dtype=float)

    pickledump(word_vector,vector_load_path.split(".")[0]+"_temp.pickle")

def pickledump(data,path):
    with open(path,"wb") as f:
        pickle.dump(data,f)

mkdir(train_data_PATH)
mkdir(word2id_PATH)
mkdir(model_storage_PATH)

print("\nmaking dict and preparing data...")
word2id,tag2id,pos2id={PAD:0},{TAG_O_CHAR:0},{POS_X_CHAR:0}
word2id,tag2id,W_train,t_train,pos2id,P_train=gen_dictanddata(train_data_PATH,word2id,tag2id,split_sign=SPLIT_SIGN,pos2id=pos2id)
word2id,tag2id,W_valid,t_valid,pos2id,P_valid=gen_dictanddata(valid_data_PATH,word2id,tag2id,split_sign=SPLIT_SIGN,pos2id=pos2id)
word2id,tag2id,W_test,t_test,pos2id,P_test=gen_dictanddata(test_data_PATH,word2id,tag2id,split_sign=SPLIT_SIGN,pos2id=pos2id)

X_train,X_valid,X_test=[data_encode(W,word2id) for W in [W_train,W_valid,W_test]]
y_train,y_valid,y_test=[data_encode(t,tag2id) for t in [t_train,t_valid,t_test]]
P_train,P_valid,P_test=[data_encode(P,pos2id) for P in [P_train,P_valid,P_test]]
print("DONE!")

print("\nmaking train data...")
X_train_batch,P_train_batch,y_train_batch,X_train_len=creat_train_data(X=X_train,y=y_train,batch_size=BATCH_SIZE,
                                                                       tag_O_id=tag2id[TAG_O_CHAR],shuffle=True,
                                                                       use_pos=USE_POS,pos=P_train,
                                                                       pos_x_id=pos2id[POS_X_CHAR])
X_valid_batch,P_valid_batch,y_valid_batch,X_valid_len=creat_train_data(X=X_valid,y=y_valid,batch_size=BATCH_SIZE,
                                                                       tag_O_id=tag2id[TAG_O_CHAR],shuffle=True,
                                                                       use_pos=USE_POS,pos=P_valid,
                                                                       pos_x_id=pos2id[POS_X_CHAR])
X_test_batch,P_test_batch,y_test_batch,X_test_len=creat_train_data(X=X_test,y=y_test,batch_size=BATCH_SIZE,
                                                                   tag_O_id=tag2id[TAG_O_CHAR],shuffle=True,
                                                                   use_pos=USE_POS,pos=P_test,
                                                                   pos_x_id=pos2id[POS_X_CHAR])
print("DONE!")

print("\nsaving data...")
pickledump(word2id,word2id_PATH)
pickledump(tag2id,tag2id_PATH)
pickledump(pos2id,pos2id_PATH)

if USE_POS:
    pickledump([X_train_batch,P_train_batch,y_train_batch],train_PATH)
    pickledump([X_valid_batch,P_valid_batch,y_valid_batch],valid_PATH)
    pickledump([X_test_batch,P_test_batch,y_test_batch],test_PATH)
else:
    pickledump([X_train_batch, y_train_batch], train_PATH)
    pickledump([X_valid_batch, y_valid_batch], valid_PATH)
    pickledump([X_test_batch, y_test_batch], test_PATH)

pickledump(X_train_len,X_train_len_PATH)
pickledump(X_valid_len,X_valid_len_PATH)
pickledump(X_test_len,X_test_len_PATH)

pickledump(X_test,X_test_PATH)
pickledump(y_test,y_test_PATH)
if USE_POS:
    pickledump(P_test,P_test_PATH)
print("DONE!")

if USE_pretrained_vector:
    print("\npreparing vector...")
    read_save_wordvector(vector_load_path=word_vector_PATH,embed_size=EMBED_SIZE,word2id=word2id)
    print("DONE!")