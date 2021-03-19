import torch

train_data_PATH= "data/train.txt"
valid_data_PATH="data/valid.txt"
test_data_PATH= "data/test.txt"


BATCH_SIZE=64
EPOCH=20
LR=0.001
EMBED_SIZE=100
HIDDEN_SIZE=150
USE_POS=False
POS_EMBED_SIZE=30 if USE_POS else 0

DROPOUT_RATE=0.1

MODEL_type="lstm+crf"
n_transformer_layers=1
n_heads=5

USE_pretrained_vector=False
word_vector_PATH="data/wordvec100d.txt"

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE=torch.device("cpu")

model_storage_PATH="model/model_NER.pth"

TAG_O_CHAR="O"
POS_X_CHAR="x"
PAD="<pad>"
SPLIT_SIGN="\t"



word2id_PATH= "temp/word2id_dict.pickle"
pos2id_PATH="temp/pos2id_dict.pickle"
tag2id_PATH= "temp/tag2id_dict.pickle"


train_PATH="temp/train_ls.pickle"
valid_PATH="temp/valid_ls.pickle"
test_PATH="temp/test_ls.pickle"

X_train_len_PATH= "temp/X_train_len_ls.pickle"
X_valid_len_PATH="temp/X_valid_len_ls.pickle"
X_test_len_PATH= "temp/X_test_len_ls.pickle"

X_test_PATH= "temp/X_test_OR_ls.pickle"
y_test_PATH= "temp/y_test_OR_ls.pickle"
P_test_PATH="temp/P_test_OR_ls.pickle"

'''
python data.py
python train.py
python evaluate.py

'''