from Model import NER
from tqdm import tqdm
import pickle,time,torch

from Config import word2id_PATH,tag2id_PATH,pos2id_PATH,train_PATH,valid_PATH,test_PATH,\
    X_train_len_PATH,X_valid_len_PATH,EMBED_SIZE,HIDDEN_SIZE,POS_EMBED_SIZE,DEVICE,LR,EPOCH,\
    USE_pretrained_vector,word_vector_PATH,model_storage_PATH,MODEL_type,USE_POS,n_heads,DROPOUT_RATE

def validate(model,validdata,validdata_len,device):
    model.eval()
    model = model.to(device)

    data, pos, tags = validdata[0], validdata[1],validdata[-1]

    start_time, valid_loss = time.time(), 0
    print("\nvalidate...")
    for i in tqdm(range(len(data))):
        sents, target = torch.from_numpy(data[i]).long().to(device), torch.from_numpy(tags[i]).long().to(device)

        (score_mask, att_mask), lengths = get_mask(validdata_len[i], device), torch.from_numpy(validdata_len[i]).to(device)

        P = torch.from_numpy(pos[i]).long().to(device) if USE_POS else None

        loss = model.compute_loss(sents, target, lengths, score_mask, att_mask=att_mask, P=P).mean()

        valid_loss += loss.item()
    print("valid time:{:.3f} s ,valid loss:{}".format(time.time() - start_time, valid_loss))
    return valid_loss

def get_score_att_mask(batchdata_len,device):
    batch_size,max_len=batchdata_len.shape[0],batchdata_len[-1]
    score_mask,att_mask,mask_temp=[],torch.tensor([]),torch.ones((max_len, max_len))
    for i in batchdata_len:
        score_mask.extend([1.0]*i+[0.0]*(max_len-i))
        mask_temp[:i,:i]=torch.zeros((i,i))
        att_mask=torch.cat([att_mask,mask_temp.repeat(n_heads,1,1)])
    return torch.tensor(score_mask).reshape(batch_size,max_len).to(device),-100*att_mask.to(device)

def get_score_mask(batchdata_len,device):
    return torch.cat([torch.tensor([1.0]*i+[0.0]*(batchdata_len[-1]-i)) for i in batchdata_len]).reshape(len(batchdata_len),-1).to(device)

def get_mask(batchdata_len,device):
    if "transformer" in MODEL_type:
        return get_score_att_mask(batchdata_len,device)
    else:
        return get_score_mask(batchdata_len,device),None

def train(model,traindata,traindata_len,validdata,validdate_len,optimizer,epoch,device,print_train_time=False):
    best_loss,best_epoch=validate(model,validdata,validdate_len,device),-1

    model.train()
    model=model.to(device)

    data, pos, tags=traindata[0],traindata[1],traindata[-1]
    start_time=time.time()

    for e in range(epoch):
        print("\nThe {} of {} epoch...".format(e+1,epoch))
        curtrain_stime,curtrain_loss=time.time(),0
        for i in tqdm(range(len(data))):
            sents, target = torch.from_numpy(data[i]).long().to(device), torch.from_numpy(tags[i]).long().to(device)

            (score_mask,att_mask),lengths=get_mask(traindata_len[i],device),torch.from_numpy(traindata_len[i]).to(device)

            P=torch.from_numpy(pos[i]).long().to(device) if USE_POS else None

            loss=model.compute_loss(sents,target,lengths,score_mask,att_mask=att_mask,P=P).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curtrain_loss+=loss.item()
        print("train time:{:.3f} s , train loss:{}".format(time.time()-curtrain_stime,curtrain_loss))
        valid_loss=validate(model,validdata,validdate_len,device)
        model.train()
        if valid_loss<best_loss:
            best_loss,best_epoch=valid_loss,e
            torch.save(model,model_storage_PATH)
            print("the model is saved , best loss: {}".format(best_loss))

    if print_train_time:
        print("time for training and validating:{:.3f} s ".format(time.time()-start_time),"epoch:",epoch,
              "best epoch:",best_epoch+1)

def pickleload(path):
    with open(path,"rb") as f:
        data=pickle.load(f)
    return data

word2id=pickleload(word2id_PATH)
tag2id=pickleload(tag2id_PATH)
pos2id=pickleload(pos2id_PATH)

train_data=pickleload(train_PATH)
valid_data=pickleload(valid_PATH)
test_data=pickleload(test_PATH)

X_train_len=pickleload(X_train_len_PATH)
X_valid_len=pickleload(X_valid_len_PATH)

model=NER(vocab_size=len(word2id),embed_size=EMBED_SIZE,hidden_size=HIDDEN_SIZE,num_tags=len(tag2id),device=DEVICE,
          dropout_rate=DROPOUT_RATE,pos_vocab_size=len(pos2id),pos_embed_size=POS_EMBED_SIZE)

if USE_pretrained_vector:
    model.embed.weight.data.copy_(torch.from_numpy(pickleload(word_vector_PATH.split(".")[0]+"_temp.pickle")))


# optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=LR)
optimizer=torch.optim.Adam(model.parameters(),lr=LR)

train(model=model,traindata=train_data,traindata_len=X_train_len,validdata=valid_data,validdate_len=X_valid_len,
      optimizer=optimizer,epoch=EPOCH,device=DEVICE,print_train_time=True)

# save complete model
# torch.save(model,model_storage_PATH)
# print("The model has been completely saved! The path is:",model_storage_PATH)

# Only save the parameters of model
# torch.save(model.state_dict(),model_storage_path)
# print("model is saved! the path is:",model_storage_path)


print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))