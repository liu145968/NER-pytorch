import torch,sys
import numpy as np
import torch.nn as nn
from Config import USE_POS,POS_EMBED_SIZE,MODEL_type,n_transformer_layers,n_heads


class NER(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_tags,device,dropout_rate=0.1,pos_vocab_size=0,pos_embed_size=POS_EMBED_SIZE):
        super(NER, self).__init__()
        self.device=device
        self.num_tags=num_tags
        self.EPS = 1e-8

        self.embed=nn.Embedding(vocab_size,embed_size)
        if USE_POS:
            self.pos_embed=nn.Embedding(pos_vocab_size,pos_embed_size)

        self.lstm=nn.LSTM(embed_size+pos_embed_size,hidden_size,bidirectional=True,batch_first=True) if USE_POS else \
            nn.LSTM(embed_size,hidden_size,bidirectional=True,batch_first=True)

        self.linear=nn.Linear(2*hidden_size,num_tags)

        self.transformer_embed_linear=nn.Linear(embed_size+pos_embed_size,num_tags)

        self.trans_matrix=nn.Parameter(torch.randn(num_tags,num_tags))

        self.transformer_embed=nn.ModuleList(
            [transformer(n_heads,embed_size+pos_embed_size,4*(embed_size+pos_embed_size),dropout_rate,device) for _ in range(n_transformer_layers)]
        )

        self.transformer_blocks=nn.ModuleList(
            [transformer(n_heads,2*hidden_size,8*hidden_size,dropout_rate,device) for _ in range(n_transformer_layers)])

        self.dropout=nn.Dropout(dropout_rate)



    def forward(self,X,X_len=None,P=None,att_mask=None):
        X_embed=torch.cat([self.embed(X),self.pos_embed(P)],dim=2) if USE_POS else self.embed(X)

        X_len=torch.tensor([X.shape[1]]*X.shape[0]).to(self.device) if X_len is None else X_len

        if MODEL_type[:11]=="transformer":
            for t in self.transformer_embed:
                X_embed=t.forward(X_embed,att_mask)
            if MODEL_type=="transformer+crf":
                return self.transformer_embed_linear(self.dropout(X_embed))


        X_embed_packed=nn.utils.rnn.pack_padded_sequence(X_embed,X_len.to(torch.device("cpu")),batch_first=True,enforce_sorted=False)
        X_embed_packed_hid,_=self.lstm(X_embed_packed)
        X_embed_hid,_=nn.utils.rnn.pad_packed_sequence(X_embed_packed_hid,batch_first=True)

        if MODEL_type=="lstm+transformer+crf":
            for t in self.transformer_blocks:
                X_embed_hid=t.forward(X_embed_hid,att_mask)

        output=self.linear(self.dropout(X_embed_hid))


        return output


    def compute_loss(self,batch_sent,batch_tags,batch_length,score_mask,att_mask=None,P=None):
        tag_score=self.forward(batch_sent,batch_length,P,att_mask)


        emission_score=torch.sum(tag_score.gather(dim=2, index=batch_tags.unsqueeze(2)).squeeze(2)*score_mask,dim=1)

        trans_score=torch.zeros(size=(batch_tags.shape[0],)).to(self.device)
        if batch_tags.shape[1]>1:
            windows=batch_tags.unfold(dimension=1,size=2,step=1).reshape(-1,2)
            idx1,idx2=windows[:,0],windows[:,1]
            trans_score=(self.trans_matrix[idx1].gather(dim=1,index=idx2.reshape(-1,1)).squeeze(1).reshape(-1,batch_tags.shape[1]-1)*score_mask[:,1:]).sum(dim=1)

        sents_score=emission_score+trans_score

        log_norm_score=self.forward_algo_exp(tag_score,score_mask)

        log_prob=sents_score-log_norm_score

        return -log_prob

    def forward_algo_exp(self,tags_score,score_mask):
        start=tags_score[:,0,:].unsqueeze(2)
        for i in range(1,tags_score.shape[1]):
            M=self.trans_matrix+tags_score[:,i,:].unsqueeze(1).expand(-1,self.num_tags,-1)
            mask_i=score_mask[:,i].reshape(-1,1)
            start=(torch.logsumexp((start.expand(-1,-1,self.num_tags)+M),dim=1)*mask_i+start.squeeze(2)*(1-mask_i)).unsqueeze(2)
        return torch.logsumexp(start.squeeze(2),dim=1)

    def log(self,x):
        return torch.log(x+self.EPS)

    def predict(self,X,P=None):
        # output=self.lstm(torch.cat([self.embed(X),self.pos_embed(P)],dim=2))[0] if USE_POS else self.lstm(self.embed(X))[0]
        #
        # if "transformer" in MODEL_type:
        #     for t in self.transformer_blocks:
        #         output=t.forward(output)
        #
        # pred_tag_prob = self.linear(output).reshape(-1, self.num_tags).cpu().numpy()
        # # pred_tag_prob = self.log(torch.softmax(self.linear(output).reshape(-1, self.num_tags).cpu(), dim=-1)).numpy()

        pred_tag_prob=self.forward(X,P=P).reshape(-1,self.num_tags).cpu().numpy()

        trans=self.trans_matrix.cpu().detach().numpy()
        # trans = self.log(torch.softmax(self.trans_matrix,dim=1)).cpu().detach().numpy()


        distance=pred_tag_prob[0,:]
        max_idx=[]

        for i in range(1,pred_tag_prob.shape[0]):
            max_idx_temp=[]
            distance_temp=np.zeros((self.num_tags,))
            for j in range(self.num_tags):
                temp=distance+trans[:,j]+pred_tag_prob[i][j]
                distance_temp[j]=np.max(temp)
                max_idx_temp.append(np.argmax(temp))

            max_idx.append(max_idx_temp)
            distance=distance_temp

        idx=[np.argmax(distance)]
        for i in range(len(max_idx)-1,-1,-1):
            idx.append(max_idx[i][idx[-1]])
        idx.reverse()

        return idx



class transformer(nn.Module):
    def __init__(self,n_heads,hidden_size,ff_size,dropout_rate,device):
        super(transformer, self).__init__()
        self.device=device
        self.h = n_heads
        self.d = hidden_size // n_heads
        self.Wq = nn.Parameter(torch.randn(size=(self.h, hidden_size, self.d)))
        self.Wk = nn.Parameter(torch.randn(size=(self.h, hidden_size, self.d)))
        self.Wv = nn.Parameter(torch.randn(size=(self.h, hidden_size, self.d)))
        self.Wo = nn.Parameter(torch.randn(size=(hidden_size, hidden_size)))
        self.LN = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),  # dimension-feed_forward_hidden, usually 4*hidden_size
            nn.ReLU(),  # the GELU instead of RELU
            nn.Linear(ff_size, hidden_size)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X,att_mask=None):  # X: [batch_size, sen_len, hidden_size]
        batch_size, sen_len = X.shape[0], X.shape[1]

        Q = torch.matmul(X.unsqueeze(1), self.Wq.unsqueeze(0))  # [batch_size, h, sen_len, d]
        K = torch.matmul(X.unsqueeze(1), self.Wk.unsqueeze(0))
        V = torch.matmul(X.unsqueeze(1), self.Wv.unsqueeze(0))

        weight_mask = att_mask if att_mask is not None else torch.zeros((batch_size * self.h, sen_len, sen_len)).to(self.device)
        weight = torch.softmax(
            torch.bmm(Q.reshape(-1, sen_len, self.d), K.reshape(-1, sen_len, self.d).permute(0, 2, 1)) / pow(self.d,0.5) + weight_mask,dim=2)

        X_attention = torch.bmm(weight, V.reshape(-1, sen_len, self.d)).reshape(batch_size, self.h, sen_len, self.d)
        X_attention = X + self.dropout(
            torch.matmul(self.LN(X_attention.permute(0, 1, 3, 2).reshape(batch_size, -1, sen_len).permute(0, 2, 1)),self.Wo))
        # X_atten_addnorm=self.LN(X_attention+X)  # Add&Norm

        output = X_attention + self.dropout(self.ff(self.LN(X_attention)))

        # X_embed = nn.utils.rnn.pack_padded_sequence(X_embed, X_len, batch_first=True, enforce_sorted=False) if X_len!=None else X_embed
        # X_embed, _ = self.lstm(X_embed)
        # X_embed, _ = nn.utils.rnn.pad_packed_sequence(X_embed, batch_first=True)  if X_len!=None else X_embed, 0

        return output