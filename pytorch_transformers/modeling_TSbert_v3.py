import torch
from torch import nn
from torch.nn import BCELoss
from .modeling_bert import BertPreTrainedModel,BertModel
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.functional as F
from torch.autograd import Variable
def masked_softmax(vector, mask):
    mask = Variable(mask, requires_grad=False)
    result = torch.nn.functional.softmax(vector * mask, dim=-1)
    # a=(vector*mask).view(-1)
    # b=vector.view(-1)
    # for i, j in zip(a, b):
    #     if i != j:
    #         print(i, j)

    result = result * mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result

class TransformerBlock(nn.Module):

    def __init__(self, device,input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.device = device
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V,V_mask, episilon=1e-8):
        '''
        :param Q: (batch_size*, max_r_words, embedding_dim)
        :param K: (batch_size*, max_u_words, embedding_dim)
        :param V: (batch_size*, max_u_words, embedding_dim)
        :param V_mask: (batch_size*, max_u_words)
        :return: output: (batch_size*, max_r_words, embedding_dim)  same size as Q
        '''
        dk = torch.Tensor([max(1.0, Q.size(-1))]).to(self.device)

        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon) #(batch_size, max_r_words, max_u_words)

        Q_K_score=masked_softmax(Q_K,V_mask[:,None,:])
        # Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_r_words, max_u_words)

        V_att = Q_K_score.bmm(V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        return output


class BertForSequenceClassificationTSv3(BertPreTrainedModel):
    def __init__(self, config, max_seg_num,max_seq_len,device):
        super(BertForSequenceClassificationTSv3, self).__init__(config)
        self.config=config
        # self.seq_len=seq_len
        # self.finaldim=finaldim
        # self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.alpha = 0.5
        self.device=device
        self.linear_word = nn.Linear(2 * max_seq_len, 1) ##max_Seq_len
        self.W_word = nn.Parameter(data=torch.Tensor(self.config.hidden_size, self.config.hidden_size, max_seg_num))
        self.v = nn.Parameter(data=torch.Tensor(max_seg_num, 1))

        self.transformer_ur = TransformerBlock(device=device,input_size=self.config.hidden_size)
        self.transformer_ru = TransformerBlock(device=device,input_size=self.config.hidden_size)
        self.AU1 = nn.Parameter(data=torch.Tensor(self.config.hidden_size, self.config.hidden_size))
        # self.AU2 = nn.Parameter(data=torch.Tensor(self.config.hidden_size, self.config.hidden_size))
        self.AU3 = nn.Parameter(data=torch.Tensor(self.config.hidden_size, self.config.hidden_size))

        # self.utt_cnn_2d_1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3))
        # self.utt_maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.utt_cnn_2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        # self.utt_maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.utt_cnn_2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        # self.utt_maxpooling3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.utt_affine2 = nn.Linear(in_features=3 * 3 * 64, out_features=self.finaldim)
        self.key_trans = nn.Linear(in_features=2*self.config.hidden_size, out_features=2*self.config.hidden_size)

        self.utt_gru_acc = nn.GRU(input_size=2*self.config.hidden_size, hidden_size=2*self.config.hidden_size, batch_first=True)
        self.affine_out = nn.Linear(in_features=4*self.config.hidden_size, out_features=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.loss_func=BCELoss()
        self.init_weights()

    # def my_init_weights(self):
    #     init.uniform_(self.W_word)
    #     init.uniform_(self.v)
    #     init.uniform_(self.linear_word.weight)
    #     # init.uniform_(self.linear_score.weight)
    #
    #     init.xavier_normal_(self.AU1)
    #     init.xavier_normal_(self.AU2)
    #     init.xavier_normal_(self.AU3)
    #     init.xavier_normal_(self.utt_cnn_2d_1.weight)
    #     init.xavier_normal_(self.utt_cnn_2d_2.weight)
    #     init.xavier_normal_(self.utt_cnn_2d_3.weight)
    #     init.xavier_normal_(self.utt_affine2.weight)
    #     for weights in [self.utt_gru_acc.weight_hh_l0, self.utt_gru_acc.weight_ih_l0]:
    #         init.orthogonal_(weights)
    #     init.xavier_normal_(self.key_trans.weight)
    #
    #     init.xavier_normal_(self.affine_out.weight)

    def word_selector(self, key, context,segment_turnmask):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz,max_utterances, max_u_words, d)
        :return: score:
        '''
        # print("key.size():",key.size())
        # print("context.size()",context.size())
        dk = torch.sqrt(torch.Tensor([self.config.hidden_size])).to(self.device)
        A = torch.tanh(torch.einsum("blrd,ddh,bud->blruh", context, self.W_word, key)/dk)
        A = torch.einsum("blruh,hp->blrup", A, self.v).squeeze(dim=-1)   # b x l x u x u

        a = torch.cat([A.max(dim=2)[0], A.max(dim=3)[0]], dim=-1) # b x l x 2u
        a=self.linear_word(a).squeeze(dim=-1)
        mask=(1.0 - segment_turnmask) * -10000.0
        mask = Variable(mask, requires_grad=False)
        s1 = torch.softmax(a+mask, dim=-1)  # b x l
        return s1

    def utterance_selector(self, key, context,segment_turnmask):
        '''
        :param key:  (bsz, max_u_words, d)
        :param context:  (bsz,max_utterances, max_u_words, d)
        :return: score:
        '''
        key = key.mean(dim=1)
        context = context.mean(dim=2)
        s2 = torch.einsum("bud,bd->bu", context, key)/(1e-6 + torch.norm(context, dim=-1)*torch.norm(key, dim=-1, keepdim=True) )
        mask = (1.0 - segment_turnmask) * -10000.0
        mask = Variable(mask, requires_grad=False)
        s2 = torch.softmax(s2+mask, dim=-1)
        return s2

    def my_context_selector(self,segment,response,segment_turnmask):
        '''
        :param seg_context: (batch_size, max_segments, max_u_words, embedding_dim)
        :return:
        '''

        # su1, su2, su3, su4 = utt_context.size()
        # utt_context_ = utt_context.view(-1, su3, su4)  # (batch_size*max_utterances, max_u_words, embedding_dim)
        # utt_context_ = self.selector_transformer_utt(utt_context_, utt_context_, utt_context_)
        # utt_context_ = utt_context_.view(su1, su2, su3, su4)
        #
        # ss1, ss2, ss3, ss4 = seg_context.size()
        # seg_context_ = seg_context.view(-1, su3, su4)  # (batch_size*max_segments, max_u_words, embedding_dim)
        # seg_context_ = self.selector_transformer_seg(seg_context_, seg_context_, seg_context_)
        # seg_context_ = seg_context_.view(ss1, ss2, ss3, ss4)

        # multi_match_score = []
        # print("seg_context size",seg_context.size())
        # key = seg_context[:, -1:, :, :].mean(dim=1)
        # print("hiskey.size1()", key.size())
        # key = self.selector_transformer_seg(key, key, key)
        # print("hiskey.size2()",key.size())
        #
        # print("seg_context_.size()",seg_context_.size())
        # key = seg_context[:, -1:, :, :].mean(dim=1)

        # utt_score1 = self.word_selector(response,segment)
        # utt_score2 = self.utterance_selector(response,segment)
        # utt_score = self.alpha * utt_score1 + (1 - self.alpha) * utt_score2
        # multi_match_score.append(utt_score)

        seg_score1 = self.word_selector(response,segment,segment_turnmask)
        seg_score2 = self.utterance_selector(response,segment,segment_turnmask)
        seg_score = self.alpha * seg_score1 + (1 - self.alpha) * seg_score2

        # multi_match_score.append(seg_score)

        # multi_match_score = torch.stack(multi_match_score, dim=-1)  # [batch,max_turn,hop]
        # match_score = self.linear_score(multi_match_score).squeeze()  # [batch,max_turn]

        # mask_utt = (utt_score >= self.gamma).float()
        # mask_seg = (seg_score >= self.gamma).float()
        # match_score_utt = utt_score * mask_utt
        # match_score_seg=  seg_score*mask_seg
        # match_score_utt = utt_score
        match_score_seg = seg_score

        # select_utt_context = utt_context * match_score_utt.unsqueeze(dim=-1).unsqueeze(dim=-1)
        select_seg_context = segment * match_score_seg.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return select_seg_context

    # def distance(self, A, B, C, epsilon=1e-6):
    #     M1 = torch.einsum("bud,dd,brd->bur", [A, B, C])
    #
    #     A_norm = A.norm(dim=-1)
    #     C_norm = C.norm(dim=-1)
    #     M2 = torch.einsum("bud,brd->bur", [A, C]) / (torch.einsum("bu,br->bur", A_norm, C_norm) + epsilon)
    #     return M1, M2
    #
    #
    #
    # def get_Matching_Map_utt(self, bU_embedding, bR_embedding):
    #     '''
    #     :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
    #     :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
    #     :return: E: (bsz*max_utterances, max_u_words, max_r_words)
    #     '''
    #     # M1 = torch.einsum("bud,dd,brd->bur", bU_embedding, self.A1, bR_embedding)  # (bsz*max_utterances, max_u_words, max_r_words)
    #     MU1, MU2 = self.distance(bU_embedding, self.AU1, bR_embedding)
    #     # Hu = self.transformer_utt(bU_embedding, bU_embedding, bU_embedding)
    #     # Hr = self.transformer_res(bR_embedding, bR_embedding, bR_embedding)
    #     # # M2 = torch.einsum("bud,dd,brd->bur", [Hu, self.A2, Hr])
    #     # MU3, MU4 = self.distance(Hu, self.AU2, Hr)
    #     Hur = self.transformer_ur(bU_embedding, bR_embedding, bR_embedding)
    #     Hru = self.transformer_ru(bR_embedding, bU_embedding, bU_embedding)
    #     # M3 = torch.einsum("bud,dd,brd->bur", [Hur, self.A3, Hru])
    #     MU5, MU6 = self.distance(Hur, self.AU3, Hru)
    #
    #     MU = torch.stack([MU1, MU2,  MU5, MU6], dim=1)# (bsz*max_utterances, channel, max_u_words, max_r_words)
    #     return MU

    # def UR_Matching_utt(self, bU_embedding, bR_embedding):
    #     '''
    #     :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
    #     :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
    #     :return: (bsz*max_utterances, (max_u_words - width)/stride + 1, (max_r_words -height)/stride + 1, channel)
    #     '''
    #     M= self.get_Matching_Map_utt(bU_embedding, bR_embedding)
    #
    #     Z = self.relu(self.utt_cnn_2d_1(M)) #50*50 48*48
    #     Z = self.utt_maxpooling1(Z)#24*24
    #
    #     Z = self.relu(self.utt_cnn_2d_2(Z)) #22*22
    #     Z =self.utt_maxpooling2(Z) #11*11
    #
    #     Z = self.relu(self.utt_cnn_2d_3(Z)) #9*9
    #     Z =self.utt_maxpooling3(Z) #batch_size*max_utterances, 64,3,3
    #
    #     Z = Z.view(Z.size(0), -1)  # (bsz*max_utterances, *)
    #
    #     V = self.tanh(self.utt_affine2(Z))   # (bsz*max_utterances, 300)
    #     return V

    def MatchingNet(self,select_seg_context,segment_mask_seg,response_seg,response_mask_seg):
        '''

        :param select_seg_context: batchsize(*max_segment_num),seq_len,dim
        :param segment_mask_seg: batchsize(*max_segment_num),seq_len
        :param response_seg: batchsize(*max_segment_num),seq_len,dim
        :param response_mask_seg: batchsize(*max_segment_num),dim*2
        :return:
        '''
        Hur = self.transformer_ur(select_seg_context, response_seg,response_seg,response_mask_seg)
        Hru = self.transformer_ru(response_seg,select_seg_context,select_seg_context,segment_mask_seg)
        result=torch.cat([torch.mean(Hur,dim=1), torch.mean(Hru,dim=1) ], dim=1)
        return result

    def forward(self, seg_input_ids,seg_token_type_ids, seg_attention_mask,cls_sep_pos,true_len,labels=None):
        self.utt_gru_acc.flatten_parameters()
        # b,u,w=utt_input_ids.size()
        # _,s,_=seg_input_ids.size()
        #utt_input_ids=(batchsize,max_utts,max_words)
        #seg_input_ids=(batchsize,max_segs,max_words)
        #res_input_ids=(batchsize,max_words)

        b, s = seg_input_ids.size()
        n = cls_sep_pos.size()[1] - 2
        # print(cls_sep_pos.size())
        # utt_sequence_output, utt_sequence_pool = self.bert(utt_input_ids.view(-1,w),attention_mask=utt_attention_mask.view(-1,w),token_type_ids=utt_token_type_ids.view(-1,w))
        # seg_sequence_output, seg_sequence_pool = self.bert(seg_input_ids.view(-1,w),attention_mask= seg_attention_mask.view(-1,w),token_type_ids=seg_token_type_ids.view(-1,w),)
        # res_sequence_output, res_sequence_pool = self.bert(res_input_ids, attention_mask=res_attention_mask,token_type_ids=res_token_type_ids)

        sequence_output, pooled_output = self.bert(input_ids=seg_input_ids,
                                                   attention_mask=seg_attention_mask,
                                                   token_type_ids=seg_token_type_ids)

        _,_,d=sequence_output.size()
        # b,max_seg_num,max_seq_len,dim
        segment = torch.zeros(b,n,s, d).to(self.device)
        segment_mask = torch.zeros(b , n, s).to(self.device)
        segment_turnmask=torch.zeros(b , n).to(self.device)

        # b,max_seq_len,dim
        response = torch.zeros(b, s, d).to(self.device)
        response_mask = torch.zeros(b , s).to(self.device)

        for bind,seq in enumerate(sequence_output):
            cls_seq_pos_temp=cls_sep_pos[bind][:true_len[bind]]
            for posind,pos in enumerate(cls_seq_pos_temp):
                if(posind==true_len[bind]-1):
                    break
                m = cls_seq_pos_temp[posind + 1] - cls_seq_pos_temp[posind] - 1
                if(posind==true_len[bind]-2):
                    response[bind][0:m] = sequence_output[bind][cls_seq_pos_temp[posind] + 1:cls_seq_pos_temp[posind + 1]]
                    response_mask[bind][0:m]= 1
                else:
                    segment[bind][posind][0:m]=\
                        sequence_output[bind][cls_seq_pos_temp[posind]+1:cls_seq_pos_temp[posind+1]]
                    segment_mask[bind][posind][0:m] =1
                    segment_turnmask[bind][posind]=1

        # utt_sequence_output=utt_sequence_output.view(b,u,w,d) #[batch_size,max_utts,max_words,dim]
        # seg_sequence_output=seg_sequence_output.view(b,s,w,d) #[batch_size,max_utt_len,max_words,dim]
        # #res_sequence_output [batch_size,max_words,dim]

        #segment b,n,s,d mask n,n,s
        #response b.s.d mask b,s
        # m = torch.gt(segment * segment_mask.unsqueeze(dim=-1), segment)
        # w=(segment * segment_mask.unsqueeze(dim=-1)).view(-1)
        # e=segment.view(-1)
        # for i,j in zip(a,b):
        #     if i!=j:
        #         print(i,j)

        select_seg_context = self.my_context_selector(segment,response,segment_turnmask)
        # m = torch.gt(select_seg_context * segment_mask.unsqueeze(dim=-1), select_seg_context)
        # w = (select_seg_context * segment_mask.unsqueeze(dim=-1)).view(-1)
        # e = select_seg_context.view(-1)
        # for i, j in zip(a, b):
        #     if i != j:
        #         print(i, j)
        # select_utt_context=select_utt_context.view(-1,w,d)
        select_seg_context=select_seg_context.view(b*n,s,d)
        segment_mask_seg = segment_mask.view(b * n, s)

        # res_sequence_output_utt=res_sequence_output.unsqueeze(dim=1).repeat(1, u, 1, 1)
        # res_sequence_output_utt=res_sequence_output_utt.view(-1,w,d)
        res_sequence_output_seg= response.unsqueeze(dim=1).repeat(1, n, 1, 1)
        response_mask_seg=response_mask.unsqueeze(dim=1).repeat(1, n, 1, 1)
        res_sequence_output_seg=res_sequence_output_seg.view(b*n,s,d)
        response_mask_seg = response_mask_seg.view(b * n, s)

        # V_utt = self.UR_Matching_utt(select_utt_context, res_sequence_output_utt)
        # V_utt = V_utt.view(b, u, -1)  # (bsz, max_utterances, 300)

        V_seg = self.MatchingNet(select_seg_context,segment_mask_seg,res_sequence_output_seg,response_mask_seg)
        V_seg=V_seg.view(b, n, 2*d)

        V_key = self.MatchingNet(segment[:, -1:, :, :].mean(dim=1),segment_mask[:, -1:, :].mean(dim=1), response,response_mask)  # (bsz,2dim)

        # H_utt, _ = self.utt_gru_acc(V_utt)  # (bsz, max_utterances, rnn2_hidden)
        H_seg, _ = self.utt_gru_acc(V_seg)  # (bsz, max_segments, rnn2_hidden)
        H_key = self.key_trans(V_key)
        # L = self.attention(V, u_mask_sent)
        # L_utt = self.dropout(H_utt[:, -1, :])  # (bsz, rnn2_hidden)
        L_seg = self.dropout(H_seg[:, -1, :])  # (bsz, rnn2_hidden)
        L_key = self.dropout(H_key)  # (bsz, rnn2_hidden)

        logits = torch.sigmoid(self.affine_out(torch.cat((L_seg, L_key), 1))).squeeze(dim=-1)
        # labels=torch.FloatTensor(labels)

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss = self.loss_func(logits, target=labels)
            return logits,loss
        else:
            return logits
