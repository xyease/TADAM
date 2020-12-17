import os
import sys
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from pytorch_transformers.modeling_TSbert_v3 import BertForSequenceClassificationTSv3
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.configuration_bert import BertConfig
from pytorch_transformers.optimization_bert import WarmupLinearSchedule,BertAdam
import argparse
import torch
import random
import numpy as np
import pickle
import logging
from tqdm import tqdm, trange
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from utils_TSbert_v3 import MyDataProcessorSegres,convert_examples_to_features_Segres,Metrics
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# from torch.utils.data.distributed import DistributedSampler


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
## Required parameters
# parser.add_argument("--finaldim",
#                     default=300,
#                     type=int,
#                     help="..")
parser.add_argument("--data_dir",
                    default="data/",
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default="bert-base-chinese", type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--task_name",
                    default="alime",
                    type=str,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default="model_save_v3",
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--temp_score_file_path",
                    default="temp_score_file.txt",
                    type=str,
                    help="temp score_file_path where the model predictions will be written for metrics.")
parser.add_argument("--log_save_path",
                    default="log.txt",
                    type=str,
                    help="log written when training")
parser.add_argument("--max_segment_num",
                    default=10,
                    type=int,
                    help="The maximum total segment number.")
parser.add_argument("--max_seq_length",
                    default=350,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--input_cache_dir",
                    default="input_cache_v3",
                    type=str,
                    help="Where do you want to store the processed model input")
parser.add_argument("--do_train",
                    type=bool,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_lower_case",
                    default=True,
                    type=bool,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--num_train_epochs",
                    default=5,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--train_batch_size",
                    default=20,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
args = parser.parse_args()
args.temp_score_file_path=os.path.join(args.data_dir,args.task_name,args.output_dir,args.temp_score_file_path)
args.log_save_path=os.path.join(args.data_dir,args.task_name,args.output_dir,args.log_save_path)
args.output_dir=os.path.join(args.data_dir,args.task_name,args.output_dir)
args.input_cache_dir=os.path.join(args.data_dir, args.task_name, args.input_cache_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.exists(args.input_cache_dir):
    os.makedirs(args.input_cache_dir)

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(args.log_save_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)
logger.info(args)

def set_seed():
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子


def get_dataloader(tokenizer,examples,label_list,tag):
    logger.info("start prepare input data")

    cached_train_features_file = os.path.join(args.input_cache_dir,tag+"input.pkl")
    # train_features = None

    try:
        with open(cached_train_features_file, "rb") as reader:
            features= pickle.load(reader)
    except:
        logger.info("start prepare features_res_lab")
        features = convert_examples_to_features_Segres(
            examples, label_list, max_seg_num=args.max_segment_num,max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        # logger.info("start prepare features_utt")
        # features_utt = convert_examples_to_features(
        #     examples_utt, label_list, args.max_seq_length, tokenizer)
        # logger.info("start prepare features_seg")
        # features_seg = convert_examples_to_features(examples_seg, label_list, args.max_seq_length, tokenizer)
        # # if args.local_rank == -1:
        logger.info("  Saving train features into cached file %s", cached_train_features_file)
        with open(cached_train_features_file, "wb") as writer:
            pickle.dump(features, writer)



    # logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(examples))
    # print(torch.tensor([f.input_ids for f in features_seg], dtype=torch.long).size())
    # print(torch.tensor([f.input_ids for f in features_utt], dtype=torch.long).size())

    # utt_input_ids = torch.tensor([f.input_ids for f in features_utt], dtype=torch.long). \
    #     view(-1, args.max_utterance_num, args.max_seq_length)
    # utt_token_type_ids = torch.tensor([f.segment_ids for f in features_utt], dtype=torch.long). \
    #     view(-1, args.max_utterance_num, args.max_seq_length)
    # utt_attention_mask = torch.tensor([f.input_mask for f in features_utt], dtype=torch.long). \
    #     view(-1, args.max_utterance_num, args.max_seq_length)

    seg_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    seg_token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    seg_attention_mask = torch.tensor([f.input_mask  for f in features], dtype=torch.long)
    cls_sep_pos = torch.tensor([f.cls_sep_pos for f in features], dtype=torch.long)
    true_len = torch.tensor([f.true_len for f in features], dtype=torch.long)

    # res_input_ids = torch.tensor([f.input_ids for f in features_res_lab], dtype=torch.long)
    # res_token_type_ids = torch.tensor([f.segment_ids for f in features_res_lab], dtype=torch.long)
    # res_attention_mask = torch.tensor([f.input_mask for f in features_res_lab], dtype=torch.long)

    labels = torch.FloatTensor([f.label_id for f in features])
    # print(utt_input_ids[0]==utt_input_ids[1])
    # print(seg_input_ids[0] == seg_input_ids[1])
    # print(utt_input_ids.size(),utt_attention_mask.size(),utt_token_type_ids.size())
    # print(seg_input_ids.size(),seg_token_type_ids.size(),seg_attention_mask.size())
    # print(res_input_ids.size(),res_attention_mask.size(),res_token_type_ids.size())
    # print(labels.size())
    train_data = TensorDataset(seg_input_ids,seg_token_type_ids, seg_attention_mask ,
                                cls_sep_pos,true_len,labels)

    if(tag=="train"):
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader

def eval(model,tokenizer,device,myDataProcessorSeg):
    logger.info("start evaluation")
    # uttdatafile = os.path.join(args.data_dir, args.task_name, "test.txt")
    segdatafile = os.path.join(args.data_dir, args.task_name, "testseg.txt")
    examples= myDataProcessorSeg.get_test_examples(segdatafile)
    # examples_seg = myDataProcessorSeg.get_test_examples(segdatafile)
    # print("dev: len(examples_res_lab)", len(examples_res_lab))
    # print("dev: len(examples_utt)", len(examples_utt))
    # print("dev:len(examples_seg)", len(examples_seg))
    label_list = myDataProcessorSeg.get_labels()
    eval_dataloader = get_dataloader(tokenizer, examples,label_list, "valid")
    y_pred = []
    y_label=[]

    metrics = Metrics(args.temp_score_file_path)

    for batch in tqdm(eval_dataloader,desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        seg_input_ids, seg_token_type_ids, seg_attention_mask, \
        cls_sep_pos, true_len,labels = batch
        y_label+=labels.data.cpu().numpy().tolist()
        with torch.no_grad():
            logits= model(seg_input_ids, seg_token_type_ids, seg_attention_mask,cls_sep_pos, true_len,labels=None)
            y_pred += logits.data.cpu().numpy().tolist()

    with open(args.temp_score_file_path, 'w',encoding='utf-8') as output:
        for score, label in zip(y_pred, y_label):
            output.write(
                str(score) + '\t' +
                str(int(label)) + '\n'
            )
    result = metrics.evaluate_all_metrics()
    return result

def train(model,tokenizer,device,myDataProcessorSeg,n_gpu):
    # uttdatafile=os.path.join(args.data_dir,args.task_name,"train.txt")
    segdatafile = os.path.join(args.data_dir, args.task_name, "trainseg.txt")
    best_result = [0, 0, 0, 0, 0, 0]
    # examples_res_lab, examples_utt= myDataProcessorUtt.get_train_examples(uttdatafile)
    examples=myDataProcessorSeg.get_train_examples(segdatafile)
    # print("train: len(examples_res_lab)", len(examples_res_lab))
    # print("train: len(examples_utt)", len(examples_utt))
    # print("train:len(examples_seg)" ,len(examples_seg))
    # print("examples_utt[0]==examples_utt[10]",examples_utt[0].text_a==examples_utt[10].text_a)
    # print("examples_seg[0]==examples_seg[5]",examples_seg[0].text_a==examples_seg[5].text_a)
    num_train_optimization_steps = int(
        len(examples) / args.train_batch_size) * args.num_train_epochs
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    label_list = myDataProcessorSeg.get_labels()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    train_dataloader=get_dataloader(tokenizer,examples,label_list,"train")
    set_seed()
    model.train()
    global_step = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        # nb_tr_examples, nb_tr_steps = 0, 0
        s=0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            seg_input_ids, seg_token_type_ids, seg_attention_mask,cls_sep_pos, true_len,labels = batch

            # define a new function to compute loss values for both output_modes
            logits,loss = model(seg_input_ids,seg_token_type_ids, seg_attention_mask ,cls_sep_pos, true_len,labels)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            tr_loss += loss.item()
            s+=1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.info('Epoch{} Batch{} - loss: {:.6f}  batch_size:{}'.format(epoch,step, loss.item(), labels.size(0)) )
            global_step += 1

        logger.info("average loss(:.6f)".format(tr_loss/s))
        # Save a trained model, configuration and tokenizer
        model.eval()
        result=eval(model, tokenizer, device, myDataProcessorSeg)
        logger.info("Evaluation Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
                    result[0], result[1], result[2], result[3], result[4], result[5])
        if(result[3] + result[4] + result[5] > best_result[3] + best_result[4] + best_result[5]):
            logger.info("save model")
            model_to_save = model.module if hasattr(model, 'module') else model

            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)
            best_result=result

        logger.info("best result")
        logger.info("Best Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
                     best_result[0], best_result[1], best_result[2],
                     best_result[3],best_result[4],best_result[5] )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    set_seed()
    # args.output_dir=os.path.join(args.data_dir,args.task_name,args.output_dir)
    # args.temp_score_file_path=os.path.join(args.data_dir,args.task_name,args.temp_score_file_path)
    # args.input_cache_dir=os.path.join(args.data_dir, args.task_name, args.input_cache_dir)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # if not os.path.exists(args.input_cache_dir):
    #     os.makedirs(args.input_cache_dir)
    # myDataProcessorUtt = MyDataProcessorUtt(args.max_utterance_num)
    myDataProcessorSeg=MyDataProcessorSegres()
    # label_list = myDataProcessorUtt.get_labels()
    # num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.bert_model)
    if args.do_train:
        logger.info("start train...")
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # if(os.path.exists(output_model_file)):
        #     logger.info("load dict...")
        #     model_state_dict = torch.load(output_model_file)
        #     model = BertForSequenceClassificationTS.from_pretrained(args.bert_model, config=config,
        #                                                             state_dict=model_state_dict, num_labels=num_labels)
        # else:
        model = BertForSequenceClassificationTSv3.from_pretrained(args.bert_model,
                                                                    config=config,
                                                                    max_seg_num=args.max_segment_num,
                                                                    max_seq_len=args.max_seq_length,
                                                                    device=device)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        train(model,tokenizer,device,myDataProcessorSeg,n_gpu)
    else:
        logger.info("start test...")
        logger.info("load dict...")
        output_model_file = os.path.join(args.output_dir,  WEIGHTS_NAME)
        model_state_dict = torch.load(output_model_file)
        model = BertForSequenceClassificationTSv3.from_pretrained(args.bert_model, config=config,
                                                                  state_dict=model_state_dict,
                                                                  max_seg_num=args.max_segment_num,
                                                                  max_seq_len=args.max_seq_length,
                                                                  device=device)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # similar_score(model, tokenizer, device,myDataProcessorSeg)
        result = eval(model, tokenizer, device, myDataProcessorSeg)
        logger.info("Evaluation Result: \nMAP: %f\tMRR: %f\tP@1: %f\tR1: %f\tR2: %f\tR5: %f",
                    result[0], result[1], result[2], result[3], result[4], result[5])
        print(result)

if __name__ == "__main__":
    main()
