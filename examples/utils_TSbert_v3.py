import logging
import numpy as np
logger = logging.getLogger(__name__)
from tqdm import tqdm

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir_seg):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir_seg):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self,data_dir_seg):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
#
#     # @classmethod
#     # def _read_tsv(cls, input_file, quotechar=None):
#     #     """Reads a tab separated value file."""
#     #     with open(input_file, "r", encoding="utf-8-sig") as f:
#     #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#     #         lines = []
#     #         for line in reader:
#     #             if sys.version_info[0] == 2:
#     #                 line = list(unicode(cell, 'utf-8') for cell in line)
#     #             lines.append(line)
#     #         return lines
#
# class MyDataProcessorUtt(DataProcessor):
#     def __init__(self,max_utts=10):
#         super(DataProcessor, self).__init__()
#         self.max_utts=max_utts
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(data_dir,"train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(data_dir, "dev")
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(data_dir, "test")
#
#     def _create_examples(self, data_dir, set_type):
#         """Creates examples for the training and dev sets."""
#         examples_res_lab = []
#         examples_utt = []
#         i=0
#         with open(data_dir,'r',encoding='utf-8') as rf:
#             for line in rf:
#                 line=line.strip()
#                 if(line):
#                     guid = "%s-%s" % (set_type, i)
#                     label=line.split("\t")[0]
#                     res=line.split("\t")[-1]
#                     examples_res_lab.append(
#                         InputExample(guid=guid, text_a=res, label=label))
#
#                     con_list=line.split("\t")[1:-1][-self.max_utts:]
#                     con_list_pad=[None for _ in range(self.max_utts-len(con_list))]
#                     con_list_pad.extend(con_list)
#                     for utt in con_list_pad:
#                         examples_utt.append(InputExample(guid=guid, text_a=utt))
#                     i+=1
#         return examples_res_lab,examples_utt
#
#
#     def get_labels(self):
#         """See base class."""
#         return ["0","1"]
#
# class MyDataProcessorSeg(DataProcessor):
#     def __init__(self,max_segs,train_intv=2,dev_intv=10,test_intv=10):
#         super(DataProcessor, self).__init__()
#         self.max_segs=max_segs
#         self.train_intv=train_intv
#         self.dev_intv=dev_intv
#         self.test_intv=test_intv
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(data_dir,"train",self.train_intv)
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(data_dir, "dev",self.dev_intv)
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(data_dir, "test",self.test_intv)
#
#     def get_labels(self):
#         """See base class."""
#         return ["0","1"]
#
#     def _create_examples(self, data_dir, set_type,inteval):
#         """Creates examples for the training and dev sets."""
#         examples_seg = []
#         i=0
#         with open(data_dir,'r',encoding='utf-8') as rf:
#             for line in rf:
#                 line=line.strip()
#                 if(line):
#                     guid = "%s-%s" % (set_type, i)
#                     seg_list=line.split("\t")[-self.max_segs:]
#                     seg_list_pad=[None for _ in range(self.max_segs-len(seg_list))]
#                     seg_list_pad.extend(seg_list)
#                     for _ in range(inteval):
#                         for utt in seg_list_pad:
#                             examples_seg.append(InputExample(guid=guid, text_a=utt))
#                     i+=1
#         return examples_seg


class MyDataProcessorSegres(DataProcessor):
    def __init__(self):
        super(DataProcessor, self).__init__()
        # self.train_intv=train_intv
        # self.dev_intv=dev_intv
        # self.test_intv=test_intv


    def get_train_examples(self, data_dir_seg):
        """See base class."""
        return self._create_examples(data_dir_seg,"train")

    def get_dev_examples(self, data_dir_seg):
        """See base class."""
        return self._create_examples(data_dir_seg,"dev")

    def get_test_examples(self ,data_dir_seg):
        """See base class."""
        return self._create_examples(data_dir_seg, "test")

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, data_dir_seg, set_type):
        """Creates examples for the training and dev sets."""
        examples_seg = []
        i=0
        with open(data_dir_seg,'r',encoding='utf-8') as rf:
            for line in rf:
                line=line.strip()
                if(line):
                    guid = "%s-%s" % (set_type, i)
                    labsegres_list=line.split("\t")
                    examples_seg.append(InputExample(guid=guid,segment=labsegres_list[1:-1],
                                                     response=labsegres_list[-1],label=labsegres_list[0]))
                    i+=1
        return examples_seg


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, segment=None, response=None,label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.segment=segment
        self.response=response
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,cls_sep_pos,true_len,label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_sep_pos = cls_sep_pos
        self.true_len = true_len
        self.label_id = label_id


def convert_examples_to_features_Segres(examples, label_list,max_seg_num,max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    pbar=tqdm(total=len(examples),desc="convert_examples_to_features")
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = []
        for seg in example.segment[-max_seg_num:]: ##todo
            tokens_a += tokenizer.tokenize(seg) + [sep_token]
        tokens_a = tokens_a[:-1]  # 去掉最后一个[SEP] s1[SEP]s2[SEP]s3[SEP]S4

        tokens_b = tokenizer.tokenize(example.response)
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        if (tokens_a[0] == sep_token):
            tokens_a = tokens_a[1:]
        # tokens_b = None
        # if example.text_b:
        #     tokens_b = tokenizer.tokenize(example.text_b)
        #     # Modifies `tokens_a` and `tokens_b` in place so that the total
        #     # length is less than the specified length.
        #     # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        #     special_tokens_count = 4 if sep_token_extra else 3
        #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        # else:
        #     # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        #     special_tokens_count = 3 if sep_token_extra else 2
        #     if len(tokens_a) > max_seq_length - special_tokens_count:
        #         tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids #[CLS]s1[SEP]s2[SEP]s3[SEP]S4[SEP]res[SEP]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # if(not example.text_a and not example.text_b):
        #     input_ids=[]
        #     segment_ids=[]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        cls_sep_pos = [0]

        for tok_ind, tok in enumerate(tokens):
            if (tok == sep_token):
                cls_sep_pos.append(tok_ind)
        true_len = len(cls_sep_pos)
        while (len(cls_sep_pos) < max_seg_num + 2):
            cls_sep_pos.append(-1)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        # print("len(input_ids)",input_ids,len(input_ids))
        # print("len(input_mask)",input_mask,len(input_mask))
        # print("len(segment_ids)",segment_ids,len(segment_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if output_mode == "classification":
        # print(label_map)
        if(example.label):
            label_id = label_map[example.label]
        else:
            label_id=None
        # elif output_mode == "regression":
        #     label_id = float(example.label)
        # else:
        #     raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              cls_sep_pos=cls_sep_pos,
                              true_len=true_len,
                              label_id=label_id))
        pbar.update(1)
    pbar.close()
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()

class Metrics(object):

    def __init__(self, score_file_path:str):
        super(Metrics, self).__init__()
        self.score_file_path = score_file_path
        self.segment = 10

    def __read_socre_file(self, score_file_path):
        sessions = []
        one_sess = []
        with open(score_file_path, 'r',encoding='utf-8') as infile:
            i = 0
            for line in infile.readlines():
                i += 1
                tokens = line.strip().split('\t')
                one_sess.append((float(tokens[0]), int(tokens[1])))
                if i % self.segment == 0:
                    one_sess_tmp = np.array(one_sess)
                    if one_sess_tmp[:, 1].sum() > 0:
                        sessions.append(one_sess)
                    one_sess = []
        return sessions


    def __mean_average_precision(self, sort_data):
        #to do
        count_1 = 0
        sum_precision = 0
        for index in range(len(sort_data)):
            if sort_data[index][1] == 1:
                count_1 += 1
                sum_precision += 1.0 * count_1 / (index+1)
        return sum_precision / count_1


    def __mean_reciprocal_rank(self, sort_data):
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))

    def __precision_at_position_1(self, sort_data):
        if sort_data[0][1] == 1:
            return 1
        else:
            return 0

    def __recall_at_position_k_in_10(self, sort_data, k):
        sort_label = [s_d[1] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)


    def evaluation_one_session(self, data):
        '''
        :param data: one conversion session, which layout is [(score1, label1), (score2, label2), ..., (score10, label10)].
        :return: all kinds of metrics used in paper.
        '''
        np.random.shuffle(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        m_a_p = self.__mean_average_precision(sort_data)
        m_r_r = self.__mean_reciprocal_rank(sort_data)
        p_1   = self.__precision_at_position_1(sort_data)
        r_1   = self.__recall_at_position_k_in_10(sort_data, 1)
        r_2   = self.__recall_at_position_k_in_10(sort_data, 2)
        r_5   = self.__recall_at_position_k_in_10(sort_data, 5)
        return m_a_p, m_r_r, p_1, r_1, r_2, r_5


    def evaluate_all_metrics(self):
        sum_m_a_p = 0
        sum_m_r_r = 0
        sum_p_1 = 0
        sum_r_1 = 0
        sum_r_2 = 0
        sum_r_5 = 0

        sessions = self.__read_socre_file(self.score_file_path)
        total_s = len(sessions)
        for session in sessions:
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = self.evaluation_one_session(session)
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5

        return (sum_m_a_p/total_s,
                sum_m_r_r/total_s,
                  sum_p_1/total_s,
                  sum_r_1/total_s,
                  sum_r_2/total_s,
                  sum_r_5/total_s)