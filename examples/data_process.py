import json
input_file="data/alime/train.txt"
output_file_seg="data/alime/train_seg.txt"
cut_list_file="data/alime/cutlist_train.json"
interval=2

def generate_output_file_seg():
    contexts=[]
    num=0
    with open(input_file,'r',encoding='utf-8') as rf:
        for line in rf:
            line=line.strip()
            if(line and num%interval==0):
                    contexts.append(line.split("\t")[1:-1])
            num+=1
    print(num)
    with open(cut_list_file,'r',encoding='utf-8') as rf:
        cutlist=json.loads(rf.read())

    print(len(contexts))
    print(len(cutlist))

    with open(output_file_seg,'w',encoding='utf-8') as wf:
        for context,cutl in zip(contexts,cutlist):
            seg_list=[]
            seg=""
            index_1 = 0
            index_2 = 0
            for index, utt in enumerate(context):
                if(seg):
                    seg=seg+" "+utt
                else:
                    seg=utt
                if (index_1 == cutl[index_2]):
                    index_2 += 1
                    seg_list.append(seg)
                    seg=""
                index_1 += 1
            # a="\t".join(seg_list)+'\n'
            wf.write("\t".join(seg_list)+'\n')


generate_output_file_seg()
# with open("data/alime/test_seg.txt",'r',encoding='utf-8') as rf:
#     for line in rf:
#         a=line