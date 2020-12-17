import json
input_file = "data/alime/train.txt"
output_file_seg = "data/alime/train_seg.txt"
cut_list_file = "data/alime/cutlist_train.json"

final_outputfile="data/alime/trainseg.txt"
interval = 2

def generate_output_file_seg():
    contexts=[]
    num=0
    with open(input_file,'r',encoding='utf-8') as rf:
        for line in rf:
            line=line.strip()
            if(line and num%interval==0):
                    contexts.append(line.split("\t")[1:-1])
            num += 1
    print(num)
    with open(cut_list_file, 'r', encoding='utf-8') as rf:
        cutlist = json.loads(rf.read())

    print(len(contexts))
    print(len(cutlist))

    with open(output_file_seg,'w',encoding='utf-8') as wf:
        for context, cutl in zip(contexts, cutlist):
            seg_list = []
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


def write_segfile(inputfile_utt, inputfile_seg, outputfile, interval):
    lab_res_list=[]
    with open(inputfile_utt, 'r', encoding='utf-8') as rf:
        for line in rf:
            line=line.strip()
            if(line):
                lab=line.split('\t')[0]
                res=line.split('\t')[-1]
                lab_res_list.append([lab,None,res])
    print(len(lab_res_list))
    seg_list = []
    with open(inputfile_seg, 'r', encoding='utf-8') as rf:
        for line in rf:
            line = line.strip()
            if(line):
                seg_list.append(line)
    print(len(seg_list))
    i = 0
    for seg in seg_list:
        for _ in range(interval):
            lab_res_list[i][1] = seg
            i += 1
    print(i)

    with open(outputfile,'w',encoding='utf-8') as wf:
        for lab_seg_res in lab_res_list:
            wf.write('\t'.join(lab_seg_res)+'\n')


generate_output_file_seg()
write_segfile(input_file, output_file_seg, final_outputfile, interval)