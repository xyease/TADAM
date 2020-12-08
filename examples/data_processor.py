inputfile_utt="data/alime/train.txt"
inputfile_seg="data/alime/train_seg.txt"
outputfile="data/alime/trainseg.txt"
interval=2
def write_segfile(inputfile_utt,inputfile_seg,outputfile,interval):
    lab_res_list=[]
    with open(inputfile_utt,'r',encoding='utf-8') as rf:
        for line in rf:
            line=line.strip()
            if(line):
                lab=line.split('\t')[0]
                res=line.split('\t')[-1]
                lab_res_list.append([lab,None,res])
    print(len(lab_res_list))
    seg_list=[]
    with open(inputfile_seg,'r',encoding='utf-8') as rf:
        for line in rf:
            line=line.strip()
            if(line):
                seg_list.append(line)
    print(len(seg_list))
    i=0
    for seg in seg_list:
        for _ in range(interval):
            lab_res_list[i][1]=seg
            i+=1
    print(i)

    with open(outputfile,'w',encoding='utf-8') as wf:
        for lab_seg_res in lab_res_list:
            wf.write('\t'.join(lab_seg_res)+'\n')

write_segfile(inputfile_utt,inputfile_seg,outputfile,interval)
# i=0
# with open(outputfile,'r') as rf:
#     for line in rf:
#         i+=1
# print(i)

