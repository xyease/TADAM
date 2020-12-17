## Dataset
Please download datasets to the corresponding directory under "data"

E-commerce
https://drive.google.com/file/d/154J-neBo20ABtSmJDvm7DK0eTuieAuvw/view?usp=sharing.

Ubuntu
https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntudata.zip?dl=0

Douban
https://www.dropbox.com/s/90t0qtji9ow20ca/DoubanConversaionCorpus.zip?dl=0&file_subpath=%2FDoubanConversaionCorpus

Our own dataset for segmentation is under DATASET directory

## Source Code
* prepare data

    generate cutlist.txt 
    
    <code>python segmentation_BERTCLS.py --datapath=data/xxx/xxx.txt</code>
    
    gather segmented data: data/xxx/xxxseg.txt:
    
    set interval = 2 for train.txt, interval = 10 for test.txt
    set corresponding datafile and dataset in data_process.py
    
    <code>python data_process.py</code>  
    
* train

    <code>python run_TSbert_v3.py  --task=alime  --do_train  --train_batch_size=20  --learning_rate=2e-5</code> 
    
    The data will be saved in data/alime/input_cache_v3 
    
    model will be saved in data/alime/model_save_v3, training log will also be saved in log.txt
    

* eval

    <code>python run_TSbert_v3.py --task=xxx</code> 
    
    You can also load our trained model for testing https://jbox.sjtu.edu.cn/l/5odMwy
   
### Environment:
we use pre-trained BERT of pytorch version from https://github.com/huggingface/transformers

torch>=1.0.0

package: tqdm, boto3, requests, regex, sacremoses, openpyxl, numpy, sentencepiece

### Reference
 
If you use this code please cite our paper:
```
@article{xu2020topic,
  title={Topic-aware multi-turn dialogue modeling},
  author={Xu, Yi and Zhao, Hai and Zhang, Zhuosheng},
  journal={arXiv preprint arXiv:2009.12539},
  year={2020}
}
```
