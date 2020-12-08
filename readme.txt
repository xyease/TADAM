our own data for segmentation is in TADAM/examples/data direstory

environment: python 3.5 or 3.6  

package: 
torch>=1.0.0, tqdm, boto3, requests, regex, sacremoses, openpyxl, numpy, sentencepiece


topic segmentation code for three datasets:
segmentation_BERTCLS_Douban.py 
segmentation_BERTCLS_Ubuntu.py
segmentation_BERTCLS_Ecom.py


The main code for TADAM is in examples/run_TSBERT_v3.py
we use pre-trained BERT of pytorch version from https://github.com/huggingface/transformers

metrics for response selection is in utils_TSbert_v3.py

