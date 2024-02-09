# Understanding the Deception: Enhancing Threat Intelligence through Crowdsourced Analysis of Fake Cybersecurity Information
Code used for [Understanding the Deception: Enhancing Threat Intelligence through Crowdsourced Analysis of Fake Cybersecurity Information](http://).



## Updates
- Feb-9-2024: first released


## 1. Set-up
### environment requirements:
python = 3.7
```
pip install -r requirements.txt
```

## 2. Generate Dataset
prepare data folder, and choose from [opt 1 - use our dataset] or [opt 2 - generate your own dataset]
```
mkdir data
```

### Opt 1 - use our dataset: run following commands and jump to Step 3.
```
cp ./dataset/CTI_long.xlsx ./data/dataset_long.xlsx
```

### Opt 2 - generate your own dataset
download corpus dataset
```
cd data
git clone https://github.com/UMBC-Onramp/CyEnts-Cyber-Blog-Dataset.git
git clone https://github.com/Ebiquity/CASIE.git
```

generate short cti sample
```
cd ..
python generate_corpus.py --input CyEnts-Cyber-Blog-Dataset/Sentences/ --output UMBC_finetune.txt
```

prepare model folder
```
mkdir model
cd model
mkdir gpt2finetune
cd gpt2finetune
git clone https://github.com/nshepperd/gpt-2
```

Grant Colab read and execute access to the cloned folder.
```
chmod 755 -R ./gpt-2
```

Download the required GPT-2 model from the available four options, 124M, 355M, 774M, 1558M.
We use 355M
```
cd gpt-2
python download_model.py 355M
```

set python IO encoding to UTF-8
```
export PYTHONIOENCODING=UTF-8
```

Finetune gpt-2 model using CASIE_finetune.txt
```
PYTHONPATH=src ./train.py --dataset ../../../data/CASIE_finetune.txt --model_name 355M --batch_size 1 --memory_saving_gradients 2>&1 | tee casie_log1.txt
```

Save finetune model to 355M-v1
```
cd models
mkdir 355M-v1
cd ..
cp -r ./checkpoint/run1/*  ./models/355M-v1/
```

Copy generate_dataset_from_corpus.py and run
```
cp ../../../generate_dataset_from_corpus.py ./src/generate_dataset_from_corpus.py
python src/generate_dataset_from_corpus.py --top_k 40 --model_name 355M-v1 --input_dataset ../../../data/CASIE_finetune.txt --output_file ../../../data/dataset_long.xlsx
cd ../../..
```

