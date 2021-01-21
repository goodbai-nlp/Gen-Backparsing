# AMR-Backparsing
An implementation for paper "Online Back-Parsing for AMR-To-Text Generation" (in EMNLP 2020)

# Requirements
+ python 3.6
+ pytorch 1.0

# Data Preprocessing
We follow [this work](https://github.com/Amazing-J/structural-transformer) to preprocess AMR. Since AMR corpus has LDC license, we upload some examples for format reference. If you have the license, feel free to contact us for getting the preprocessed data.

# Runing
```
bash ./src/train-LDC2015.sh
bash ./src/train-LDC2017.sh
```

#Evaluation
```
bash ./src/translate-LDC15.sh
bash ./src/translate-LDC17.sh
```
