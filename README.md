# AMR-Backparsing
An implementation for paper "Online Back-Parsing for AMR-To-Text Generation" (in EMNLP 2020)

# Requirements
+ python 3.6
+ pytorch 1.0

# Data Preprocessing
We follow [this work](https://github.com/Amazing-J/structural-transformer) to preprocess AMR. Since AMR corpus require LDC license, we upload some examples for format reference. If you have the license, feel free to contact us for getting the preprocessed data.

# Runing
```
bash ./src/train-LDC2015.sh
bash ./src/train-LDC2017.sh
```

# Evaluation
```
bash ./src/translate-LDC15.sh
bash ./src/translate-LDC17.sh
```

# Results

|Setting|  BLEU-tok  | BLEU-nltk  | Meteor | chrF++ |
|  :----:  | :----:  |:---:|  :----:  | :----:  |
| LDC15  | 31.58 | 32.27 | 36.38 | 65.33 |
| LDC17  | 34.36 | 34.98 | 38.09 | 67.90 |

# References
```
@inproceedings{bai-etal-2020-online,
    title = "Online Back-Parsing for {AMR}-to-Text Generation",
    author = "Bai, Xuefeng  and
      Song, Linfeng  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.92",
    doi = "10.18653/v1/2020.emnlp-main.92",
    pages = "1206--1219",
}
```
