# <span style="font-variant:small-caps;">VAT_D</span>

This repository contains the data and code for our paper:
*Jungsoo Park\*, Gyuwan Kim\*, Jaewoo Kang. ["Consistency Training with Virtual Adversarial Discrete Perturbation"](https://arxiv.org/pdf/2104.07284.pdf).* (NAACL 2022)

## Requirements

```bash
$ conda create -n vat_d python=3.6
$ conda activate vat_d
$ pip install transformers==3.1.0
$ pip install pandas
```
Note that Pytorch has to be installed depending on the version of CUDA (>= 1.4.0)

## Datasets

For training models on the classification tasks, you will need the dataset.
You can download the datasets by running the code below. This will download and unzip files into the ./data directory.
For pre-processing the data, we referenced the paper [MixText](https://arxiv.org/pdf/2004.12239.pdf).

```bash
$ bash scripts/download_data.sh
```

## Train

The following example fine-tunes the BERT-base model on the AG_NEWS dataset with our proposed method VAT-D (5 different runs with different seeds).
The model that we report in the paper was trained with a P40 GPU model.

- If you want to train with a different dataset, you can change the script from `train_agnews` to `train_yahoo` or `train_dbpedia`.
- If you want to adjust the number of labeled training samples, you can change the number in the `N_LABELED` field.
- For more details on the training arguments, you can reference arguments in `train.py`.

```bash
for seed in 0 1 2 3 4; do
    make train_agnews SEED=seed LR=3e-5 N_LABELED=10
done
```

## Results

For every run, you will obtain the best validation performance as well as the evaluation results on the test set below.

```bash
INFO:root: | Best Performance at Train Step : 1250
INFO:root: | Best Validation Accuracy : 88.13750457763672
INFO:root: | Best Test Accuracy : 88.11842346191406
```

## Citation

If you find the <span style="font-variant:small-caps;">VAT_D</span> method useful, please cite our paper:

```bibtex
@inproceedings{park2021consistency,
  title={Consistency training with virtual adversarial discrete perturbation},
  author={Park, Jungsoo and Kim, Gyuwan and Kang, Jaewoo},
  booktitle={ NAACL },
  year={2022}
}
```

## License
```
VAT_D
Copyright (c) 2022-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Contact

Please contact Jungsoo Park `jungsoopark.1993@gmail.com` and Gyuwan Kim `gyuwankim@ucsb.edu, `, or leave Github issues for any questions.
