## Table of Contents
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Pretrained Models and Logs](#Pretrained-Models-and-Logs)
- [Data Preparation](#Data-Preparation)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Citations](#Citations)
- [License](#License)
- [Acknowledgements](#Acknowledgements)
- [Contact](#Contact)

## Introduction
This repository provides the code for the paper ["FakeMix Augmentation Improves Transparent Object Detection"](https://arxiv.org/pdf/2103.13279.pdf).

## Requirements
- python3
- PyTorch=1.1.0
- torchvision
- Pillow
- numpy
- pyyaml

## Installation

Please make sure that there is at least one gpu when compiling. Then run:

- `python3 setup.py develop`

## Pretrained Models and Logs
The pretrained models and logs can be fould here:

- [Google Drive](https://drive.google.com/drive/folders/1XNdDKfC9oBEeoOOWL4xe7xFS-CLlFrNH?usp=sharing)

- [Baidu Drive](https://pan.baidu.com/s/1A-5ZWc8RiihYXuCFEdTHfQ) with code: j3qp

## Data Preparation
1. create dirs './datasets/Trans10K'
2. download the data from [Trans10K](https://xieenze.github.io/projects/TransLAB/TransLAB.html).
3. put the train/validation/test data under './datasets/Trans10K'.
The data Structure is shown below:

```
Trans10K/
├── test
│   ├── easy
│   └── hard
├── train
│   ├── images
│   └── masks
└── validation
    ├── easy
    └── hard
```

## Demo
```
CUDA_VISIBLE_DEVICES=0 python3 -u ./tools/test_demo.py --config-file configs/trans10K/trans10K.yaml DEMO_DIR ./demo/imgs
```

## Training
```
bash tools/dist_train.sh configs/trans10K/trans10K.yaml 8
```

## Evaluation
```
CUDA_VISIBLE_DEVICES=0 python3 -u ./tools/test_translab.py --config-file configs/trans10K/trans10K.yaml 
```

## Citations
Please cite our paper if the project helps.
```
@article{cao2021fakemix,
  title={FakeMix Augmentation Improves Transparent Object Detection},
    author={Cao, Yang and Zhang, Zhengqiang and Xie, Enze and Hou, Qibin and Zhao, Kai and Luo, Xiangui and Tuo, Jian},
      journal={arXiv preprint arXiv:2103.13279},
        year={2021}
        }

@misc{fanet,
    author = {Cao, Yang and Zhang, Zhengqiang},
    title = {fanet},
    howpublished = {\url{https://github.com/yangcao1996/fanet}},
    year ={2021}
}
        
```
## License
For academic use, this project is licensed under [the Apache 2.0 License](https://github.com/yangcao1996/fanet/blob/main/LICENSE)

For commercial use, please contact the authors.

## Acknowledgements
Our codes are mainly based on [TransLab](https://github.com/xieenze/Segment_Transparent_Objects). Thanks to their wonderful works.

## Contact
Any discussion is welcome. Please contact the authors:

Yang Cao:         yangcao.cs@gmail.com

Zhengqiang Zhang: zhengqiang.zhang@hotmail.com
