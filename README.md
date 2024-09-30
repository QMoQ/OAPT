# OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal 
[Qiao Mo](), [Yukang Ding](), [Jinhua Hao](), [Qiang Zhu](), [Ming Sun](), [Chao Zhou](), [Feiyu Chen](), [Shuyuan Zhu]()

UESTC, Kuaishou Techonology

>Official implement of OAPT in ECCV2024, which is a transformer-based network deigned for double (or multiple) compressed image restoration.

[Paper Link](https://arxiv.org/abs/2408.11480)

---



# TODO List
- [ ] Submit to ARXIV
- [ ] Release main codes of models
- [ ] Release test codes
- [ ] Release training codes




---

### Architecture
![architecture](https://github.com/QMoQ/OAPT/blob/main/pics/pipeline.png)

### Pattern clustering & inv operation 
![pattern clustering](https://github.com/QMoQ/OAPT/blob/main/pics/patternclustering.png)

### Experimental results on gray double JPEG images
![results](https://github.com/QMoQ/OAPT/blob/main/pics/gray_results.png)

### Visual results
![gray visual results](https://github.com/QMoQ/OAPT/blob/main/pics/visuals.png)


### Training details

| Model(Gray) | Params(M) | Multi-Adds(G) | TrainingSets | Pretrain model | iterations |
|--------|:---------:|:---------:|:---------:|:---------:|:---------:|
| [SwinIR](https://github.com/JingyunLiang/SwinIR) |   11.49    | 293.42 | DF2K | 006_CAR_DFWB_s126w7_SwinIR-M_jpeg10 | 200k |
| [HAT-S](https://github.com/XPixelGroup/HAT) |   9.24    | 227.14 | DF2K | HAT-S_SRx2 | 800k |
| [ART](https://github.com/gladzhang/ART) |   16.14    | 415.51 | DF2K | CAR_ART_q10 | 200k |
| [OAPT](https://arxiv.org/abs/2408.11480) |   12.96    | 293.60 | DF2K | 006_CAR_DFWB_s126w7_SwinIR-M_jpeg10 | 200k |

### Setup
This project is mainly based on [swinir](https://github.com/JingyunLiang/SwinIR) and [hat](https://github.com/XPixelGroup/HAT). 

The version of PyTorch we used is 1.7.0.
'''
pip install -r requirements.txt
python setup.py develop
'''

### Test
'''
CUDA_VISIBLE_DEVICES=0 python oapt/test.py -opt ./options/Gray/test/test_oapt.yml
'''


