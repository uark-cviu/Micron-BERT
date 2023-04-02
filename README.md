# Micron-BERT: BERT-based Facial Micro-Expression Recognition

This repo is an official implementations of Micron-BERT

## Abstract
Micro-expression recognition is one of the most challenging topics in affective computing. It aims to recognize tiny facial movements difficult for humans to perceive in a brief period, i.e., 0.25 to 0.5 seconds. Recent advances in pre-training deep Bidirectional Transformers (BERT) have significantly improved self-supervised learning tasks in computer vision. However, the standard BERT in vision problems is designed to learn only from full images or videos, and the architecture cannot accurately detect details of facial micro-expressions. This paper presents Micron-BERT, a novel approach to facial micro-expression recognition. The proposed method can automatically capture these movements in an unsupervised manner based on two key ideas. First, we employ Diagonal Micro-Attention (DMA) to detect tiny differences between two frames. Second, we introduce a new Patch of Interest (PoI) module to localize and highlight micro-expression interest regions and simultaneously reduce noisy backgrounds and distractions. By incorporating these components into an end-to-end deep network, the proposed Micron-BERT significantly outperforms all previous work in various micro-expression tasks. Micron-BERT can be trained on a large-scale unlabeled dataset, i.e., up to 8 million images, and achieves high accuracy on new unseen facial micro-expression datasets. Empirical experiments show Micron-BERT consistently outperforms state-of-the-art performance on four micro-expression benchmarks, including SAMM, CASME II, SMIC, and CASME3, by significant margins. Code will be available at \url{https://github.com/uark-cviu/Micron-BERT}

## Training
The full source code will be released soon.

## Testing
We provide the pretrained Micron-BERT model [here](https://uark-my.sharepoint.com/:u:/g/personal/xnguyen_uark_edu/EQHhC8oKiUhElDZ3dfXueeABSefj3MetYv8YOoDljX_Hbg?e=6nwCxv). To finetune the Micron-BERT on your micro-expression database, please refer to [micron_bert.py](micron_bert.py)

## Acknowledgement

## Citation
If you find this repository useful, please consider giving a star :star: and citation
```
@INPROCEEDINGS{nguyen2023micronbert,
  author={Nguyen, Xuan-Bac and Duong, Chi Nhan and Xin, Li and Susan, Gauch and Han-Seok, Seo and Luu, Khoa},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Micron-BERT: BERT-based Facial Micro-Expression Recognition}, 
  year={2023},
}
```
