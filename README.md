# DEPN for BERT
DEPN is a privacy-preserving framework in the post-processing stage of language models, which erases privacy data by detecting privacy neurons and editing them. Paper Link: https://arxiv.org/pdf/2310.20138.pdf.

## Abstract
Pretrained language models have learned a vast amount of human knowledge from large-scale corpora, but their powerful memorization capability also brings the risk of data leakage. Some risks may only be discovered after the model training is completed, such as the model memorizing a specific phone number and frequently outputting it. In such cases, model developers need to eliminate specific data influences from the model to mitigate legal and ethical penalties. To effectively mitigate these risks, people often have to spend a significant amount of time and computational costs to retrain new models instead of finding ways to cure the 'sick' models. Therefore, we propose a method to locate and erase risky neurons in order to eliminate the impact of privacy data in the model. We use a new method based on integrated gradients to locate neurons associated with privacy texts, and then erase these neurons by setting their activation values to zero.Furthermore, we propose a risky neuron aggregation method to eliminate the influence of privacy data in the model in batches. Experimental results show that our method can effectively and quickly eliminate the impact of privacy data without affecting the model's performance. Additionally, we demonstrate the relationship between model memorization and neurons through experiments, further illustrating the robustness of our method.

## Overview
![image](https://github.com/flamewei123/DEPN/blob/main/overview.png)

## How to use

**1.processing data and model**

Please prepare experimental data according to README in the /data folder. 

(1) generating Enron data

(2) fine-tuning the private BERT model

(3) finding the private data memorized by the model

(4) sampling the data into the specified format

**2. Detect and edit privacy neurons**

Please earse the privacy from the BERT model according to README in the /src folder.

(1) detecting privacy neurons

(2) aggregating privacy neuron

(3) editing privacy neurons

## Citation

If you use this code for your research, please kindly cite our EMNLP-2023 paper:

```
@inproceedings{wu2023depn,
  title={DEPN: Detecting and Editing Privacy Neurons in Pretrained Language Models},
  author={Wu, Xinwei and Li, Junzhuo and Xu, Minghui and Dong, Weilong and Wu, Shuangzhi and Bian, Chao and Xiong, Deyi},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={2875--2886},
  year={2023}
}
```

## Future work

Our initial work was primarily focused on the encoder architecture, which is why we experimented with the largest language model being BERT-large. In our subsequent research, we shifted our focus to the decoder architecture and achieved significant protective effects on the llama2-7B model. As our latest work was submitted to ARR in February 2024 under anonymous review, the new paper and code will be disclosed after the review results are published. Once the review results are available, we will update the repository with the relevant links.

## Contact

XInwei Wu: wuxw2021@tju.edu.cn
