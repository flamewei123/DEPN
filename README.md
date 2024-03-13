# DEPN for BERT
DEPN is a privacy-preserving framework in the post-processing stage of language models, which erases privacy data by detecting privacy neurons and editing them. Paper Link: https://arxiv.org/pdf/2310.20138.pdf.

## Abstract
Pretrained language models have learned a vast amount of human knowledge from large-scale corpora, but their powerful memorization capability also brings the risk of data leakage. Some risks may only be discovered after the model training is completed, such as the model memorizing a specific phone number and frequently outputting it. In such cases, model developers need to eliminate specific data influences from the model to mitigate legal and ethical penalties. To effectively mitigate these risks, people often have to spend a significant amount of time and computational costs to retrain new models instead of finding ways to cure the 'sick' models. Therefore, we propose a method to locate and erase risky neurons in order to eliminate the impact of privacy data in the model. We use a new method based on integrated gradients to locate neurons associated with privacy texts, and then erase these neurons by setting their activation values to zero.Furthermore, we propose a risky neuron aggregation method to eliminate the influence of privacy data in the model in batches. Experimental results show that our method can effectively and quickly eliminate the impact of privacy data without affecting the model's performance. Additionally, we demonstrate the relationship between model memorization and neurons through experiments, further illustrating the robustness of our method.

## Overview
![image](https://github.com/flamewei123/DEPN/blob/main/overview.png)

## How to use

**1.processing data and model**

Please prepare experimental data according to README in the `/data`. 

(1) generating Enron data
```
python preprocess_enron.py
```
(2) fine-tuning the private BERT model
```
bash run_bert.sh
```
(3) finding the private data memorized by the model
```
bash run_get_memorization.sh
```
(4) sampling the data and converting into the specified format
```
python sample_target_privacy.py
python mask_text2json.py
```
**2. Detecting and Editing Privacy Neurons**

Please earse the privacy from the BERT model according to README in the `/src`.

(1) detecting privacy neurons
```
bash 1_run_detector.sh
```
(2) aggregating privacy neuron
```
bash 2_run_aggregator.sh
```
(3) editing privacy neurons
```
bash 3_run_editor.sh
```
## Hyperparameter

There are several key hyperparameters that determine the performance of privacy protection experiments.

(1). `threshold` in `/data`

The threshold determines the scope of privacy to be protected. We recommend adjusting the threshold to maintain the final number of memorized data between 100 and 10,000.

(2). `sampled` in `/data`

Theoretically, the larger the number of samples, the better the effect of the neuron detector, but the time overhead will increase significantly. In addition, too little sampled privacy data will lead to a lack of representativeness of the found privacy neurons, thus affecting the results. We recommend sampling values between 20-200.

(3). two ratio values of aggregator in `/src`

Please see the code comments for the specific explanation. Our experience is that the best result is when the product of two numbers is approximately equal to 0.005.

(4). `erase_kn_num` in `/src`

For phone numbers and names, the number of neurons to eliminate is recommended to be between 10-200.
For random text, the number of neurons to eliminate is recommended to be between 200-500.

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
