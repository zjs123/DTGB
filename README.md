# [DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs (Accepted by NeurIPS 2024 D&B Track)](https://arxiv.org/abs/2406.12072)

## Dataset
All eight dynamic text-attributed graphs provided by DTGB can be downloaded from [here](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk?usp=sharing).
<img width="1230" alt="image" src="https://github.com/zjs123/DTGB/assets/17922610/2f714dd7-7928-4eed-8e55-8e1fa947e463">

### Data Format
Each graph is preserved through three files.
* __edge_list.csv:__ stores each edge in DyTAG as a tuple. i.e., `(u, v, r, t, l)`. `u` is the id of the source entity, `v` is the id of the target entity, `r` is the id of the relation between them, `t` is the occurring timestamp of this edge, `l` is the label of this edge.
* __entity_text.csv:__ stores the mapping from entity ids (e.g., `u` and `v`) to the text descriptions of entities.
* __relation_text.csv:__ stores the mapping from relation ids (e.g., `r`) to the text descriptions of relations.

### Usage
* After downloading the datasets, they should be uncompressed into the `DyLink_Datasets` folder.
* Run `get_pretrained_embeddings.py` to obtain the Bert-based node and edge text embeddings. They will be saved as `e_feat.npy` and `r_feat.npy` respectively.
* Run `get_LLM_data.ipynb` to get the train and test set for the textual relation generation task. They will be saved as `LLM_train.pkl` and `LLM_test.pkl` respectively.

## Reproduce the Results

### Future Link Prediction Task
* Example of training *DyGFormer* on *GDELT* dataset without text attributes:
```{bash}
python train_link_prediction.py --dataset_name GDELT --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0 --use_feature no
```

* Example of training *DyGFormer* on *GDELT* dataset with text attributes:
```{bash}
python train_link_prediction.py --dataset_name GDELT --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0 --use_feature Bert
```
* The __AP__ and __AUC-ROC__ metrics on the test set (both transductive setting and inductive setting) will be automatically saved in `saved_resuts/DyGFormer/GDELT/DyGFormer_seed0no.json`
* The best checkpoint will be saved in `saved_resuts/DyGFormer/GDELT/` folder, and the checkpoint will be used to reproduce the performance on the node retrieval task.

### Destination Node Retrieval Task
After obtaining the best checkpoint on the Future Link Prediction Task. The __Hits@k__ metrics of the Destination Node Retrieval Task can be reproduced by running:
```{bash}
python evaluate_node_retrieval.py --dataset_name GDELT --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --negative_sample_strategy random --num_runs 5 --gpu 0  --use_feature no
```
* The `negative_sample_strategy` hyper-parameter is used to control the candidate sampling strategies, which can be `random` and `historical`.
* The `use_feature` hyper-parameter is used to control whether to use Bert-based embeddings, which can be `no` and `Bert`.

### Edge Classification Task
* Example of training *DyGFormer* on *GDELT* dataset without text attributes:
```{bash}
python train_edge_classification.py --dataset_name GDELT --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0 --use_feature no
```
* The __Precision__, __Recall__, and __F1-score__ metrics on the test set will be automatically saved in `saved_resuts/DyGFormer/GDELT/edge_classification_DyGFormer_seed0no.json`

### Textual Relation Generation Task
After obtaining the `LLM_train.pkl` and `LLM_test.pkl` files. You can directly reproduce the performance of original LLMs by running
```{bash}
python LLM_eval.py -config_path=LLM_configs/vicuna_7b_qlora_uncensored.yaml -model=raw
```
* You can change the LLMs through the `config_path` hyper-parameter.
* The generated text will be saved in `s_his_o_des_his_result_vicuna7b.pkl`.

And then to get the __Bert_score__ metrics, you should change the file path in `LLM_metric.py` and run:
```{bash}
python LLM_metric.py
```

If you want to fine-tune the LLMs, you should run:
```{bash}
python LLM_train.py LLM_configs/vicuna_7b_qlora_uncensored.yaml
```
and then reproduce the performance of the fine-tunned LLMs by running
```{bash}
python LLM_eval.py -config_path=LLM_configs/vicuna_7b_qlora_uncensored.yaml -model=lora
```

## Contact
For any questions or suggestions, you can use the issues section or contact us at (zjss12358@gmail.com).

## Acknowledge
Codes and model implementations are referred to [DyGLib](https://github.com/yule-BUAA/DyGLib) project. Thanks for their great contributions!

### Reference
```
@article{zhang2024dtgb,
  title={DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs},
  author={Zhang, Jiasheng and Chen, Jialin and Yang, Menglin and Feng, Aosong and Liang, Shuang and Shao, Jie and Ying, Rex},
  journal={arXiv preprint arXiv:2406.12072},
  year={2024}
}
```
