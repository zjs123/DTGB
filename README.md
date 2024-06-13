# DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs

## Dataset
All eight dynamic text-attributed graphs provided by DTGB can be downloaded from [here](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk?usp=sharing).
<img width="1230" alt="image" src="https://github.com/zjs123/DTGB/assets/17922610/2f714dd7-7928-4eed-8e55-8e1fa947e463">

### Data Format
Each graph is preserved through three files.
* __edge_list.csv:__ stores each edge in DyTAG as a tuple. i.e., `(u, v, r, t, l)`. `u` is the id of the source entity, `v` is the id of the target entity, `r` is the id of the relation between them, `t` is the occurring timestamp of this edge, `l` is the label of this edge.
* __entity_text.csv:__ stores the mapping from entity ids (e.g., `u` and `v`) to the text descriptions of entities.
* __relation_text.csv:__ stores the mapping from relation ids (e.g., `r`) to the text descriptions of relations.

### Using
* After downloading the datasets, they should be uncompressed into the `DyLink_Datasets` folder.
* Run `get_pretrained_embeddings.py` to obtain the Bert-based node and edge text embeddings. They will be saved as `e_feat.npy` and `r_feat.npy` respectively.
* Run `get_LLM_data.ipynb` to get the train and test set for the textual relation generation task. They will be saved as `LLM_train.pkl` and `LLM_test.pkl` respectively.
