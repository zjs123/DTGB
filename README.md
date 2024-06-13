# DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs

## Dataset
All eight dynamic text-attributed graphs provided by DTGB can be downloaded from [here](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk?usp=sharing).
### Format
Each graph is preserved through three files.
* __edge_list.csv:__ stores each edge in DyTAG as a tuple. i.e., `(u, v, r, t, l)`. `u` is the id of the source entity, `v` is the id of the target entity, `r` is the id of the relation between them, `t` is the occurring timestamp of this edge, `l` is the label of this edge.
* __entity_text.csv:__ stores the mapping from entity ids (e.g., `u` and `v`) to the text descriptions of entities.
* __relation_text.csv:__ stores the mapping from relation ids (e.g., `r`) to the text descriptions of relations.
