import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel

if __name__ == "__main__":
    precision = 8
    batch_size = 1000
    data_set_name = 'Amazon_movies'
    pretrained_model_name = 'bert-base-uncased'

    # Prepare PLM modle and tokenizers
    config = AutoConfig.from_pretrained(pretrained_model_name)
    hidden_size = config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    PLM = AutoModel.from_pretrained(pretrained_model_name).cuda()
    print("PLM initialized")

    # [ 'Amazon_movies', 'Enron', 'GDELT', 'Googlemap_CT', 'ICEWS1819', 'Stack_elec', 'Stack_english', 'Stack_ubuntu', 'Yelp']
    for data_set_name in ['GDELT']:
        print(data_set_name)
        edge_list = pd.read_csv(data_set_name + '/edge_list.csv')
        num_node = max(edge_list['u'].max(), edge_list['i'].max())
        num_rel = edge_list['r'].max()

        # Prepare datasets
        entity_embeddings = [np.zeros(hidden_size)]
        entity_text_reader = pd.read_csv(data_set_name + '/entity_text.csv', chunksize=batch_size)

        with tqdm(total = num_node) as pbar:
            for batch in entity_text_reader:
                id_batch = batch['i'].tolist()
                text_batch = batch['text'].tolist()
                if 0 in id_batch:
                    id_batch = id_batch[1:]
                    text_batch = text_batch[1:]
                if np.nan in text_batch:
                    text_batch = ['NULL' if type(i) != str else i for i in text_batch]
                encoded_input = tokenizer(text_batch, padding = True, truncation=True, max_length=512, return_tensors='pt')
                with torch.no_grad():
                    output = PLM(**encoded_input.to('cuda'))[1]
                    output = output.cpu()
                for i in range(len(output)):
                    assert len(entity_embeddings) == id_batch[i]
                    entity_embeddings.append(np.around(output[i].numpy(), precision))
                    pbar.update(1)
        entity_embeddings = np.array(entity_embeddings)
        print([entity_embeddings.shape, num_node])
        assert len(entity_embeddings) == num_node + 1
        np.save(data_set_name + '/e_feat.npy', entity_embeddings)

        # Prepare datasets
        rel_embeddings = [np.zeros(hidden_size)]
        rel_text_reader = pd.read_csv(data_set_name + '/relation_text.csv', chunksize=batch_size)

        with tqdm(total = num_rel) as pbar:
            for batch in rel_text_reader:
                id_batch = batch['i'].tolist()
                text_batch = batch['text'].tolist()
                if 0 in id_batch:
                    id_batch = id_batch[1:]
                    text_batch = text_batch[1:]
                if np.nan in text_batch:
                    text_batch = ['NULL' if type(i) != str else i for i in text_batch]
                encoded_input = tokenizer(text_batch, padding = True, truncation=True, max_length=512 , return_tensors='pt')
                with torch.no_grad():
                    output = PLM(**encoded_input.to('cuda'))[1]
                    output = output.cpu()
                for i in range(len(output)):
                    assert len(rel_embeddings) == id_batch[i]
                    rel_embeddings.append(np.around(output[i].numpy(), precision))
                    pbar.update(1)

        rel_embeddings = np.array(rel_embeddings)
        assert len(rel_embeddings) == num_rel + 1
        print(rel_embeddings.shape)
        np.save(data_set_name + '/r_feat.npy', rel_embeddings)
