import argparse
import pickle
import torch
import yaml
import random
import networkx as nx
import numpy as np
from tqdm import tqdm
#from langchain import PromptTemplate
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM,
                          AutoTokenizer, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, pipeline, AutoModelForCausalLM)

import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu

"""
Ad-hoc sanity check to see if model outputs something coherent
Not a robust inference platform!
"""

def get_Bleu_score(candidate, reference):
    reference = reference.strip().split(' ')
    candidate = candidate.strip().split(' ')

    score = sentence_bleu(reference, candidate)
    return score

def get_ROUGE_score(candidate, reference):
    rouge_score = rouge.get_scores(hyps=candidate, refs=reference)
    return rouge_score[0]["rouge-l"]['p'], rouge_score[0]["rouge-l"]['r'], rouge_score[0]["rouge-l"]['f']

def get_bert_score(candidate, reference):
    P, R, F1 = score([candidate], [reference])
    return P, R, F1

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def _generate_s_his_o_des_his(data_point: dict, eos_token: str, instruct: bool = False):
    Q = human_instruction[0] + "\n"
    Q = Q + "Description of item A: " + data_point['o_des_text'] + '\n'
    Q = Q + "Recent reviews of item A from other users: \n"
    for i in range(len(data_point['o_his_r_text'])):
        Q = Q + str(i) + '. ' + data_point['o_his_r_text'][i] + '\n'
    
    Q = Q + "Recent reviews of User P to other items: \n"
    for i in range(len(data_point['s_his_o_text'])):
        Q = Q + str(i) + '. item: ' + data_point['s_his_o_text'][i] + ' review: ' + data_point['s_his_r_text'][i] + '\n'
    
    Q = Q + "Give me three possible reviews of User P to Item A. \n"
    Q = Q + human_instruction[1] + "\n"

    print(Q)
    return Q

def test_s_his_o_des_his(): # 1, 3, 5, 10
    sample_num = 3 #3
    results = []
    for key in list(test_data.keys())[:500]:
        try:
            s, r, o, l, t = key
            s_his, o_his, pair_his = test_data[key]

            s_des_text, o_des_text = E_id_2_text[s], E_id_2_text[o]
            ground_truth = R_id_2_text[r]

            s_his_sample = s_his[:sample_num]
            o_his_sample = o_his[:sample_num]
            pair_his_sample = pair_his[:sample_num]

            s_his_r_text, s_his_o_text, o_his_r_text, pair_his_r_text = [], [], [], []

            for sample in s_his_sample:
                his_r, his_o = sample[1], sample[2]
                s_his_r_text.append(R_id_2_text[his_r])
                s_his_o_text.append(E_id_2_text[his_o])

            for sample in o_his_sample:
                his_r = sample[1]
                o_his_r_text.append(R_id_2_text[his_r])
            
            for sample in pair_his_sample:
                his_r = sample[1]
                pair_his_r_text.append(R_id_2_text[his_r])

            datapoint = {'s_des_text':s_des_text, 'o_des_text':o_des_text, 's_his_r_text':s_his_r_text, 's_his_o_text':s_his_o_text, 'o_his_r_text':o_his_r_text, 'pair_his_r_text':pair_his_r_text}
            prompt = _generate_s_his_o_des_his(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
            res = ans.strip().split(human_instruction[1]+'\n')[-1]

            print(res)
            results.append([ground_truth, res, key, datapoint])
            print([len(results), len(test_data)])
        except:
            continue
    pickle.dump(results, open('/gpfs/radev/scratch/ying_rex/jz875/DyLink_Datasets/'+ dataset_name +'/s_his_o_des_his_result_vicuna13b.pkl', 'wb'))

    
def get_llm_response(prompt):
    raw_output = pipe(prompt)
    return raw_output

if __name__ == "__main__":
    # init paramters
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", help="Path to the config YAML file")
    parser.add_argument("-model", help="Path to the config YAML file")
    parser.add_argument("-prompt_num", help="Path to the config YAML file", default = 1)
    args = parser.parse_args()
    config = read_yaml_file(args.config_path)

    # init model
    print("Load model")
    if args.model == 'raw':
        model_path = config["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)
    if args.model == 'lora': 
        model_path = config["base_model"]
        if 'llama' in model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, load_in_8bit=True)
            tokenizer.model_max_length = 512
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)

        adapter_save_path = f"{config['model_output_dir']}/{config['model_name']}_adapter_Stack_elec"
        model = PeftModel.from_pretrained(base_model, adapter_save_path)
        model = model.merge_and_unload()

    if 'mistral' in model_path:
        pipe = pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=6096,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            #max_new_tokens=1024,
        )
        human_instruction = ['[INST]', '[/INST]']
    
    elif 'llama' in model_path:
        pipe = pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=4096,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            #max_new_tokens=128,
        )
        human_instruction = ['### HUMAN:', '### RESPONSE:']
    else:
        pipe = pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=4096,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            #max_new_tokens=256,
        )
        human_instruction = ['### HUMAN:', '### RESPONSE:']
    rouge = Rouge()

    
    # init dataset
    dataset_name = 'Stack_ubuntu' # Stack_elec Googlemap_CT
    test_data = pickle.load(open('/gpfs/radev/project/ying_rex/jz875/DyLink_Datasets/'+ dataset_name +'/LLM_test.pkl', 'rb'))
    entity_text_reader = pd.read_csv('/gpfs/radev/scratch/ying_rex/jz875/DyLink_Datasets/' + dataset_name + '/entity_text.csv', chunksize=1000)
    E_id_2_text = {}
    for batch in tqdm(entity_text_reader):
        id_batch = batch['i'].tolist()
        text_batch = batch['text'].tolist()
        if 0 in id_batch:
            id_batch = id_batch[1:]
            text_batch = text_batch[1:]
        if np.nan in text_batch:
            text_batch = ['NULL' if type(i) != str else i for i in text_batch]
        for i in range(len(id_batch)):
            E_id_2_text[id_batch[i]] = text_batch[i]
    

    relation_text_reader = pd.read_csv('/gpfs/radev/scratch/ying_rex/jz875/DyLink_Datasets/' + dataset_name + '/relation_text.csv', chunksize=1000)
    R_id_2_text = {}
    for batch in tqdm(relation_text_reader):
        id_batch = batch['i'].tolist()
        text_batch = batch['text'].tolist()
        if 0 in id_batch:
            id_batch = id_batch[1:]
            text_batch = text_batch[1:]
        if np.nan in text_batch:
            text_batch = ['NULL' if type(i) != str else i for i in text_batch]
        for i in range(len(id_batch)):
            R_id_2_text[id_batch[i]] = text_batch[i]
    
    
    #test_s_his_o_des()
    test_s_his_o_des_his()

# python LLM_eval.py -config_path=LLM_configs/vicuna_7b_qlora_uncensored.yaml -model=raw
        
