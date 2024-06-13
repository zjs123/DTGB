import json
import yaml
import torch
import random
import pickle
import argparse
import numpy as np
import transformers
import pandas as pd
import networkx as nx
from tqdm import tqdm
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer)

# Define a custom data collator
class CustomDataCollator:
    def __call__(self, batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

class QloraTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        self.merged_model = None
        if 'vicuna' in self.config["base_model"] or 'llama' in self.config["base_model"]:
            self.human_instruction = ['### HUMAN:', '### RESPONSE:']
        else:
            self.human_instruction = ['[INST]', '[/INST]']

    def load_base_model(self):
        model_id = self.config["base_model"]
        print(model_id)

        if 'llama' in model_id:
            bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.bfloat16
            )
            print('load llama 3')
            access_token = 'your_token'
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
            tokenizer.model_max_length = 512
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map={"":0}, token=access_token)
        else:
            bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.bfloat16
            )
            access_token = 'your_token'
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=access_token)
            if not tokenizer.pad_token:
                # Add padding token if missing, e.g. for llama tokenizer
                #tokenizer.pad_token = tokenizer.eos_token  # https://github.com/huggingface/transformers/issues/22794
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        self.tokenizer = tokenizer
        self.base_model = model

    def load_adapter_model(self, adapter_path: str):
        """ Load pre-trained lora adapter """
        self.adapter_model = PeftModel.from_pretrained(self.base_model, adapter_path)

    def train(self):
        # Set up lora config or load pre-trained adapter
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=self.config["target_modules"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(self.base_model, config)
        self._print_trainable_parameters(model)

        print("Start data preprocessing")
        # TODO: Expand this to cover more dataset types and processing patterns
        data = self._process_data() #self._process_vicuna_data()

        print("Start training")
        trainer = transformers.Trainer(
            model=model,
            train_dataset=data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                warmup_steps=100,
                #max_steps=200,  # short run for debugging
                num_train_epochs=1,  # full run
                learning_rate=2e-4,
                fp16=True,
                logging_steps=20,
                output_dir=self.config["trainer_output_dir"],
                report_to="wandb",
                #optim="adamw"
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        try:
            trainer.train()
        except:
            pass
        model_save_path = f"{self.config['model_output_dir']}/{self.config['model_name']}_adapter_Stack_elec"
        trainer.save_model(model_save_path)
        self.adapter_model = model
        print(f"Training complete, adapter model saved in {model_save_path}")

    def push_to_hub(self):
        """ Push merged model to HuggingFace Hub """
        raise NotImplementedError("push_to_hub not implemented yet")

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    
    def _process_data(self):
        sample_num = 3
        context_window = 2048
        dataset_name = 'Stack_elec' # Stack_elec Googlemap_CT
        train_data = pickle.load(open(dataset_name +'/LLM_train.pkl', 'rb'))

        entity_text_reader = pd.read_csv(dataset_name + '/entity_text.csv', chunksize=1000)
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
        

        relation_text_reader = pd.read_csv(dataset_name + '/relation_text.csv', chunksize=1000)
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

        data_s_his_o_des_his = []
        for key in list(train_data.keys()):
            s, r, o, l, t = key
            s_his, o_his, pair_his = train_data[key]

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

            datapoint = {'s_des_text':s_des_text, 'o_des_text':o_des_text, 's_his_r_text':s_his_r_text, 's_his_o_text':s_his_o_text, 'o_his_r_text':o_his_r_text, 'pair_his_r_text':pair_his_r_text, 'truth':ground_truth}
            data_s_his_o_des_his.append(datapoint)

        data_tokenized = []
        data_tokenized += [self.tokenizer(self._generate_s_his_o_des_his(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_s_his_o_des_his)]
        random.shuffle(data_tokenized)
        return data_tokenized

    def _generate_s_his_o_des_his(self, data_point: dict, eos_token: str, instruct: bool = False):
        Q = self.human_instruction[0] + "\n"
        Q = Q + "Description of item A: " + data_point['o_des_text'] + '\n'
        Q = Q + "Recent reviews of item A from other users: \n"
        for i in range(len(data_point['o_his_r_text'])):
            Q = Q + str(i) + '. ' + data_point['o_his_r_text'][i] + '\n'
        
        Q = Q + "Recent reviews of User P to other items: \n"
        for i in range(len(data_point['s_his_o_text'])):
            Q = Q + str(i) + '. item: ' + data_point['s_his_o_text'][i] + ' review: ' + data_point['s_his_r_text'][i] + '\n'
        
        Q = Q + "If User P visit Item A in the next time, please give me one possible review of User P to Item A. \n"
        Q = Q + self.human_instruction[1] + "\n"

        Q = Q + data_point['truth'] + eos_token
        

        return Q

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()
    
    def read_yaml_file(file_path):
        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as e:
                print(f"Error reading YAML file: {e}")

    config = read_yaml_file(args.config_path)
    trainer = QloraTrainer(config)

    print("Load base model")
    trainer.load_base_model()

    print("Start training")
    trainer.train()

#CUDA_VISIBLE_DEVICES=0  python LLM_train.py LLM_configs/vicuna_7b_qlora_uncensored.yaml
