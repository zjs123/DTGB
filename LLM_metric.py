from rouge import Rouge
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import pickle
import ipdb
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
chencherry = SmoothingFunction()
def get_Bleu_score(candidate, reference, smoothing_function=chencherry.method1):
    reference = reference.strip().split(' ')
    candidate = candidate.strip().split(' ')
    score = sentence_bleu(reference, candidate)
    return score
def get_ROUGE_score(candidate, reference):
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=candidate, refs=reference)
    return rouge_score[0]["rouge-l"]['p'], rouge_score[0]["rouge-l"]['r'], rouge_score[0]["rouge-l"]['f']
def get_bert_score(candidate, reference):
    P, R, F1 = score([candidate], [reference],lang="en")
    return P, R, F1
dataset_name = "Stack_ubuntu"

#["llama3", "vicuna7b","vicuna13b","mistral"]:
test_data = pickle.load(open(dataset_name +'/s_his_o_des_his_result_llama.pkl', 'rb'))
result = {}
result["bleu"] = []
result["rouge_p"] = []
result["rouge_r"] = []
result["rouge_f"] = []
result["bert_p"] = []
result["bert_r"] = []
result["bert_f"] = []
for item in tqdm(test_data):
    try:
        pred = item[1]
        groundtruth = item[0]
        belu = get_Bleu_score(pred, groundtruth)
        rouge_p, rouge_r, rouge_f = get_ROUGE_score(pred, groundtruth)
        bert_p, bert_r, bert_f = get_bert_score(pred, groundtruth)
        result["bleu"].append(belu)
        result["rouge_p"].append(rouge_p)
        result["rouge_r"].append(rouge_r)
        result["rouge_f"].append(rouge_f)
        result["bert_p"].append(bert_p)
        result["bert_r"].append(bert_r)
        result["bert_f"].append(bert_f)
    except:
        continue
final = {}
final["bleu"] = float(sum(result["bleu"]) / len(result["bleu"]))
final["rouge_p"] = float(sum(result["rouge_p"])/ len(result["rouge_p"]))
final["rouge_r"] = float(sum(result["rouge_r"])/ len(result["rouge_r"]))
final["rouge_f"] = float(sum(result["rouge_f"])/ len(result["rouge_f"]))
final["bert_p"] = float(sum(result["bert_p"])/ len(result["bert_p"]))
final["bert_r"] = float(sum(result["bert_r"])/ len(result["bert_r"]))
final["bert_f"] = float(sum(result["bert_f"])/ len(result["bert_f"]))
print(final)
#pickle.dump(final, open(‘dataset/‘+ dataset_name +f’/eval_result_{model}.pkl’, ‘wb’))
