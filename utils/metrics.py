import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

def get_rank(target_score, candidate_score):
    tmp_list = target_score - candidate_score
    rank = len(tmp_list[tmp_list < 0]) + 1
    return rank


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_retrival_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    """
    get metrics for the link prediction task
    :param pos_scores: Tensor, shape (num_samples, )
    :param neg_scores: Tensor, shape (neg_size, num_samples)
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    try:
        pos_scores = pos_scores.cpu().detach().numpy()
    except:
        pass
    try:
        neg_scores = np.array([sub_score.cpu().numpy() for sub_score in neg_scores]).T # num_samples * neg_size
    except:
        neg_scores = np.array([sub_score for sub_score in neg_scores]).T # num_samples * neg_size

    H1, H3, H10 = [], [], []
    for i in range(len(pos_scores)):
        rank = get_rank(pos_scores[i], neg_scores[i])
        if rank <= 1:
            H1.append(1)
        else:
            H1.append(0)
        
        if rank <= 3:
            H3.append(1)
        else:
            H3.append(0)

        if rank <= 10:
            H10.append(1)
        else:
            H10.append(0)

    return {'H1': np.mean(H1), 'H3': np.mean(H3), 'H10': np.mean(H10)}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}

def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}

def get_edge_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the edge classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    P_macro = precision_score(labels, predicts, average="macro")
    R_macro = recall_score(labels, predicts, average="macro")
    F_macro = f1_score(labels, predicts, average="macro")

    P_micro = precision_score(labels, predicts, average="micro")
    R_micro = recall_score(labels, predicts, average="micro")
    F_micro = f1_score(labels, predicts, average="micro")

    P_weight = precision_score(labels, predicts, average="weighted")
    R_weight = recall_score(labels, predicts, average="weighted")
    F_weight = f1_score(labels, predicts, average="weighted")

    return {'p_macro': P_macro, 'R_macro': R_macro, 'F_macro': F_macro, 'p_micro': P_micro, 'R_micro': R_micro, 'F_micro': F_micro, 'p_weight': P_weight, 'R_weight': R_weight, 'F_weight': F_weight}
