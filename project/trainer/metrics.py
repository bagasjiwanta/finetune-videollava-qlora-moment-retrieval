from typing import List
from project.trainer.loss import batch_str_ts_to_int_arr, temporal_iou, batch_int_arr_to_spans, generalized_temporal_iou
import torch



def ao_exact_score(preds: List[str], labels: List[List[str]]):
    r'''
    Parameters:
        preds (`List[str]`): list of predictions, each prediction is a string from decoded
        labels (`List[str]`): list of labels, labels are string
    '''
    correct = 0
    for pred, label in zip(preds, labels): # str, List[str]
        try :
            pred = [p.strip() for p in pred.split(',')] # pred = List[str]
            correct += (
                len(pred) == len(label) and 
                all([pred[i] == label[i] for i in range(len(label))])
            )
        except:
            pass

    return correct / len(labels), correct


def mr_iou_score(
        preds: List[str], 
        ts_infos: List[List[str]], 
        labels: List[List[str]]
    ):   
    ts_preds, ts_infos, ts_labels = batch_str_ts_to_int_arr(preds, ts_infos, labels)
    span_preds, span_labels = batch_int_arr_to_spans(ts_preds, ts_infos, ts_labels)
    iou, _ = temporal_iou(span_preds, span_labels)
    
    return float(torch.diag(iou)), float(torch.diag(iou) * 4)

def mr_giou_score(
        preds: List[str], 
        ts_infos: List[List[str]], 
        labels: List[List[str]]
    ):  
    ts_preds, ts_infos, ts_labels = batch_str_ts_to_int_arr(preds, ts_infos, labels)
    span_preds, span_labels = batch_int_arr_to_spans(ts_preds, ts_infos, ts_labels)
    giou = generalized_temporal_iou(span_preds, span_labels)
    print(giou)
    return torch.diag(giou)