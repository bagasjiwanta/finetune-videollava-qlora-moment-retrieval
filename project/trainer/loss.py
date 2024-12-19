import torch
from torch import Tensor
from torch import tensor
from typing import List, Tuple 
from bisect import bisect_left, bisect_right
import time
from torch.nn import L1Loss


def mr_l1_loss():
    return L1Loss()


def temporal_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> temporal_iou(test_spans1, test_spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M)

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / union
    return iou, union


def generalized_temporal_iou(spans1, spans2):
    """
    Generalized IoU from https://giou.stanford.edu/
    Also reference to DETR implementation of generalized_box_iou
    https://github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40

    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor

    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> generalized_temporal_iou(test_spans1, test_spans2)
    tensor([[ 0.6667,  0.2000],
        [-0.2000,  0.5000]])
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    # assert (spans1[:, 1] >= spans1[:, 0]).all()
    # assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    enclosing_area = (right - left).clamp(min=0)  # (N, M)

    return iou - (enclosing_area - union) / enclosing_area


def ts_str_to_int_sec(ts: str) -> int:
    m, s = 0, 0
    try:
        ts_2 = ts.split(":")
        s = int(ts_2[-1])
        m = int(ts_2[-2]) * 60
    except:
        pass
    finally:
        return m + s


def ts_str_range_to_int_sec_range(ts: str, is_eval = True) -> Tuple[int, int]:
    ts_st, ts_ed = 0, 0
    try:
        ts2 = ts.split(",")
        ts_st2, ts_ed2 = ts2[0].strip(), ts2[1].strip()
        ts_st = ts_str_to_int_sec(ts_st2)
        ts_ed = ts_str_to_int_sec(ts_ed2)
    except:
        pass 
    finally:
        return (ts_st, ts_ed)
    

def batch_int_arr_to_bbox(
    ts_preds: List[Tuple[int, int]], 
    ts_infos: List[List[int]],
    ts_labels: List[Tuple[int, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(ts_infos)
    T = len(ts_infos[0])
    preds = torch.zeros((B, T), dtype=torch.int)
    labels = torch.zeros((B, T), dtype=torch.int)

    for b in range(B):
        pred, info, label = ts_preds[b], ts_infos[b], ts_labels[b]
        ts_pred_st = bisect_left(info, pred[0])  # find leftmost closest (0 if less than 0)
        ts_label_st = bisect_left(info, label[0])  

        ts_pred_ed = bisect_right(info, pred[1])  # find rightmost closest
        ts_label_ed = bisect_right(info, label[1])

        preds[b][ts_pred_st:ts_pred_ed + 1] = 1  # mask fill preds 
        labels[b][ts_label_st:ts_label_ed + 1] = 1  # mask fill labels
 
    return preds, labels


def batch_int_arr_to_spans(
    ts_preds: List[Tuple[int, int]], 
    ts_infos: List[List[int]],
    ts_labels: List[Tuple[int, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Convert preds and labels to (B, 2) and (B, 2) sized tensor
    Each element is tensor(st, ed)
    For model evaluation
    '''
    B = len(ts_infos)
    # print(ts_preds, ts_labels)
    preds, labels = torch.zeros((B,2), dtype=torch.int), torch.zeros((B,2), dtype=torch.int)
    for b in range(B):
        p, l, info = ts_preds[b], ts_labels[b], ts_infos[b]
        preds[b][0] = bisect_left(info, p[0])
        preds[b][1] = bisect_right(info, p[1])
        labels[b][0] = bisect_left(info, l[0])
        labels[b][1] = bisect_right(info, l[1])
        
    return preds, labels


def batch_str_ts_to_int_arr(
    ts_preds: List[str],
    ts_infos: List[List[str]],
    ts_labels: List[List[str]] 
) -> Tuple[
    Tuple[Tuple[int]], 
    tuple[Tuple[int]], 
    Tuple[Tuple[int]]
    ]:
    B = len(ts_infos)
    for b in range(B):  
        ts_infos[b] = tuple(ts_str_to_int_sec(ts_info) for ts_info in ts_infos[b])
        ts_labels[b] = tuple(ts_str_to_int_sec(_ts_q) for _ts_q in ts_labels[b])
    ts_preds = [ts_str_range_to_int_sec_range(_ts_a) for _ts_a in ts_preds]

    return ts_preds, ts_infos, ts_labels


def batch_raw_ts_to_tensor_bbox(
    ts_preds: List[str],
    ts_infos: List[List[str]],
    ts_labels: List[List[str]] 
) -> Tuple[Tensor, Tensor]:
    ts_preds, ts_infos, ts_labels = batch_str_ts_to_int_arr(ts_preds, ts_infos, ts_labels)

    preds, labels = batch_int_arr_to_bbox(
        ts_preds=ts_preds,
        ts_infos=ts_infos,
        ts_labels=ts_labels
    )
    return preds, labels


