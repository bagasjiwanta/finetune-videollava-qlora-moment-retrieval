from project.trainer.metrics import mr_iou_score, mr_giou_score
from project.trainer.loss import temporal_iou
from torch import tensor
import torch


if __name__ == "__main__":
    iou = mr_iou_score(
        preds=['00:02, 00:06'],
        ts_infos=[
            ['00:00', '00:02', '00:04', '00:06', '00:08', '00:11', '00:13', '00:15', '00:17', '00:19', '00:21', '00:23', '00:25', '00:27'],
        ],
        labels=[['00:11', '00:13']]
    )
    print(iou)
    giou = mr_giou_score(
        preds=['00:02, 00:06'],
        ts_infos=[
            ['00:00', '00:02', '00:04', '00:06', '00:08', '00:11', '00:13', '00:15', '00:17', '00:19', '00:21', '00:23', '00:25', '00:27'],
        ],
        labels=[['00:10', '00:13']]
    )
    print(giou)
    # print(pred, label, sep="\n")

    # print(temporal_iou(
    #     tensor([[2., 5.], [1., 3.]]),
    #     tensor([[0., 6.], [0., 6.]])
    # ))

    # print(temporal_iou(
    #     tensor([[12., 24.], [4., 7.]]),
    #     tensor([[0., 30.], [0., 19.]])
    # ))


