from typing import List


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

