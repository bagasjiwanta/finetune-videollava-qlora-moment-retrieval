import torch 


class DataCollatorWithPadding():
    r'''
    Pads and collates samples. This is needed as each batch has different shapes for flexibility.

    Use the `__call__` method to collate a batch

    returns batch with keys `['pixel_values_videos', 'labels' (if eval), 'attention_mask', 'input_ids']`
    '''
    def __init__(self, processor):
        self.processor = processor


    def __call__(self, features):
        input_ids = [feat['input_ids'][0] for feat in features]
        attention_mask = [feat['attention_mask'][0] for feat in features]
        pixel_values_videos = torch.cat([feat['pixel_values_videos'] for feat in features], dim=0)
        is_eval = 'answer' in features[0]
        self.processor.tokenizer.padding_side = 'left' if is_eval else 'right'
        
        batch = self.processor.tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True, return_tensors="pt",
        )

        if not is_eval:
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 # ignore index for nn.CrossEntropyLoss
        else:
            labels = [feat['answer'] for feat in features]

        output = (batch['input_ids'], batch['attention_mask'], pixel_values_videos, labels)
        return output if not is_eval else output + ([feat['ts_info'] for feat in features],)