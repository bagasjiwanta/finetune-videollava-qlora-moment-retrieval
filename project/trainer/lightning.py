from lightning import LightningModule
from project.trainer.metrics import ao_exact_score, mr_iou_score
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch import Tensor


class VideoLlavaModelPLModule(LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.processor = processor
        self.model = model


    def training_step(self, batch):
        input_ids: Tensor
        attention_mask: Tensor
        pixel_values_videos: Tensor
        labels: Tensor
        input_ids, attention_mask, pixel_values_videos, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss


    def validation_step(self, batch):

        input_ids, attention_mask, pixel_values_videos, labels = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            max_new_tokens=50,
            do_sample=False,
        )
        # turn them back into text, chopping of the prompt
        predictions = self.processor.batch_decode(
            generated_ids[:, input_ids.size(1):], 
            skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        is_ao = len(labels[0][0]) == 1 

        if is_ao:
            score, correct = ao_exact_score(predictions, labels)
        else:
            frame_info = batch[-1]
            score, correct = mr_iou_score(predictions, frame_info, labels) 
            
        self.log("val_accuracy", score)

        return correct


    def configure_optimizers(self):
        # use 8 bit optimizer
        # optimizer = Adam8bit(self.parameters(), min_8bit_size=4096, lr=self.config.get("lr"))
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=2e-5)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer