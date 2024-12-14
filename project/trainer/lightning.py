import lightning as L
import bitsandbytes as bnb
from project.trainer.metrics import ao_exact_score


class VideoLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

    def training_step(self, batch):

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
        is_ao = len(labels[0][0]) == 1 

        if not is_ao:
            frame_info = batch[-1]

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

        if is_ao:
            score, correct = ao_exact_score(predictions, labels)
        else:
            score = 1
            correct = len(predictions)
        self.log("val_accuracy", score)

        return correct

    def configure_optimizers(self):
        # use 8 bit optimizer
        optimizer = bnb.optim.Adam8bit(self.parameters(), min_8bit_size=16384, lr=self.config.get("lr"))
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer