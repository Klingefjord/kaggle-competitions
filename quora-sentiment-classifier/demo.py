#!/usr/bin/env python3

from absl import app
from absl import flags
from absl import logging

import sh

import torch
import pytorch_lightning as pl

import nlp
import transformers

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('learning_rate', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

class QuoraSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)
        def _tokenize(x): # single leading underscore -> this is for internal use only!
            x['input_ids'] = tokenizer.encode(
                x['text'], 
                max_length=FLAGS.seq_length, 
                pad_to_max_length=True
            )
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset('imdb', split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else "5%"}]') 
            ds = ds.map(_tokenize)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_index):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_index):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        accuracy = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': accuracy}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(         
            self.train_ds,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=True,
        )

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=FLAGS.learning_rate,
            momentum=FLAGS.momentum
        )

def main(_):
    model = QuoraSentimentClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0),
    )
    trainer.fit(model)

if __name__ == "__main__":
    app.run(main)