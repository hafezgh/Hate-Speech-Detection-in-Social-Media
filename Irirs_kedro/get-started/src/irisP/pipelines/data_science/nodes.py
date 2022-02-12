# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name
import gc
import torch
import logging
import datasets
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer, TrainingArguments, AdamW
from .custom_models.BERTShallowFC import BERTShallowFC
from .custom_models.BERTDeepFC import BERTDeepFC
from .custom_models.BERTRNN import BERTRNN
from .custom_models.BERTCNN import BERTCNN
from typing import Any, Dict
from torch import nn


def _compute_metrics(prediction):
    labels = prediction.label_ids
    predictions = prediction.predictions.argmax(-1)
    # calculate metrics using sklearn's function
    accuracy = accuracy_score(labels, predictions)

    f_score_macro = f1_score(labels, predictions, pos_label=1, average='macro')
    precision_macro = precision_score(labels, predictions, pos_label=1, average='macro')
    recall_macro = recall_score(labels, predictions, pos_label=1, average='macro')

    f1_score_weighted = f1_score(labels, predictions, pos_label=1, average='weighted')
    precision_weighted = precision_score(labels, predictions, pos_label=1, average='weighted')
    recall_weighted = recall_score(labels, predictions, pos_label=1, average='weighted')

    return {
        'accuracy': accuracy,
        'f1_score_weighted': f1_score_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_score_macro': f_score_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
    }


def _classic_training(model, dataloader, device, cross_entropy, optimizer, batch_size):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    total = len(dataloader)
    for i, batch in enumerate(dataloader):

        # # verbose
        step = i+1
        percent = "{0:.2f}".format(100 * (step / float(total)))
        lossp = "{0:.2f}".format(total_loss/(total*batch_size))
        filledLength = int(100 * step // total)
        bar = '█' * filledLength + '>'  *(filledLength < 100) + '.' * (99 - filledLength)
        print(f'\rBatch {step}/{total} |{bar}| {percent}% complete, loss={lossp}, accuracy={total_accuracy}', end='')

        # push the batch to gpu
        batch = [r.to(device) for r in batch.values()]
        mask, sent_id, labels = batch

        del batch
        gc.collect()
        torch.cuda.empty_cache()
        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)[1]

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss += float(loss.item())

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # append the model predictions
        total_preds.append(preds.detach().cpu().numpy())

    gc.collect()
    torch.cuda.empty_cache()

    # compute the training loss of the epoch
    avg_loss = total_loss / (len(dataloader)*batch_size)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


def _classic_evaluate(model, dataloader, batch_size, device, cross_entropy, optimizer):
    print("\n\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    total = len(dataloader)
    for i, batch in enumerate(dataloader):

        step = i+1
        percent = "{0:.2f}".format(100 * (step / float(total)))
        lossp = "{0:.2f}".format(total_loss/(total*batch_size))
        filledLength = int(100 * step // total)
        bar = '█' * filledLength + '>' * (filledLength < 100) + '.' * (99 - filledLength)
        print(f'\rBatch {step}/{total} |{bar}| {percent}% complete, loss={lossp}, accuracy={total_accuracy}', end='')

        # push the batch to gpu
        batch = [r.to(device) for r in batch.values()]
        mask, sent_id, labels = batch

        del batch
        gc.collect()
        torch.cuda.empty_cache()
        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)[1]

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss += float(loss.item())
            #preds = preds.detach().cpu().numpy()

            #total_preds.append(preds)
            total_preds.append(preds.detach().cpu().numpy())

    gc.collect()
    torch.cuda.empty_cache()

    # compute the validation loss of the epoch
    avg_loss = total_loss / (len(dataloader)*batch_size)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


def train(
        BERT_model: str,
        model_name: str,
        dataset: str,
        sentence_max_length: int,
        training_type: str,
        train_args: Dict[str, Any],
        device: str,
        train_dataset: datasets.Dataset,
        eval_dataset: datasets.Dataset,
) -> Dict[str, Any]:

    # Set number of classes
    if dataset in ["davidson", "waseem", "multilingual"]:
        num_classes = 3
    elif dataset in ["sentiment"]:
        num_classes = 2
    else:
        raise Exception("Unclear number of classes")

    # Get BERT model
    if BERT_model == "shallow_fc":
        model = BERTShallowFC(model_name=model_name,
                              device=device,
                              num_classes=num_classes)
    elif BERT_model == "deep_fc":
        model = BERTDeepFC(model_name=model_name,
                           device=device,
                           num_classes=num_classes)
    elif BERT_model == "rnn":
        model = BERTRNN(model_name=model_name,
                        device=device,
                        num_classes=num_classes)
    elif BERT_model == "cnn":
        model = BERTCNN(model_name=model_name,
                        device=device,
                        max_length=sentence_max_length,
                        num_classes=num_classes)
    else:
        raise Exception("Unknown model")

    if training_type == "transformers":
        training_args = TrainingArguments(**train_args)

        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=eval_dataset,          # evaluation dataset
            compute_metrics=_compute_metrics,     # the callback that computes metrics of interest
        )

        trainer.train()

        return dict(
            trained_model=trainer.model
        )

    elif training_type == "classic":
        # sampler for sampling the data during training
        train_sampler = RandomSampler(train_dataset)

        # sampler for sampling the data during training
        val_sampler = SequentialSampler(eval_dataset)

        dataloaders = {
            'train': DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size=train_args["per_device_train_batch_size"]
                                ),
            'eval': DataLoader(eval_dataset,
                               sampler=val_sampler,
                               batch_size=train_args["per_device_train_batch_size"]
                               )
        }

        # push the model to GPU
        model = model.to(device)

        # define the optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # loss function
        cross_entropy = nn.NLLLoss()

        # set initial loss to infinite
        best_valid_loss = float('inf')

        # number of training epochs
        epochs = train_args["num_train_epochs"]
        current = 1

        training_params = {
            "model": model,
            "dataloader": dataloaders["train"],
            "device": device,
            "cross_entropy": cross_entropy,
            "optimizer": optimizer,
            "batch_size": train_args["per_device_train_batch_size"]
           }

        eval_params = training_params.copy()
        eval_params["dataloader"] = dataloaders["eval"]

        # for each epoch
        while current <= epochs:

            print(f'\nEpoch {current} / {epochs}:')

            # train model
            train_loss, _ = _classic_training(**training_params)

            # evaluate model
            valid_loss, _ = _classic_evaluate(**eval_params)

            # save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            # append training and validation loss

            print(f'\n\nTraining Loss: {train_loss:.4f}')
            print(f'Validation Loss: {valid_loss:.4f}')

            current = current + 1

        return dict(
            trained_model=model
        )

    else:
        raise Exception("Unknown training type")


def predict(model: Any,
            test_ds: datasets.Dataset,
            training_type: str,
            train_args: Dict[str, Any],
            device: str
) -> Dict[str, Any]:
    """

    :param model:
    :param test_ds:
    :param training_type:
    :param train_args:
    :param device:
    :return:
    """
    if training_type == "transformers":
        training_args = TrainingArguments(**train_args)

        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            compute_metrics=_compute_metrics,     # the callback that computes metrics of interest
        )

        predictions = trainer.predict(test_ds)

        return dict(
            predictions=predictions
        )

    elif training_type == "classic":
        test_mask = test_ds[:]["attention_mask"]
        test_seq = test_ds[:]["input_ids"]
        test_y = test_ds[:]["label"]

        with torch.no_grad():
            predictions = model(test_seq.to(device), test_mask.to(device))
            predictions = predictions[1].detach().cpu().numpy()


        # model's performance
        predictions = {
            "pred": np.argmax(predictions, axis=1),
            "test_y": test_y
        }

        return dict(
            predictions=predictions
        )
    else:
        raise Exception("Unknown training type")


def report_metrics(
        predictions: Any,
        training_type: str
    ) -> None:
    """

    :param training_type:
    :param predictions:
    :return:
    """

    if training_type == "transformers":
        report = predictions[2]

        # Log the accuracy of the model
        log = logging.getLogger(__name__)
        log.info(f"Model accuracy on test set: {report['test_accuracy']:.4f}")

    elif training_type == "classic":
        print("Performance:")
        print('Classification Report')
        print(classification_report(predictions["test_y"], predictions["pred"]))
        print("Accuracy: " + str(accuracy_score(predictions["test_y"], predictions["pred"])))
