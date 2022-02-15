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

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""
import copy

import numpy as np
from sklearn.model_selection import train_test_split
import re
import string
import pandas as pd
import emoji as emoji
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from datasets import Dataset
from typing import Any, Dict


def _tokenizer_wrapper(batch,
                       tokenizer,
                       tokenizer_max_length):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["text"],
        max_length=tokenizer_max_length,
        pad_to_max_length=True,
        padding="max_length",
        truncation=True,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["label"] = batch['class']

    return batch


def clean_data(
        dataset: str,
        multilingual_task: str,
        davidson: pd.DataFrame,
        waseem: pd.DataFrame,
        # ctc: pd.DataFrame,
        multilingual: pd.DataFrame,
        # sentiment: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """

    :param dataset:
    :param multilingual_task:
    :param davidson:
    :param waseem:
    :param ctc:
    :param multilingual:
    :param sentiment:
    :return:
    """

    if dataset in ["davidson", "waseem"]:
        data = davidson if dataset == "davidson" else waseem
        # print(data['label'].unique())
        data = data.drop(['count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
        tweets = data['tweet'].tolist()
        classes = data['class']

        new_values = list()
        # Emoticons
        emoticons = [':-)', ':)', '(:', '(-:', ':))', '((:', ':-D', ':D', 'X-D', 'XD', 'xD', 'xD', '<3', '</3', ':\*',
                     ';-)',
                     ';)', ';-D', ';D', '(;', '(-;', ':-(', ':(', '(:', '(-:', ':,(', ':\'(', ':"(', ':((', ':D', '=D',
                     '=)',
                     '(=', '=(', ')=', '=-O', 'O-=', ':o', 'o:', 'O:', 'O:', ':-o', 'o-:', ':P', ':p', ':S', ':s', ':@',
                     ':>',
                     ':<', '^_^', '^.^', '>.>', 'T_T', 'T-T', '-.-', '*.*', '~.~', ':*', ':-*', 'xP', 'XP', 'XP', 'Xp',
                     ':-|',
                     ':->', ':-<', '$_$', '8-)', ':-P', ':-p', '=P', '=p', ':*)', '*-*', 'B-)', 'O.o', 'X-(', ')-X']

        for value in tweets:
            # Remove dots
            text = value.replace(".", "").lower()
            text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
            users = re.findall("[@]\w+", text)
            for user in users:
                text = text.replace(user, "<user>")
            urls = re.findall(r'(https?://[^\s]+)', text)
            if len(urls) != 0:
                for url in urls:
                    text = text.replace(url, "<url >")
            for emo in text:
                if emo in emoji.UNICODE_EMOJI:
                    text = text.replace(emo, "<emoticon >")
            for emo in emoticons:
                text = text.replace(emo, "<emoticon >")
            numbers = re.findall('[0-9]+', text)
            for number in numbers:
                text = text.replace(number, "<number >")
            text = text.replace('#', "<hashtag >")
            text = re.sub(r"([?.!,¿])", r" ", text)
            text = "".join(l for l in text if l not in string.punctuation)
            text = re.sub(r'[" "]+', " ", text)
            new_values.append(text)

        df = pd.DataFrame.from_dict({
            'text': new_values,
            'class': classes
        })

    elif dataset == 'multilingual':
        assert(multilingual_task == 'task1' or multilingual_task == 'task2')

        data = multilingual

        df = pd.DataFrame.from_dict({
            'text': data['text'],
            'class': data[multilingual_task]
        })
    # elif dataset == "sentiment":
    #     pass
    # elif dataset == "ctc":
    #     df = ctc.copy()
    #     df.columns = ["class", "text"]
    #
    #     df['class'] = df['class'].map(
    #         lambda x: int(x.split('__label__')[1])
    #     )
    #
    #     df['text'] = df['text'].map(
    #         lambda x: x.split('study interventions are ')[1]
    #     )
    else:
        raise Exception("Unknown dataset")

    return dict(
        cleaned_dataset=df
    )


def prepare_data(cleaned_dataset: pd.DataFrame,
                 model_name: str,
                 tokenize_batch_size: int,
                 tokenizer_max_length: int
) -> Dict[str, Any]:
    """

    :param cleaned_dataset:
    :param model_name:
    :param tokenize_batch_size:
    :param tokenizer_max_length:
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    # Tokenize texts
    train_ds = Dataset.from_pandas(cleaned_dataset)

    train_ds = train_ds.map(
        lambda x: _tokenizer_wrapper(x, tokenizer, tokenizer_max_length),
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=["text", "class"]
    )

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return dict(
        dataset=train_ds
    )


def split_data(dataset: pd.DataFrame,
               unbalanced: bool,
               train_size_ratio: float,
               test_size_ratio: float
) -> Dict[str, pd.DataFrame]:
    """

    :param dataset:
    :param unbalanced:
    :param train_size_ratio:
    :param test_size_ratio:
    :return:
    """

    if unbalanced:
        df = dataset.to_pandas()
        # print(df.head())
        train_df, valtest_df = train_test_split(df, train_size=train_size_ratio, stratify=df['label'])
        val_df, test_df = train_test_split(valtest_df, test_size=test_size_ratio, stratify=valtest_df['label'])

        return dict(
            train_dataset=train_df,
            eval_dataset=val_df,
            test_dataset=test_df,
        )

    train_testvalid = dataset.train_test_split(train_size=train_size_ratio)
    test_valid = train_testvalid['test'].train_test_split(test_size=test_size_ratio)

    return dict(
        train_dataset=train_testvalid['train'],
        eval_dataset=test_valid['train'],
        test_dataset=test_valid['test'],
    )
