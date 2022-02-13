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

from sklearn.model_selection import train_test_split
import re
import string
import pandas as pd
import emoji as emoji
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from datasets import Dataset
from typing import Any, Dict


def _tokenizer_wrapper(batch, tokenizer, tokenizer_max_length):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["tweets"],
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
        data: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """

    :param dataset:
    :param multilingual_task:
    :param data:
    :return:
    """

    if dataset in ["davidson", "waseem"]:
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
            'tweets': new_values,
            'class': classes
        })

    # elif dataset == 'waseem':
    #     pass
    elif dataset == 'multilingual_extension':
        assert(multilingual_task == 'task1' or multilingual_task == 'task2')

        # Get datasets
        url_english_train = 'https://github.com/suman101112/hasoc-fire-2020/blob/main/2020/hasoc_2020_en_train_new_a' \
                            '.xlsx?raw=true '
        url_german_train = 'https://github.com/suman101112/hasoc-fire-2020/blob/main/2020/hasoc_2020_de_train_new_a' \
                           '.xlsx?raw=true '
        url_hindi_train = 'https://github.com/suman101112/hasoc-fire-2020/blob/main/2020/hasoc_2020_hi_train_a.xlsx' \
                          '?raw=true '
        url_english_test = 'https://github.com/suman101112/hasoc-fire-2020/blob/main/2020/english_test_1509.csv?raw' \
                           '=true '
        url_german_test = 'https://github.com/suman101112/hasoc-fire-2020/blob/main/2020/german_test_1509.csv?raw=true'
        url_hindi_test = 'https://github.com/suman101112/hasoc-fire-2020/blob/main/2020/hindi_test_1509.csv?raw=true'

        # Read datasets
        ## English
        data_en = pd.read_excel(url_english_train)
        data_en_test = pd.read_csv(url_english_test)

        ## German
        data_de = pd.read_excel(url_german_train)
        data_de_test = pd.read_csv(url_german_test)

        ## Hindi
        data_hi = pd.read_excel(url_hindi_train)
        data_hi_test = pd.read_csv(url_hindi_test)

        # Set labels
        data_en['language'] = 0
        data_en_test['language'] = 0

        data_de['language'] = 1
        data_de_test['language'] = 1

        data_hi['language'] = 2
        data_hi_test['language'] = 2

        data = copy.deepcopy(data_en)
        data = data.append(data_de, ignore_index=True)
        data = data.append(data_hi, ignore_index=True)
        data_test = copy.deepcopy(data_en_test)
        data_test = data_test.append(data_de_test, ignore_index=True)
        data_test = data_test.append(data_hi_test, ignore_index=True)

        labels = data[['task1', 'task2', 'language']]
        le = LabelEncoder()
        labels['task1'] = le.fit_transform(labels['task1'])
        le = LabelEncoder()
        labels['task2'] = le.fit_transform(labels['task2'])

        labels_test = data_test[['task1', 'task2', 'language']]
        le = LabelEncoder()
        labels_test['task1'] = le.fit_transform(labels_test['task1'])
        le = LabelEncoder()
        labels_test['task2'] = le.fit_transform(labels_test['task2'])

        df = pd.DataFrame.from_dict({
            'tweets': data['text'],
            'class': labels[multilingual_task]
        })

    elif dataset == 'sentiment':
        pass
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

    # Tokenize tweets
    train_ds = Dataset.from_pandas(cleaned_dataset)

    train_ds = train_ds.map(
        lambda x: _tokenizer_wrapper(x, tokenizer, tokenizer_max_length),
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=["tweets", "class"]
    )

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return dict(
        dataset=train_ds
    )


def split_data(dataset: pd.DataFrame,
               train_size_ratio: float,
               test_size_ratio: float
) -> Dict[str, pd.DataFrame]:
    """

    :param dataset:
    :param train_size_ratio:
    :param test_size_ratio:
    :return:
    """

    train_testvalid = dataset.train_test_split(train_size=train_size_ratio)
    test_valid = train_testvalid['test'].train_test_split(test_size=test_size_ratio)

    return dict(
        train_dataset=train_testvalid['train'],
        eval_dataset=test_valid['train'],
        test_dataset=test_valid['test'],
    )
