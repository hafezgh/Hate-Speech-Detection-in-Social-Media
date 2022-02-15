# Hate-Speech-Detection-in-Social-Media

In  this  project,  we  implement and replicate the experiments in 
first work that proposed BERT fine-tuning strategies for hate 
speech  detection. Mozafari  et al. introduced four fine-tuning 
strategies based on fully-connected neural networks, bidirectional LSTM,
and  convolutional neural networks (CNN) and achieved 
state-of-the-art performance on two hate-speech datasets. After the 
replication these experiments, we also propose two extensions. The 
first extension is applying the proposed fine-tuning strategies to 
two other text classification datasets, the first one being an IMDB 
review dataset for sentiment classification, and the second one a 
dataset for determining eligibility of cancer patients for clinical 
trials. The second extension is adapting the current models to be 
compatible with multi-lingual hate speech detection.

A Kedro implementation of this project can be found in folder "Iris_kedro".
It is possible to run the experiments, firstly by installing Kedro and run
the Kedro framework with the desire parameters. A Collab notebook implementation
of how to use kedro can be found in the *KedroTest.ipynb* notebook.

For installing Kedro use the following code: 
```
! pip install kedro
```

To run the experiments, go to the Iris_kedro/get-started folder and
run the command:
```
! kedro run
```

In order to change the experiment parameters run:
```
! kedro run  --params BERT_model:cnn,training_type:transformers,device:cuda
```

- BERT_model: model to use (cnn, rnn, deep_fc, shallow_fc).
- training_type: implementation to use (classic: paper replica, transformers: transformers library adaptation)

Additional modifiable parameters can be found in the
*Iris_kedro/get-started/conf/base/parameters.yml* file.

This implementation consists of two pipelines: data engineering (de) and data science (ds).
The data engineering pipeline manages the pre-processing steps of the datasets, while the data
science one focuses on the implementation of the models. It is possible to run only one
pipeline with the command:
```
! kedro run --pipeline de
```

Kedro-viz provides a visual and interactive representation of the pipelines. To do this,
install the kedro-viz package and run the command:
```
pip install kedro-viz
kedro viz
```

Further uses of the Kedro framework can be found in https://kedro.readthedocs.io/en/stable/index.html