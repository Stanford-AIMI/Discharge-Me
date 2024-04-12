# Scoring Scripts

This folder contains the scoring scripts for the ["Discharge Me!"](https://stanford-aimi.github.io/discharge-me/) task. These scripts will used to evaluate generated discharge summary sections.

## Setup

To run the scoring scripts, you need to install the following dependencies:

- numpy
- six
- bert_score
- rouge_score
- torch
- spacy

Additionally, you will need to install AlignScore by following installation instructions listed on the AlignScore repo: https://github.com/yuh-zha/AlignScore.

To install MEDCON:
```
wget https://storage.googleapis.com/vilmedic_dataset/packages/medcon/UMLSScorer.zip
unzip UMLSScorer.zip
pip install quickumls
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
```

You will have to specify the location of ```quickumls_fp``` (which is in UMLSScorer after unzipped) when initializing ```UMLSScorer```. It is currently set to ```/home/quickumls```.

Finally, ensure to modify the data paths appropriately in ```scoring.py```:
```
########################################################
reference_dir = os.path.join("/app/input/", "ref")
generated_dir = os.path.join("/app/input/", "res")
score_dir = "/app/output/"
########################################################
```

You can execute the script by running:
```
python scoring.py
```

Please reach out to xujustin@stanford.edu if you have any trouble getting the evaluation scripts to work.
