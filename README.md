# JOINT INTENT DETECTION SYSTEM


## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)

## Dependencies

- python>=3.6
- torch==1.6.0
- transformers==3.0.2
- seqeval==0.0.12
- pytorch-crf==0.7.2

## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Results

- Run 5 ~ 10 epochs (Record the best result)
- Only test with `uncased` model
- ALBERT xxlarge sometimes can't converge well for slot prediction.

|           |                  | Intent acc (%) | Slot F1 (%) |
| --------- | ---------------- | -------------- | ----------- |
| **Snips** | BERT             | **99.14**      | 96.90       |
|           | BERT + CRF       | 98.57          | **97.24**   |

| **ATIS**  | BERT             | 97.87          | 95.59       |
|           | BERT + CRF       | **97.98**      | 95.93       | 



