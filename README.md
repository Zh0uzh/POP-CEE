# POP-CEE: Position-oriented Prompt-tuning Model for Causal Emotion Entailment

The code for our paper was modified based on the code for RECCON, which is available at https://github.com/declare-lab/RECCON.

### Dependencies

- numpy==1.18.2
- pandas==1.0.1
- scikit-learn==0.23.1
- torch==1.6.0
- transformers==4.0.0
- tokenizers==0.9.4
- tqdm==4.48.0
- simpletransformers==0.50.0

### Usage

1. Run the `data/process_data.py` file to generate the prompt processed data, which is stored in `data/processed_data`.

   `python data/process_data.py`

2. Run the `train_classification.py` file to train the model.

   `python train_classification.py`

3. Run the `eval_classification.py` file to evaluate the model.

   `python eval_classification.py`

