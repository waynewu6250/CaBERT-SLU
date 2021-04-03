# CaBERT-SLU
Context-Aware Hierarchical BERT Fushion Network for multiple dialog act detection

This code demonstrates how to train/test on e2e/sgd data.

## Training/Testing

1. To train
    >
        python bert_context.py train

2. To test: select mode in config.py: data/validation
    >
        python bert_context.py test 

3. To visualize:

    bert_model_context.py will return ffscores and store it as a list of size (batch, head, time_step, time_step) tensors. <br>
    The length of list is the number of total attention layers.

## Baselines

1. MIDSF:
    >
        python baseline_midsf.py train

2. ECA:

    Change model to ECA
    >
        python bert_context.py train

## Parse Data (Optional: data is provided in data/)

1. Go to data/
2. Run the following command to create parsed data
    >
        python dialogue_data.py



    
