# CaBERT-SLU
Context-Aware Hierarchical BERT Fushion Network for multiple dialog act detection

## Parse Data

1. Go to data/
2. Run the following command to create parsed data
    >
        python dialogue_data.py

## Training/Testing

1. To train
    >
        python bert_context.py train

2. To test: select mode: data/validation
    >
        python bert_context.py test 

3. To visualize:

bert_model_context.py returns ffscores, store at ffscores list of (b, h, t, t) <br>
length of list is total layers.


## Baselines

1. MIDSF:
    >
        python baseline_midsf.py train

2. ECA:

    Change to model ECA
    >
        python bert_context.py train



    
