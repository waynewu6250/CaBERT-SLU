class Config:

    #################### For BERT fine-tuning ####################
    # control
    datatype = "e2e"
    retrain = False                     # Reuse trained model weights
    test_mode = "data" # "validation", "data"
    data_mode = "multi" #"single"      # single or multi intent in data
    #################################
    
    if datatype == "e2e":
        # Microsoft e2e dialogue dataset
        train_path = "data/e2e_dialogue/dialogue_data.pkl" if data_mode == "single" else "data/e2e_dialogue/dialogue_data_multi_with_slots.pkl"
        test_path = "data/e2e_dialogue/dialogue_data_multi.pkl"
        dic_path = "data/e2e_dialogue/intent2id.pkl" if data_mode == "single" else "data/e2e_dialogue/intent2id_multi.pkl"
        dic_path_with_tokens = "data/e2e_dialogue/intent2id_multi_with_tokens.pkl"
        slot_path = "data/e2e_dialogue/slot2id.pkl"
        pretrain_path = "data/e2e_dialogue/dialogue_data_pretrain.pkl"
    
    elif datatype == "sgd":
        # dstc8-sgd dialogue dataset
        train_path = "data/sgd_dialogue/dialogue_data.pkl" if data_mode == "single" else "data/sgd_dialogue/dialogue_data_multi_with_slots.pkl"
        test_path = "data/sgd_dialogue/dialogue_data_multi.pkl"
        dic_path = "data/sgd_dialogue/intent2id.pkl" if data_mode == "single" else "data/sgd_dialogue/intent2id_multi.pkl"
        dic_path_with_tokens = "data/sgd_dialogue/intent2id_multi_with_tokens.pkl"
        slot_path = "data/sgd_dialogue/slot2id.pkl"
        pretrain_path = "data/sgd_dialogue/dialogue_data_pretrain.pkl"

    model_path = None if not retrain else "checkpoints/best_{}_{}.pth".format(datatype, data_mode)

    maxlen = 60
    batch_size = 4 #CaBERT-SLU: e2e 16/8/4 sgd 4  # multi 128 eca 8 
    epochs = 20
    learning_rate_bert = 2e-5 #1e-3
    learning_rate_classifier = 5e-3

    rnn_hidden = 256


opt = Config()