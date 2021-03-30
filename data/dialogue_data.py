import torch as t
from torch.autograd import Variable
import numpy as np
import pandas as pd
import re
import pickle
import h5py
import json
import os
import csv
import spacy
from nltk.tokenize import word_tokenize 
from train_data import Data
import time 

class E2EData(Data):

    def __init__(self, data_path, rawdata_path, intent2id_path, slot2id_path, done=True):

        super(E2EData, self).__init__(data_path, rawdata_path, intent2id_path)
        self.slot2id_path = slot2id_path
        self.train_data, self.intent2id, self.slot2id = self.prepare_dialogue(done)
        self.num_labels = len(self.intent2id)
    
    def get_tags(self, slot_name, string):
        tags = []
        slot_words = word_tokenize(string.lower())
        for i, slot in enumerate(slot_words):
            if i == 0:
                tags.append('B-'+slot_name)
            else:
                tags.append('I-'+slot_name)
        if len(slot_words) > 0:
            return slot_words[0], (tags, ' '.join(slot_words))
        else:
            return None, None
    
    def modify_slots(self, slots):
        slot_dic = {}
        for slot_pair in slots:
            slot_n = slot_pair[0].strip()
            if slot_n != 'other' and slot_n != 'description':
                if slot_pair[1].find('{') == -1:
                    # only one slot value
                    key, value = self.get_tags(slot_n, slot_pair[1])
                    if key:
                        slot_dic[key] = value
                else:
                    # more than one slot value
                    strings = slot_pair[1][1:-1].split('#')
                    for string in strings:
                        key, value = self.get_tags(slot_n, string)
                        if key:
                            slot_dic[key] = value
        return slot_dic
    
    def text_prepare_tag(self, tokens, text_labels):
        """Auxiliary function for parsing tokens.
        @param tokens: raw tokens
        @param text_labels: raw_labels
        """
        tokenized_sentence = []
        labels = []

        # Reparse the labels in parallel with the results after Bert tokenization
        for word, label in zip(tokens, text_labels):

            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            if label.find('B-') != -1:
                labels.extend([label])
                labels.extend(['I-'+label[2:]] * (n_subwords-1))
            else:
                labels.extend([label] * n_subwords)
        
        tokenized_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]']+tokenized_sentence+['[SEP]'])
        labels = ['[PAD]']+labels+['[PAD]']

        return tokenized_sentence, tokenized_ids, labels
    
    def prepare(self, data_path, intent2id, counter, slot2id, scounter):

        print('Parsing file: ', data_path)

        all_data = []
        data = []
        prev_id = '1'

        with open(self.data_path+data_path, 'r') as f:
        
            for i, line in enumerate(f):
                if i == 0:
                    continue
                
                infos = line.split('\t')
                dialogue_id = infos[0]
                message_id = infos[1]
                speaker = infos[3]
                text = infos[4]
                intents = []
                slots = []
                for act in infos[5:]:
                    if act[:act.find('(')] != '':
                        intents.append(act[:act.find('(')])
                    s = re.findall('\((.*)\)', act)
                    if s:
                        slots.append(s[0].split(';'))
                
                ############################### single intent ###############################
                # intents = "@".join(sorted(intents))
                # if intents not in intent2id:
                #     intent2id[intents] = counter
                #     counter += 1
                # intents = intent2id[intents]
                
                ############################### multi intents ###############################
                for intent in intents:
                    if intent not in intent2id:
                        intent2id[intent] = (counter, self.text_prepare(intent, 'Bert')) # counter
                        counter += 1
                intents = [intent2id[intent][0] for intent in intents]
                intents = list(set(intents))

                #################################### slots ###################################
                text = word_tokenize(text.lower())
                if len(slots) == 0:
                    final_tags = ['O']*len(text)
                else:
                    if len(slots) == 1:
                        slots_split = [slot.split('=') for slot in slots[0] if len(slot.split('=')) == 2]
                    else:
                        news = []
                        for slot in slots:
                            news.extend(slot)
                        slots_split = [slot.split('=') for slot in news if len(slot.split('=')) == 2]
                    slot_dic = self.modify_slots(slots_split)
                    final_tags = []
                    cc = 0
                    for i, word in enumerate(text):
                        if i < cc:
                            continue
                        if word in slot_dic and ' '.join(text[i:i+len(slot_dic[word][0])]) == slot_dic[word][1]:
                            final_tags.extend(slot_dic[word][0])
                            cc += len(slot_dic[word][0])
                        else:
                            final_tags.append('O')
                            cc += 1
                
                if data and prev_id != dialogue_id:
                    all_data.append(data)
                    data = []
                    prev_id = dialogue_id
                
                utt, utt_ids, final_tags = self.text_prepare_tag(text, final_tags)

                ############################ slots conver to ids ###################################
                for slot in final_tags:
                    if slot not in slot2id:
                        slot2id[slot] = scounter # counter
                        scounter += 1
                slots_ids = [slot2id[slot] for slot in final_tags]


                data.append((utt_ids, slots_ids, intents))
                # data.append((utt, utt_ids, final_tags, slots_ids, intents))
                # data.append((text, intents, slots))
        
        return all_data, counter, scounter
    
    def prepare_dialogue(self, done):
        """
        train_data:
        
        a list of dialogues
        for each dialogue:
            [(sent1, [label1, label2], [slot1, slot2]), 
             (sent2, [label2], [slot2]),...]
        """

        if done:
            with open(self.rawdata_path, "rb") as f:
                train_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            with open(self.slot2id_path, "rb") as f:
                slot2id = pickle.load(f)
            return train_data, intent2id, slot2id
        
        ptime = time.time()

        # if os.path.exists(self.intent2id_path):
        #     with open(self.intent2id_path, "rb") as f:
        #         intent2id = pickle.load(f)
        #     counter = len(intent2id)
        # else:
        intent2id = {}
        counter = 0
        slot2id = {}
        scounter = 0
        
        all_data = []
        for data_path in os.listdir(self.data_path):
            data, counter, scounter = self.prepare(data_path, intent2id, counter, slot2id, scounter)
            all_data += data
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(all_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(intent2id, f)
        with open(self.slot2id_path, "wb") as f:
            pickle.dump(slot2id, f)
        
        print("Process time: ", time.time()-ptime)
        
        return all_data, intent2id, slot2id


############################################################################


class SGDData(Data):

    def __init__(self, data_path, rawdata_path, intent2id_path, slot2id_path, turn_path, done=True):

        super(SGDData, self).__init__(data_path, rawdata_path, intent2id_path)
        self.slot2id_path = slot2id_path
        self.turn_path = turn_path
        self.train_data, self.intent2id, self.slot2id, self.turn_data_all = self.prepare_dialogue(done)
        self.num_labels = len(self.intent2id)
        self.num_slot_labels = len(self.slot2id)
    
    def build_ids(self, items, item2id, counter):
        for item in items:
            if item not in item2id:
                item2id[item] = (counter, self.text_prepare(item, 'Bert')) # counter
                counter += 1
        items = [item2id[item][0] for item in items]
        return items, item2id, counter
    
    def get_tags(self, slot_name, string):
        tags = []
        slot_words = word_tokenize(string.lower())
        for i, slot in enumerate(slot_words):
            if i == 0:
                tags.append('B-'+slot_name)
            else:
                tags.append('I-'+slot_name)
        if len(slot_words) > 0:
            return slot_words[0], (tags, ' '.join(slot_words))
        else:
            return None, None
    
    def text_prepare_tag(self, tokens, text_labels):
        """Auxiliary function for parsing tokens.
        @param tokens: raw tokens
        @param text_labels: raw_labels
        """
        tokenized_sentence = []
        labels = []

        # Reparse the labels in parallel with the results after Bert tokenization
        for word, label in zip(tokens, text_labels):

            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            if label.find('B-') != -1:
                labels.extend([label])
                labels.extend(['I-'+label[2:]] * (n_subwords-1))
            else:
                labels.extend([label] * n_subwords)
        
        tokenized_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]']+tokenized_sentence+['[SEP]'])
        labels = ['[PAD]']+labels+['[PAD]']

        return tokenized_sentence, tokenized_ids, labels
    
    def prepare_dialogue(self, done):
        """
        train_data:
        
        a list of dialogues (utterance-level)
        for each dialogue:
            [(sent1, [label1, label2], [slot1, slot2]), 
             (sent2, [label2], [slot2]),...]
        
        a list of dialogues (turn-level)
        for each dialogue:
            [(turn1, intents1, requested_slots1, slots1, values1),...
             (turn2, intents2, requested_slots2, slots2, values2),...]
        """

        if done:
            with open(self.rawdata_path, "rb") as f:
                train_data = pickle.load(f)
            with open(self.intent2id_path, "rb") as f:
                intent2id = pickle.load(f)
            with open(self.slot2id_path, "rb") as f:
                slot2id = pickle.load(f)
            with open(self.turn_path, "rb") as f:
                turn_data_all = pickle.load(f)
            return train_data, intent2id, slot2id, turn_data_all
        
        ptime = time.time()

        # if os.path.exists(self.intent2id_path):
        #     with open(self.intent2id_path, "rb") as f:
        #         intent2id = pickle.load(f)
        #     counter = len(intent2id)
        # else:
        intent2id = {}
        counter = 0
        
        aintent2id = {}
        acounter = 0
        request2id = {}
        rcounter = 0
        slot2id = {}
        scounter = 0

        all_data = []
        all_data_turn = []
        services = []

        for file in sorted(os.listdir(self.data_path))[:-1]:
            
            with open(os.path.join(self.data_path, file), 'r') as f:
                print('Parsing file: ', file)
                raw_data = json.load(f)
                for dialogue in raw_data:

                    # if len(dialogue['services']) == 1:
                    #     continue

                    # utterance data
                    data = []

                    # turn data
                    prev_text = 'this is a dummy sentence'
                    prev_data = ('', '', '')
                    data_turn = []

                    for turns in dialogue['turns']:

                        ###################### utterance ##########################
                        intents = []
                        slots = []
                        for action in turns['frames'][0]['actions']:
                            intents.append(action['act'])
                            slots.append((action['slot'], action['values']))
                        
                        intents = list(set(intents))

                        # single intent
                        # intents = "@".join(intents)
                        # if intents not in intent2id:
                        #     intent2id[intents] = counter
                        #     counter += 1
                        # intents = intent2id[intents]
                        
                        ###################### multi intents ######################
                        for intent in intents:
                            if intent not in intent2id:
                                intent2id[intent] = (counter, self.text_prepare(intent, 'Bert')) # counter
                                counter += 1
                        intents = [intent2id[intent][0] for intent in intents]

                        # slot values number
                        if 'slots' in turns['frames'][0]:
                            slot_nums = turns['frames'][0]['slots']
                        else:
                            slot_nums = []

                        ###################### slots ######################
                        utt = turns['utterance']
                        utt_token = word_tokenize(utt.lower())
                        slot_dic = {}
                        if len(slot_nums) == 0:
                            final_tags = ['O']*len(utt_token)
                        else:
                            for slot_dic_example in slot_nums:
                                start = slot_dic_example['start']
                                end = slot_dic_example['exclusive_end']
                                slot_name = slot_dic_example['slot']
                                slot_words = utt[start:end]
                                key, value = self.get_tags(slot_name, slot_words)
                                if key:
                                    slot_dic[key] = value
                            
                            final_tags = []
                            rc = 0
                            for i, word in enumerate(utt_token):
                                if i < rc:
                                    continue
                                if word in slot_dic and ' '.join(utt_token[i:i+len(slot_dic[word][0])]) == slot_dic[word][1]:
                                    final_tags.extend(slot_dic[word][0])
                                    rc += len(slot_dic[word][0])
                                else:
                                    final_tags.append('O')
                                    rc += 1
                        
                        utt, utt_ids, final_tags = self.text_prepare_tag(utt_token, final_tags)

                        ############################ slots conver to ids ###################################
                        for slot in final_tags:
                            if slot not in slot2id:
                                slot2id[slot] = scounter # counter
                                scounter += 1
                        slots_ids = [slot2id[slot] for slot in final_tags]

                        # data.append((self.text_prepare(turns['utterance'], 'Bert'), intents, slots))
                        data.append((utt_ids, slots_ids, intents))
                        # data.append((utt_token, utt_ids, slot_nums, slots_ids, intents))

                        ###################### turn ##########################
                        if 'state' in turns['frames'][0]:
                            slot_values = turns['frames'][0]['state']['slot_values']
                            if not slot_values:
                                s_turn = []
                                v_turn = []
                            else:
                                s_turn, v_turn = zip(*[(k,v[0]) for k, v in slot_values.items()])
                            
                            encoded = self.tokenizer.encode_plus(prev_text, text_pair=turns['utterance'], return_tensors='pt')
                            aintents, aintent2id, acounter = self.build_ids([turns['frames'][0]['state']['active_intent']], aintent2id, acounter)
                            requests, request2id, rcounter = self.build_ids(turns['frames'][0]['state']['requested_slots'], request2id, rcounter)

                            data_turn.append((encoded['input_ids'], aintents, requests, s_turn, v_turn, (prev_data, data[-1])))
                            prev_text = turns['utterance']
                        else:
                            prev_text = turns['utterance']
                            prev_data = data[-1]

                    
                    all_data.append(data)
                    all_data_turn.append(data_turn)
                    services.append(dialogue['services'])
        
        with open(self.rawdata_path, "wb") as f:
            pickle.dump(all_data, f)
        with open(self.intent2id_path, "wb") as f:
            pickle.dump(intent2id, f)
        with open(self.slot2id_path, "wb") as f:
            pickle.dump(slot2id, f)
        with open("sgd_dialogue/services.pkl", "wb") as f:
            pickle.dump(services, f)
        turn_data_all = {'turns': all_data_turn,
                         'aintent2id': aintent2id,
                         'request2id': request2id}
        with open(self.turn_path, "wb") as f:
            pickle.dump(turn_data_all, f)
        
        print("Process time: ", time.time()-ptime)
        
        return all_data, intent2id, slot2id, turn_data_all
    
    
if __name__ == "__main__":

    if not os.path.exists('e2e_dialogue/'):
        os.mkdir('e2e_dialogue/')
    if not os.path.exists('sgd_dialogue/'):
        os.mkdir('sgd_dialogue/')

    # e2e dataset
    data_path = "../raw_datasets/e2e_dialogue/"
    rawdata_path = "e2e_dialogue/dialogue_data_multi_with_slots.pkl"
    intent2id_path = "e2e_dialogue/intent2id_multi_with_tokens.pkl"
    slot2id_path = "e2e_dialogue/slot2id.pkl"
    data = E2EData(data_path, rawdata_path, intent2id_path, slot2id_path, done=False)
    print(data.intent2id)
    print(data.slot2id)
    # for utt, utt_ids, slot, slot_ids, intents in data.train_data[10]:
    #     print(utt)
    #     print(utt_ids)
    #     print(slot)
    #     print(slot_ids)
    #     print(intents)
    #     print('--------------')
    for utt_ids, slot_ids, intents in data.train_data[10]:
        print(utt_ids)
        print(slot_ids)
        print(intents)
        print('--------------')


    # sgd dataset
    data_path = "../raw_datasets/dstc8-schema-guided-dialogue/train"
    rawdata_path = "sgd_dialogue/dialogue_data_multi_with_slots.pkl"
    intent2id_path = "sgd_dialogue/intent2id_multi_with_tokens.pkl"
    slot2id_path = "sgd_dialogue/slot2id.pkl"
    turn_path = "sgd_dialogue/turns.pkl"
    data = SGDData(data_path, rawdata_path, intent2id_path, slot2id_path, turn_path, done=False)
    # print(data.turn_data_all['turns'][0])
    # print(data.train_data[100])
    print(data.intent2id)
    print(data.slot2id)
    # for utt_token, utt_ids, slot_nums, slots_ids, intents in data.train_data[10]:
    #     print(utt_token)
    #     print(utt_ids)
    #     print(slot_nums)
    #     print(slots_ids)
    #     print(intents)
    #     print('--------------')
    for utt_ids, slot_ids, intents in data.train_data[10]:
        print(utt_ids)
        print(slot_ids)
        print(intents)
        print('--------------')