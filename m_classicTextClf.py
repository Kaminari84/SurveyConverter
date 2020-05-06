import logging
import json
import numpy as np
import random
import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import spacy
import pickle

import os
import os.path
from os import path

#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

class TextClassifier:
    data_file = None
    data_dict = {}
    all_data = None
    all_data_labels = None
    all_labels = None

    text_clf = None
    nlp = None

    use_pos_tags = False
    rem_stop_words = False

    lbl2idx = {}
    idx2lbl = []

    def __init__(self, data_file, model_path, rem_stop_words=False, use_pos_tags=False):
        self.data_file = data_file

        #tokenizer for english
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.vocab["indicate"].is_stop = True
        self.nlp.vocab["extent"].is_stop = True

        with open(self.data_file, 'r') as f:
            self.data_dict = json.load(f)

        self.processData(rem_stop_words, use_pos_tags)

        #Saved model - load it
        if path.exists(model_path):
          self.load_model(model_path)
        #Model does not exist, train one
        else:
          self.train(test_prop=0.1)
          self.save_model(model_path)

        #print(self.data_dict)

    def save_model(self, filepath):
        #save 
        config_dict = {}
        config_dict['use_pos_tags'] = self.use_pos_tags
        config_dict['rem_stop_words'] = self.rem_stop_words
        config_dict['all_data'] = self.all_data
        config_dict['all_data_pos'] = self.all_data_pos
        config_dict['all_labels'] = self.all_labels

        if not path.exists(filepath):
          os.mkdir(filepath)

        #save ML model
        pickle.dump(self.text_clf, open(path.join(filepath, "model.pcl"), 'wb'))

        #save the metadata
        pickle.dump(config_dict, open(path.join(filepath, "metadata.pcl"), 'wb'))

    def load_model(self, filepath):
        model_path = path.join(filepath, "model.pcl")
       	logging.info("---LOADING MODEL: <"+str(filepath)+">---")
        self.text_clf = pickle.load(open(model_path, 'rb'))

        meta_path = path.join(filepath, "metadata.pcl")
        logging.info("---LOADING METADATA: <"+str(meta_path)+">---")
        conf_dict = pickle.load(open(meta_path, 'rb'))

        self.use_pos_tags = conf_dict['use_pos_tags']
        self.rem_stop_words = conf_dict['rem_stop_words']
        self.all_data = conf_dict['all_data']
        self.all_data_pos = conf_dict['all_data_pos']
        self.all_data_labels = conf_dict['all_labels']

        #train accuracy
        #result = loaded_model.score(X_test, Y_test)
        #print(result)
        #train_predicted = self.text_clf.predict(self.all_data)
        #print("-train accuracy:", round(np.mean(train_predicted == self.all_labels),2))

    def processData(self, rem_stop_words=False, use_pos_tags=False):
        self.lbl2idx = {}
        self.idx2lbl = []
        
        self.all_data = []
        self.all_data_pos = []
        self.all_labels = []

        self.use_pos_tags = use_pos_tags
        self.rem_stop_words = rem_stop_words
        
        for lbl in self.data_dict:
            #print("Label:", lbl)
            if lbl not in self.lbl2idx:
                self.lbl2idx[lbl] = len(self.lbl2idx.keys())
                self.idx2lbl.append(lbl)
            
            #removing stop words step with spacy
            #sent_list = self.data_dict[lbl]
            sent_list = []
            pos_list = []
                
            for raw_sent in self.data_dict[lbl]:
                sent_tokens = self.nlp(raw_sent.lower())

                sanitized_sent = ""
                pos_sent = ""
                for token in sent_tokens:
                    pos_sent += token.pos_+" "
                    if self.rem_stop_words and token.is_stop:
                        pass
                    else:
                        sanitized_sent += token.text+" "

                sent_list.append(sanitized_sent.strip())
                pos_list.append(pos_sent.strip())

            #if not rem_stop_words:
            #    print("No rem stop words...")
            #    sent_list = self.data_dict[lbl]

            #print(""+str(lbl)+":",sent_list,", POS:",pos_list)
            self.all_data += sent_list
            self.all_data_pos += pos_list
            self.all_labels += [self.lbl2idx[lbl] for i in range(len(self.data_dict[lbl]))]

        #print("Labels:")
        #print(self.lbl2idx)
        #print(self.idx2lbl)

        #for d, dl in zip(all_data, all_labels):
        #    print("> %s => %s" % (d,self.idx2lbl[dl]))
        
        #self.count_vect = CountVectorizer()
        #X_train_counts = self.count_vect.fit_transform(all_data)
        #print(X_train_counts.shape)

        #print(self.count_vect.vocabulary_.get(u'comfortable'))
        #self.tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        #X_train_tf = self.tf_transformer.transform(X_train_counts)
        #print(X_train_tf.shape)

        #print("Labels:", all_labels)

        #self.clf = MultinomialNB().fit(X_train_tf, all_labels)

    def train(self, test_prop=0.1):
        #pipeline
        self.text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1,2), analyzer="word")),
            ('tfidf', TfidfTransformer(use_idf=True)),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',                         
                            alpha=1e-3, random_state=42,
                            max_iter=5, tol=None)),
            #('clf', MultinomialNB()),
        ])

        #train-test split
        if (test_prop > 0):
            idx_shuffled = [idx for idx in range(len(self.all_data))]
            random.shuffle(idx_shuffled)
            #print("Idx shuffled:", idx_shuffled)
            #set the test_prop % aside for testing
            sep_idx = round(test_prop*len(idx_shuffled))
            if sep_idx == 0: sep_idx = 1
            idx_train = idx_shuffled[sep_idx:] 
            idx_test = idx_shuffled[0:sep_idx]
            print("Total len: %i, sep idx: %i, train len: %i, test len: %i" 
            % (len(idx_shuffled), sep_idx, len(idx_train), len(idx_test)))
            x_train = [self.all_data[i] for i in idx_train]
            if self.use_pos_tags:
              x_train = [self.all_data[i]+" "+self.all_data_pos[i] for i in idx_train]

            y_train = [self.all_labels[i] for i in idx_train]
            #print("Train data:", x_train, ", Labels:", y_train )
            x_test = [self.all_data[i] for i in idx_test]
            if self.use_pos_tags:
              x_test = [self.all_data[i]+" "+self.all_data_pos[i] for i in idx_test]

            y_test = [self.all_labels[i] for i in idx_test]
            #print("Test data:", x_test, ", Labels:", y_test )

            #TRAIN-TEST PHASE - evaluate model performance
            print("---TRAIN-TEST PERFORMANCE EVALUATION---")
            self.text_clf.fit(x_train, y_train)
            #train accuracy
            train_predicted = self.text_clf.predict(x_train)
            print("-Train accuracy:", round(np.mean(train_predicted == y_train),2))
            #test accuracy
            test_predicted = self.text_clf.predict(x_test)
            print("-Test accuracy:", round(np.mean(test_predicted == y_test),2))
            for d, dl in zip(x_test, test_predicted):
                print("> %s => %s" % (d,self.idx2lbl[dl]))

        #FINAL MODEL - retrain on all data
        print("---FINAL MODEL---")
        #train
        self.text_clf.fit(self.all_data, self.all_labels)
        #train accuracy
        train_predicted = self.text_clf.predict(self.all_data)
        print("-Final train accuracy:", round(np.mean(train_predicted == self.all_labels),2))

    def cross_val(self, n_folds=5):
        idx_shuffled = [idx for idx in range(len(self.all_data))]
        random.shuffle(idx_shuffled)

        train_acc = []
        test_acc = []
        
        for fold_n in range(n_folds):
            s_idx = int((fold_n)*np.round(len(idx_shuffled)/n_folds))
            e_idx = int(min(len(idx_shuffled), (fold_n+1)*np.round(len(idx_shuffled)/n_folds)))
            #print("FOLD %i (%i, %i)" %(fold_n, s_idx, e_idx))
  
            test_idx = [idx_shuffled[i] for i in list(range(s_idx,e_idx))]
            train_idx = [idx_shuffled[i] for i in range(len(idx_shuffled)) if idx_shuffled[i] not in test_idx]

            #print("Train idx:", train_idx)
            #print("Test idx:", test_idx)

            x_train = [self.all_data[i] for i in train_idx]
            if self.use_pos_tags:
              x_train = [self.all_data[i]+" "+self.all_data_pos[i] for i in train_idx]
            y_train = [self.all_labels[i] for i in train_idx]

            x_test = [self.all_data[i] for i in test_idx]
            if self.use_pos_tags:
              x_test = [self.all_data[i]+" "+self.all_data_pos[i] for i in test_idx]
            y_test = [self.all_labels[i] for i in test_idx]

            #TRAIN-TEST
            self.text_clf.fit(x_train, y_train)
            #train accuracy
            train_predicted = self.text_clf.predict(x_train)
            #print("-Train accuracy:", round(np.mean(train_predicted == y_train),2))
            #test accuracy
            test_predicted = self.text_clf.predict(x_test)
            #rint("-Test accuracy:", round(np.mean(test_predicted == y_test),2))
            #for d, dl in zip(x_test, test_predicted):
            #    print("> %s => %s" % (d,self.idx2lbl[dl]))

            train_acc.append(np.mean(train_predicted == y_train))
            test_acc.append(np.mean(test_predicted == y_test))

            s_idx = e_idx

        print("AVG",str(n_folds)+"-FOLD TRAIN ACC:", np.round(np.mean(train_acc),2))
        print("AVG",str(n_folds)+"-FOLD TEST ACC:", np.round(np.mean(test_acc),2))
    
    def eval(self, eval_data, eval_labels):
        sent_list = []
        for raw_sent in eval_data:
            sent_tokens = self.nlp(raw_sent.lower())

            sanitized_sent = ""
            pos_sent = ""
            for token in sent_tokens:
                pos_sent += token.pos_+" "
                if self.rem_stop_words and token.is_stop:
                    pass
                else:
                    sanitized_sent += token.text+" "
            if self.use_pos_tags:
              sent_list.append(sanitized_sent.strip()+" "+pos_sent.strip())
            else:
              sent_list.append(sanitized_sent.strip())

        #print("Sentences with POS:", sent_list)
        pred_lbl = self.text_clf.predict(sent_list)

        for doc, lbl_id in zip(eval_data, pred_lbl):
            print("%r => %s" % (doc, self.idx2lbl[lbl_id]))

        eval_lbl_idx = [self.lbl2idx[i] for i in eval_labels]
        eval_acc = np.mean(pred_lbl == eval_lbl_idx)
        print("Evaluation accuracy:", np.round(eval_acc,2))

        return eval_acc
    
    def addNewData(self, new_data, new_labels):

        #pp = pprint.PrettyPrinter(indent=1)
        #pp.pprint(self.data_dict)

        for d,l in zip(new_data, new_labels):
            if l in self.data_dict.keys():
                print('Appending....')
                self.data_dict[l].append(d)

        #pp = pprint.PrettyPrinter(indent=1)
        #pp.pprint(self.data_dict)

        #save to file
        with open(self.data_file.replace('.json','_2.json'), 'w') as f:
            str_json = json.dumps(self.data_dict, indent=4, sort_keys=True)
            print(str_json, file=f)

        new_lbl_idx = [self.lbl2idx[i] for i in new_labels]
        self.text_clf.fit(new_data, new_lbl_idx)

        return self.data_dict

    def classify(self, new_data):
        print("---PREDICTION for %i samples" % (len(new_data)))
        #X_new_counts = self.count_vect.transform(new_data)
        #X_new_tfidf = self.tf_transformer.transform(X_new_counts)
        #predicted_lbl = self.clf.predict(X_new_tfidf)
        sent_list = []
        for raw_sent in new_data:
            sent_tokens = self.nlp(raw_sent.lower())

            sanitized_sent = ""
            pos_sent = ""
            for token in sent_tokens:
                pos_sent += token.pos_+" "
                if self.rem_stop_words and token.is_stop:
                    pass
                else:
                    sanitized_sent += token.text+" "
            if self.use_pos_tags:
              sent_list.append(sanitized_sent.strip()+" "+pos_sent.strip())
            else:
              sent_list.append(sanitized_sent.strip())
        
        #print("Sentences with POS:", sent_list)
        predicted_lbl = self.text_clf.predict(sent_list)

        for doc, lbl_id in zip(new_data, predicted_lbl):
            print("%r => %s" % (doc, self.idx2lbl[lbl_id]))

        return [self.idx2lbl[lbl_id] for lbl_id in predicted_lbl]

if __name__ == "__main__":
    print("***** PREFIX CATEGORY CLASSIFICATION ******")
    tc = TextClassifier("./static/model_data/q_prefix_data.json")
    tc.train(test_prop=0.2, use_pos_tags=False)
    print("---CROSS VAL---")
    tc.cross_val(n_folds=5)
    eval_data = ['What is your gender ?',
                'What is your marital situation at the moment?', 
                'Indicate your gender.',
                'Please specify your current state of mind.',
                'Feeling in trouble over delayed submission.',
                'No interest in continuing your PhD.',
                'Being ti   red of costant moving.',
                'Are you homeless?',
                'I need to know what I am doing?',
                "If you've had such problems, how severe were they?",
                "As long as I can remember handout material for examinations, I do not have to think too much."]
    eval_labels = ['q_verb_prefix', 
                  'q_verb_prefix',
                  'q_non_verb_prefix',
                  'q_non_verb_prefix',
                  'q_add_question',
                  'q_add_question',
                  'q_add_question',
                  'q_prefix_if_you',
                  'q_prefix_if_i',
                  'q_none',
                  'q_prefix_if_i']
    tc.eval(eval_data, eval_labels)
    pc2phrase = {'q_verb_prefix': {"prefix":"Can you tell me", 
                                   "conv": []}, 
                 
                 'q_non_verb_prefix': {"prefix":"Can you", 
                                       "conv":[{"type":"replace", "from":".", "to":"?"}]},
                 
                 'q_add_question': {"prefix":"Have you experienced",
                                    "conv":[{"type":"replace", "from":".", "to":"?"}]},
                 
                 'q_prefix_if_you': {"prefix":"Can you share whether",
                                     "conv":[{"type":"replace", "from":"are you", "to":"you are"}]},
                 
                 'q_prefix_if_i': {"prefix":"Would you say that",
                                   "conv":[
                                     {"type":"replace", "from":"are you", "to":"you are"},
                                     {"type":"replace", "from":"i ", "to":"you "},
                                     {"type":"replace", "from":"am ", "to":"are "},
                                     {"type":"replace", "from":"we ", "to":"you "},
                                     {"type":"replace", "from":"us ", "to":"you "},
                                   ]},
                 
                 'q_none':{"prefix":"",
                           "conv":[]}
                }

    lbl = tc.classify(eval_data)
    print('---Augmented---')
    for l, t in zip(lbl,eval_data):
      if l != "q_none":
        text = t.lower()
        for conv in pc2phrase[l]['conv']:
          if conv['type'] == "replace":
            text = text.replace(conv['from'], conv['to'])
        text = pc2phrase[l]['prefix']+' '+text
      else:
        text = t

      print(text)




    print("***** PREFIX MATCHING ******")

'''
    print("***** DOMAIN CLASSIFICATION ******")
    tc = TextClassifier("./static/model_data/domain_survey_data.json")
    tc.train(test_prop=0.2, rem_stop_words=True)
    print("---CROSS VAL---")
    tc.cross_val(n_folds=5)
    eval_data = ['How mentally and physically successful was your workload ?', 
                'What is your age, gender and occupation ?', 
                'Are you able to follow medical instructions ?',
                'How distressed have you felt over the past week ?',
                'I often need to understand and think about what I am doing ',
                'i often feel happiness and joy when reading this, do you ?']
    eval_labels = ['workload', 
                    'demographics',
                    'health literacy',
                    'stress',
                    'reflection',
                    'other']
    tc.eval(eval_data, eval_labels)
'''


'''
    print("***** QUESTION FRAMING *****")
    tc = TextClassifier("q_framing_data.json")
    tc.train(test_prop=0.2)
    lbl = tc.classify(['how happy do you feel interacting with this ?', 
    "is it stressful and uncomfortable for you to talk about it ?",
    "what is your age ?",
    "how safe do you feel now ?",
    "how scared and insecure do you feel ?",
    "how cheerful, happy and satisfied are you ?"])
    #print("Labels:", lbl)

    eval_data = ['i love to be happy', 
                'how often do you feel angry ?', 
                'when around , i am exhausted',
                'i often feel happiness and joy']
    eval_labels = ['positive_framing', 
                    'negative_framing',
                    'negative_framing',
                    'positive_framing']
'''

'''
    print("---CROSS VAL---")
    tc.cross_val(n_folds=5)
    tc.eval(eval_data, eval_labels)

    print("---ADD new DATA----")
    tc.addNewData(['when i am around , i feel happiness',
                    'i love to be happy',
                    'how often do you feel angry ?',
                    'i often feel happiness and joy',
                    'how often do you feel happy and joy ?'], 
                    ['positive_framing', 
                    'positive_framing',
                    'negative_framing',
                    'positive_framing',
                    'positive_framing'])

    print("---CROSS VAL2---")
    tc.cross_val(n_folds=5)
    tc.eval(eval_data, eval_labels)

    print("***** ANSWER FRAMING *****")
    tc_ans = TextClassifier("ans_types.json")
    tc_ans.train(test_prop=0.1)
    lbl = tc_ans.classify(["agree", "mostly agree", "disagree", "not sure"])
    print("---CROSS VAL---")
    tc_ans.cross_val(n_folds=5)
'''
