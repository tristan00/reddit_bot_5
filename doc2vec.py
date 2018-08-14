#Import all the dependencies
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
from gensim.models.callbacks import CallbackAny2Vec
import nltk
import multiprocessing
from comment_extractor import extract


def train_doc2vec(extract_new_data = False):
    if extract_new_data:
        extract()

    # nltk.download('all', halt_on_error=False)
    file_path = '/home/td/Documents/reddit_bot/comment_texts/'

    docLabels = []
    docLabels = [f for f in listdir(file_path) if f.endswith('.txt')]

    print(len(docLabels))
    data = []
    for doc in docLabels:
        data.append(open(file_path + doc, 'r').read())

    # tokenizer = RegexpTokenizer(r'\w+')
    # tokenizer = nltk.tokenize.w
    stopword_set = set(stopwords.words('english'))

    #This function does all cleaning of data using two objects above

    def nlp_clean(data):

        new_data = []
        for d in data:
            new_str = d.lower()
            dlist = nltk.tokenize.word_tokenize(new_str)
            dlist = [i for i in dlist if i not in stopword_set]
            # list(set(dlist).difference(stopword_set))

            new_data.append(dlist)
        return new_data

    class LabeledLineSentence(object):

        def __init__(self, doc_list, labels_list):

            self.labels_list = labels_list
            self.doc_list = doc_list

        def __iter__(self):

            for idx, doc in enumerate(self.doc_list):
                  yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]])


    class EpochLogger(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 0

        def on_epoch_begin(self, model):
            print("Epoch #{} start".format(self.epoch))

        def on_epoch_end(self, model):
            print("Epoch #{} end".format(self.epoch))
            self.epoch += 1
            model.save('/home/td/Documents/reddit_bot/doc2vec.model')



    data = nlp_clean(data)

    it = LabeledLineSentence(data, docLabels)

    print(len([i for i in it]))

    cb = EpochLogger()

    model = gensim.models.Doc2Vec(vector_size=100, min_count=10, alpha=0.05, min_alpha=0.01, workers=10, dbow_words = 1)

    model.build_vocab(it)
    model.train(it, total_examples=model.corpus_count, epochs=1000, callbacks = (cb,))


    # #training of model
    # for epoch in range(1000):
    #     print('iteration '+str(epoch+1))
    #     model.train(it, total_examples=model.corpus_count, epochs=1)

    #saving the created model

    model.save('/home/td/Documents/reddit_bot/doc2vec.model')
    print('model saved')