import pickle
import random
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.callbacks import CallbackAny2Vec
import traceback
import lightgbm as lgb
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
import copy
import operator
import nltk
from collections import Counter
import gc
from multiprocessing import Process, Queue, Pool
from doc2vec import train_doc2vec

path = r'/home/td/Documents/reddit_bot/'
path_out = r'/home/td/Documents/reddit_bot/comment_texts/'

c_splitter = ' c_splitter '



# tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

lgbm_params =  {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    "learning_rate": 0.001,
    "max_depth": -1,
    'num_leaves':255,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    'bagging_freq': 1,
}

max_iter = 1000000


def nlp_clean(data):

    new_data = []
    new_str = data.lower()
    dlist = nltk.tokenize.word_tokenize(new_str)
    dlist = [i for i in dlist if i not in stopword_set]

    return dlist

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1
        model.save('/home/td/Documents/reddit_bot/doc2vec.model')




def get_pos_perc(text):
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)

    counts = Counter(tag for word,tag in tags)


    output_dict = dict()

    for i in counts:
        output_dict['pos_feature_{0}'.format(i)] = counts[i] / max(max(counts.values()), 1)

    return output_dict


def get_parent_features(parents_dict, parent_text):

    l = len(parents_dict['created_utc'])
    c = [i - min(parents_dict['created_utc']) for i in parents_dict['created_utc']]
    max_diff = max(c)
    avg_diff = sum(c)/len(c)

    f = {'p_len':l, 'max_diff':max_diff, 'avg_diff':avg_diff}

    pos_features = get_pos_perc(parent_text)
    pos_keys = [i for i in pos_features.keys()]

    for i in pos_keys:
        pos_features['parent_feature_{0}'.format(i)] = pos_features[i]

    f['parent_len'] = len(nltk.word_tokenize(parent_text.lower()))
    f.update(pos_features)

    return f



def get_comment_features(text):
    pos_features = get_pos_perc(text)

    pos_keys = [i for i in pos_features.keys()]

    for i in pos_keys:
        pos_features['comment_feature_{0}'.format(i)] = pos_features[i]
    pos_features['comment_len'] = len(nltk.word_tokenize(text.lower()))
    return pos_features


def update_parent_features(c, d):
    d_c = copy.deepcopy(d)

    d_c['created_utc'].append(c.created_utc)
    return d_c


def get_comments(prev_text, comments, parent_dict):
    try:
        prev_text = prev_text + c_splitter + str(comments.body)
        c_f = get_comment_features(str(comments.body))
        comment_text = []

        n_parent_dict = update_parent_features(comments, parent_dict)
        p_f = get_parent_features(n_parent_dict, prev_text)


        for i in comments.replies._comments:
            comment_text.extend(get_comments(prev_text, i, n_parent_dict))

        output_dict = {'score':comments.score, 'text':prev_text}
        output_dict.update(p_f)
        output_dict.update(c_f)
        comment_text.extend([output_dict])

        return comment_text
    except:
        # traceback.print_exc()
        return []


def run_model(q1, q2):
    model = gensim.models.doc2vec.Doc2Vec.load('/home/td/Documents/reddit_bot/doc2vec.model')
    while True:
        next_item = q1.get()

        if not next_item:
            break

        preds = model.infer_vector(nlp_clean(next_item['text']))

        output_dict = {i: j for i, j in next_item.items() if i != 'text'}

        for count, j in enumerate(preds):
            output_dict['doc2vec_{0}'.format(count)] = j
        output_dict['score'] = next_item['score']

        q2.put(output_dict)


def process_input(i):
    texts = []
    post_title = i.title
    data_dicts = {'created_utc': [i.created_utc]}
    for j in i.comments._comments:
        try:
            texts= get_comments(post_title, j, data_dicts)
        except:
            traceback.print_exc()
    print(len(texts))
    return texts


def main():
    train_doc2vec()
    # model = gensim.models.doc2vec.Doc2Vec.load('/home/td/Documents/reddit_bot/doc2vec.model')

    with open(path + 'posts.plk', 'rb') as f:
        posts = pickle.load(f)

    random.shuffle(posts)
    posts = random.sample(posts, 10000)
    gc.collect()
    texts = []

    # pool = Pool(processes=10)
    # texts = pool.map(process_input, posts, chunksize=1)
    # pool.close()
    # pool.join()

    # texts = sum(texts, [])

    texts = []
    for i in tqdm.tqdm(posts):
        post_title = i.title
        data_dicts = {'created_utc': [i.created_utc]}
        for j in i.comments._comments:
            try:
                texts.extend(get_comments(post_title, j, data_dicts))
            except:
                traceback.print_exc()
    #
    # for i in tqdm.tqdm(posts):
    #     post_title = i.title
    #     data_dicts = {'created_utc':[i.created_utc]}
    #     for j in i.comments._comments:
    #         try:
    #             texts.extend(get_comments(post_title, j, data_dicts))
    #         except:
    #             traceback.print_exc()

    q1 = Queue(maxsize=25)
    q2 = Queue()

    num_of_proccesses = 5

    ps = [Process(target=run_model, args=(q1,q2,)) for _ in range(num_of_proccesses)]
    [p.start() for p in ps]

    for i in tqdm.tqdm(texts):
        q1.put(i)
    for _ in ps:
        q1.put(None)

    preprocessed_comments = []
    while len(preprocessed_comments) < len(texts):
        preprocessed_comments.append(q2.get())

    #
    # for i in tqdm.tqdm(texts):
    #     preds = model.infer_vector(nlp_clean(i['text']))
    #
    #     output_dict = {k:j for k, j in i.items() if k != 'text'}
    #
    #     for count, j in enumerate(preds):
    #         output_dict['doc2vec_{0}'.format(count)] = j
    #     output_dict['score'] = i['score']
    #     preprocessed_comments.append(output_dict)

    df = pd.DataFrame.from_dict(preprocessed_comments)
    df = df.fillna(0)

    scaler = QuantileTransformer()
    output = df['score'].values
    output = np.reshape(output, (-1, 1))
    output = scaler.fit_transform(output)

    df['scaled_score'] = output

    df = df.drop('score', axis = 1)

    df.to_csv(path + 'features.csv', index=False)

    x = df.drop('scaled_score', axis = 1)
    y = df['scaled_score']

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.1)
    print(df.shape)
    lgtrain = lgb.Dataset(x_train, y_train)
    lgvalid = lgb.Dataset(x_val, y_val)

    model = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=max_iter,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train', 'valid'],
        early_stopping_rounds=1000,
        verbose_eval=100
    )
    model.save_model('/home/td/Documents/reddit_bot/lgbmodel', num_iteration=model.best_iteration)

    fi = model.feature_importance(iteration=model.best_iteration, importance_type='gain')
    fi_dicts = [(i, j) for i, j in zip(x.columns, fi)]
    fi_dicts.sort(key=operator.itemgetter(1), reverse=True)
    print(fi_dicts)


if __name__ == '__main__':
    main()
