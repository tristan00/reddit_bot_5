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


path = r'/home/td/Documents/reddit_bot/'
path_out = r'/home/td/Documents/reddit_bot/comment_texts/'

c_splitter = ' c_splitter '



tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

lgbm_params =  {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    "learning_rate": 0.001,
    "max_depth": -1,
    'num_leaves':31,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    'bagging_freq': 1,
}

max_iter = 1000000


def nlp_clean(data):

    new_data = []
    new_str = data.lower()
    dlist = tokenizer.tokenize(new_str)
    dlist = [i for i in dlist if i not in stopword_set]
    # list(set(dlist).difference(stopword_set))

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



def get_parent_features(parents_dict):

    l = len(parents_dict['created_utc'])
    c = [i - min(parents_dict['created_utc']) for i in parents_dict['created_utc']]
    max_diff = max(c)
    avg_diff = sum(c)/len(c)

    f = {'p_len':l, 'max_diff':max_diff, 'avg_diff':avg_diff}

    return f


def update_parent_features(c, d):
    d_c = copy.deepcopy(d)

    d_c['created_utc'].append(c.created_utc)
    return d_c

def get_comments(prev_text, comments, parent_dict):
    try:
        prev_text = prev_text + c_splitter + str(comments.body)
        comment_text = []

        n_parent_dict = update_parent_features(comments, parent_dict)
        p_f = get_parent_features(n_parent_dict)


        for i in comments.replies._comments:
            comment_text.extend(get_comments(prev_text, i, n_parent_dict))

        output_dict = {'score':comments.score, 'text':prev_text}
        output_dict.update(p_f)
        comment_text.extend([output_dict])

        return comment_text
    except:
        # traceback.print_exc()
        return []


model = gensim.models.doc2vec.Doc2Vec.load('/home/td/Documents/reddit_bot/doc2vec.model')

with open(path + 'posts.plk', 'rb') as f:
    posts = pickle.load(f)

random.shuffle(posts)
posts = posts[:100]
random.shuffle(posts)
texts = []
for count, i in enumerate(posts):
    post_title = i.title
    comment_text = []
    data_dicts= dict()
    data_dicts = {'created_utc':[i.created_utc]}
    for j in i.comments._comments:
        texts.extend(get_comments(post_title, j, data_dicts))
    print(count, len(texts))


preprocessed_comments = []
for i in tqdm.tqdm(texts):
    preds = model.infer_vector(nlp_clean(i['text']))

    output_dict = {}

    output_dict['p_len'] = i['p_len']
    output_dict['max_diff'] = i['max_diff']
    output_dict['avg_diff'] = i['avg_diff']

    for count, j in enumerate(preds):
        output_dict['doc2vec_{0}'.format(count)] = j
    output_dict['score'] = i['score']
    preprocessed_comments.append(output_dict)

df = pd.DataFrame.from_dict(preprocessed_comments)

scaler = QuantileTransformer()
output = df['score'].values
output = np.reshape(output, (-1, 1))
output = scaler.fit_transform(output)

df['scaled_score'] = output

df = df.drop('score', axis = 1)

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
