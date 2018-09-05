from scraper import scrape_subreddit, get_new_posts
from comment_generator import make_sub_model, min_len, make_sub_prediction
from constants import *
import praw
import traceback
import random
import re


def get_comment_dicts_from_posts(prev_text, comments):
    try:
        prev_text = prev_text + c_splitter + str(comments.body).replace(c_splitter, ' ')
        comment_text = []

        for i in comments.replies._comments:
            comment_text.extend(get_comment_dicts_from_posts(prev_text, i))

        comment_text.append({'comment':comments, 'text':prev_text.replace('|', ' ') + c_splitter})

        return comment_text
    except:
        # traceback.print_exc()
        return []



def get_comment_dicts(posts):
    texts = []
    for p in posts:
        post_title = p.title

        for j in p.comments._comments:
            try:
                texts.extend(get_comment_dicts_from_posts(post_title, j))
            except:
                traceback.print_exc()
    return texts


def run_subreddit(sub_name, retrain = False, rescrape = False):
    if rescrape:
        scrape_subreddit(sub_name)

    if retrain:
        make_sub_model(sub_name)

    posts = get_new_posts(sub_name)
    comment_dicts = get_comment_dicts(posts)
    comment_dicts = [i for i in comment_dicts if len(i['text']) > min_len and 'bot' not in i]

    if len(comment_dicts) > 0:
        chosen_comment = random.choice(comment_dicts)
    else:
        print('no valid comments to respond to')
        return None

    pred_text = make_sub_prediction(sub_name, chosen_comment['text'])
    reply_text = pred_text.split(c_splitter)[0]
    print(chosen_comment['text'])
    print(pred_text)
    print(reply_text)

    chosen_comment['comment'].reply(reply_text)



def run_sub(sub, retrain=True, rescrape=True):
    run_subreddit(sub, retrain=retrain, rescrape=rescrape)
    time.sleep(600)


import time

import string
print(list(string.printable).index(' '))

run_sub('trees', retrain=True, rescrape=False)
# run_sub('highdeas')
# run_sub('StonerPhilosophy')
# run_sub('meirl')
# run_sub('madlads')
# run_sub('WTF')
# run_sub('youdontsurf')
# run_sub('hailcorporate')
# run_sub('dankmemes')
# run_sub('Demotivational')
# run_sub('BikiniBottomTwitter')
# run_sub('memes')
# run_sub('ProgrammerHumor')
# run_sub('wellthatsucks')




