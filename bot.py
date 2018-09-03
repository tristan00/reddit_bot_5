from scraper import scrape_subreddit, get_new_posts
from comment_generator import make_sub_model, min_len, make_sub_prediction
from constants import *
import praw
import traceback
import random
import re


def get_comment_dicts_from_posts(prev_text, comments):
    try:
        prev_text = prev_text + c_splitter + str(comments.body)
        comment_text = []

        for i in comments.replies._comments:
            comment_text.extend(get_comment_dicts_from_posts(prev_text, i))

        comment_text.append({'comment':comments, 'text':prev_text.replace('|', ' ')})

        return comment_text
    except:
        # traceback.print_exc()
        return []



def get_comment_dicts(p):
    post_title = p.title
    texts = []
    for j in i.comments._comments:
        try:
            texts.extend(get_comment_dicts_from_posts(post_title, j))
        except:
            traceback.print_exc()
    return texts


def run_subreddit(sub_name, retrain = False):
    if retrain:
        scrape_subreddit(sub_name)
        make_sub_model(sub_name)

    posts = get_new_posts(sub_name)
    comment_dicts = get_comment_dicts(posts)
    comment_dicts = [i for i in comment_dicts if len(i) > min_len and 'bot' not in i]

    if len(comment_dicts) > 0:
        chosen_comment = random.choice(comment_dicts)
    else:
        print('no valid comments to respond to')
        return None

    pred_text = make_sub_prediction(sub_name, chosen_comment['text'])
    reply_text = pred_text.split(c_splitter)[0]
    chosen_comment['comment'].reply(reply_text)




run_subreddit('dankmemes', retrain=True)










