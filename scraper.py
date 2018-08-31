import praw
import traceback
import time
import pickle
import tqdm
import random
from constants import *


path = r'/home/td/Documents/reddit_bot/'
subreddit_names_to_follow = []



def create_praw_agent():
    reddit_agent = praw.Reddit(client_id=client_id,
                                   client_secret=client_secret,
                                   username=username,
                                   password=password,
                               user_agent='user_agent')
    return reddit_agent


def get_subeddit(bot, subreddit_name):
    subreddit = bot.subreddit(subreddit_name)
    try:
        subreddit.subscribe()
    except:
        traceback.print_exc()

    posts = []
    posts.extend([p for p in subreddit.top('all', limit = 10)])
    # posts.extend([p for p in subreddit.new(limit=1000)])
    filtered_posts = []
    used_posts = []
    for i in posts:
        if i.fullname not in used_posts:
            filtered_posts.append(i)
            used_posts.append(i.fullname)
    return filtered_posts


def read_subreddit(sub_name):
    bot = create_praw_agent()
    return get_subeddit(bot, sub_name)


def get_comments(comment_chain):
    comment_text = []
    try:
        for i in comment_chain.replies._comments:
            comment_text.extend(get_comments(i))

        comment_text.extend([comment_chain.body])
    except:
        pass
    return comment_text


def get_all_comments(posts):
    texts = []
    random.shuffle(posts)
    for c1, p in enumerate(posts):
        for c2, i in enumerate(p.comments._comments):
            with open(path + 'chains/{0}_{1}.plk'.format(c1, c2), 'wb') as f:
                pickle.dump(get_comments(i), f)

    return texts


if __name__ == '__main__':


    posts = []
    for i in subreddits:
        print(i)
        posts.extend(read_subreddit(i))
        print(len(posts))

    with open(path + 'posts.plk', 'wb') as f:
        pickle.dump(posts, f)
    # get_all_comments(posts)



