import praw
import traceback
import time
import pickle
import tqdm
import random
from constants import *
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import shutil

path = r'/home/td/Documents/reddit_bot/'
subreddit_names_to_follow = []



def create_praw_agent():
    reddit_agent = praw.Reddit(client_id=client_id,
                                   client_secret=client_secret,
                                   username=username,
                                   password=password,
                               user_agent='user_agent')
    return reddit_agent


def get_subreddit(bot, subreddit_name):
    subreddit = bot.subreddit(subreddit_name)
    try:
        subreddit.subscribe()
    except:
        traceback.print_exc()

    posts = []
    posts.extend([p for p in subreddit.top('all', limit = 1000)])
    posts.extend([p for p in subreddit.new(limit=1000)])
    filtered_posts = []
    used_posts = []
    for i in posts:
        if i.fullname not in used_posts:
            filtered_posts.append(i)
            used_posts.append(i.fullname)

    # read_files(filtered_posts, subreddit)
    return posts


def read_subreddit(sub_name):
    bot = create_praw_agent()
    return get_subreddit(bot, sub_name)


def generate_subreddit_list():
    # url = 'https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits'
    # s = requests.Session()
    # r = s.get(url)
    with open('sub_list.html', 'r') as f:
        r = f.read()
    soup = BeautifulSoup(r)
    links = soup.find_all('a', {'rel':'nofollow'})
    links = [i['href'] for i in links]
    links = [i.replace('/r/', '') for i in links if '/r/' == i[0:3]]
    print(links)


def get_comments_from_posts(prev_text, comments):
    try:
        prev_text = prev_text + c_splitter + str(comments.body).replace(c_splitter, ' ')
        comment_text = []

        for i in comments.replies._comments:
            comment_text.extend(get_comments_from_posts(prev_text, i))

        comment_text.append({'score':comments.score, 'text':prev_text.replace('|', ' ')})

        return comment_text
    except:
        # traceback.print_exc()
        return []


def read_files(posts, sub):
    random.shuffle(posts)

    texts = []
    for i in tqdm.tqdm(posts):
        post_title = i.title
        for j in i.comments._comments:
            try:
                texts.extend(get_comments_from_posts(post_title, j))
            except:
                traceback.print_exc()

    # texts = [i.replace('|', ' ') for i in texts]
    # texts = [{'count':count, 'text':i} for count, i in enumerate(texts)]
    df = pd.DataFrame.from_dict(texts)
    df.to_csv('{0}/{1}/{2}.csv'.format(path, 'text_dumps', sub), sep = '|', index = False)
    return texts



def scrape_subreddit(sub_name):
    posts = read_subreddit(sub_name)
    random.shuffle(posts)

    texts = []
    for i in tqdm.tqdm(posts):
        post_title = i.title
        for j in i.comments._comments:
            try:
                texts.extend(get_comments_from_posts(post_title, j))
            except:
                traceback.print_exc()

    df = pd.DataFrame.from_dict(texts)
    try:
        shutil.rmtree('{0}/{1}'.format(path, sub_name))
    except:
        pass
    os.makedirs('{0}/{1}'.format(path, sub_name))
    df.to_csv('{0}/{1}/text.csv'.format(path, sub_name), sep = '|', index = False)


def get_new_posts(subreddit_name):
    bot = create_praw_agent()
    posts = []
    subreddit = bot.subreddit(subreddit_name)
    posts.extend([p for p in subreddit.new(limit=50)])
    return posts


if __name__ == '__main__':
    # generate_subreddit_list()
    # subreddits = subreddits[:2]
    posts = []
    import glob

    files = glob.glob('{0}/{1}/{2}.csv'.format(path, 'text_dumps', '*'))
    files = [i.split('/')[-1].split('.')[0] for i in files]
    print(len(subreddits))
    # subreddits = [i for i in subreddits if i not in files]
    print(len(subreddits))

    random.shuffle(subreddits)
    for count, i in enumerate(subreddits):
        try:
            print(count, len(subreddits), i)
            read_subreddit(i)
        except:
            traceback.print_exc()
            time.sleep(5)
    #
    # read_files()
    #
    # with open(path + 'posts.plk', 'wb') as f:
    #     pickle.dump(posts, f)
    # # get_all_comments(posts)




