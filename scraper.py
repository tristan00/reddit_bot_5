import praw
import traceback
import time
import pickle



path = r'/home/td/Documents/reddit_bot/'
subreddit_names_to_follow = []


client_id, client_secret, username, password = 'e67IU5dAiH-QRw', '6ma7vRs71fyh_69r9AmC9eSr6sw', 'dirty_cheeser', 'SVUhgJCTZrPBeN2U'

def create_praw_agent():
    reddit_agent = praw.Reddit(client_id=client_id,
                                   client_secret=client_secret,
                                   username=username,
                                   password=password,
                               user_agent='user_agent')
    return reddit_agent


def get_subeddit(bot, subreddit_name, only_allow_squares = True):
    subreddit = bot.subreddit(subreddit_name)
    try:
        subreddit.subscribe()
    except:
        traceback.print_exc()

    posts = []
    posts.extend([p for p in subreddit.top('all', limit = 1000)])
    return posts


def read_subreddit(sub_name):
    bot = create_praw_agent()
    return get_subeddit(bot, sub_name)


subreddits = ['Demotivational', 'lolcats', 'supershibe', 'copypasta', 'emojipasta',
              'TrollXChromosomes', 'trollychromosome', 'starterpacks', 'memes',
              'trippinthroughtime', 'dankmemes', 'madlads', 'bidenbro', 'memeeconomy',
              'rarepuppers', 'dankchristianmemes', 'terriblefacebookmemes', 'me_irl',
              '2meirl4meirl']

posts = []
for i in subreddits:
    print(i)
    posts.extend(read_subreddit(i))
    print(len(posts))

with open(path + 'posts.plk', 'wb') as f:
    pickle.dump(posts, f)

# with open(path + 'posts.plk', 'rb') as f:
#     posts = pickle.load(f)
#
# print(len(posts))