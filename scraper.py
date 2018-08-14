import praw
import traceback
import time
import pickle
import tqdm
import random



subreddits = ['Demotivational', 'lolcats', 'supershibe', 'copypasta', 'emojipasta',
              'TrollXChromosomes', 'trollychromosome', 'starterpacks', 'memes',
              'trippinthroughtime', 'dankmemes', 'madlads', 'bidenbro', 'memeeconomy',
              'rarepuppers', 'dankchristianmemes', 'terriblefacebookmemes',
              '2meirl4meirl', 'AdviceAnimals', 'trippinthroughtime', 'BikiniBottomTwitter',
              'wholesomememes', 'Overwatch_Memes', 'SequelMemes', 'bonehurtingjuice',
              'me_irl', 'meirl', 'anime_irl', '2meirl4meirl', 'meow_irl', 'woof_irl',
              'TooMeIrlForMeIrl', 'wallstreetbets', 'polandball', '4chan', 'KenM',
              'Classic4chan', 'greentext']


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
    posts.extend([p for p in subreddit.new(limit=1000)])
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


def get_all_comments(posts):
    texts = []
    random.shuffle(posts)
    for p in tqdm.tqdm(posts):
        for i in p.comments._comments:

            texts.extend(get_comments(i))
        print(len(texts), len(set(texts)))
        with open(path + 'possible_comments.plk', 'wb') as f:
            pickle.dump(list(set(texts)), f)
    return texts



def get_comments(comment_chain):
    comment_text = []
    try:
        for i in comment_chain.replies._comments:
            comment_text.extend(get_comments(i))

        comment_text.extend([comment_chain.body])
    except:
        pass
    return comment_text


if __name__ == '__main__':


    posts = []
    for i in subreddits:
        print(i)
        posts.extend(read_subreddit(i))
        print(len(posts))

    with open(path + 'posts.plk', 'wb') as f:
        pickle.dump(posts, f)
    get_all_comments(posts)



