import pickle
import random
import gensim
import tqdm
import traceback
import multiprocessing


path = r'/home/td/Documents/reddit_bot/'
path_out = r'/home/td/Documents/reddit_bot/comment_texts/'
c_splitter = ' c_splitter '





def get_comments(prev_text, comments):
    try:
        prev_text = prev_text + c_splitter + str(comments.body)
        comment_text = []

        for i in comments.replies._comments:
            comment_text.extend(get_comments(prev_text, i))

        comment_text.extend([prev_text])

        return comment_text
    except:
        return []


def get_all_comments(i_t):
    try:

        count, p = i_t

        post_title = p.title

        print(count, post_title)
        comment_text = []
        for j in p.comments._comments:
            comment_text.extend(get_comments(post_title, j))
        for count2, j in enumerate(comment_text):
            with open(path_out + '{0}_{1}.txt'.format(count, count2), 'w') as f:
                f.write(j)
    except:
        traceback.print_exc()


def extract():
    with open(path + 'posts.plk', 'rb') as f:
        posts = pickle.load(f)
    # random.shuffle(posts)
    # count = 0

    posts_t = [(count, p) for count, p in enumerate(posts)]

    pool = multiprocessing.Pool(processes=10)
    pool.map(get_all_comments, posts_t, chunksize=1)
    pool.close()
    pool.join()




# for i in tqdm.tqdm(posts):
#     try:
#         count += 1
#         post_title = i.title
#         comment_text = []
#         for j in i.comments._comments:
#             comment_text.extend(get_comments(post_title, j))
#         # print(count, len(comment_text))
#
#         for count2, j in enumerate(comment_text):
#             with open(path_out + '{0}_{1}.txt'.format(count, count2), 'w') as f:
#                 f.write(j)
#     except:
#         traceback.print_exc()




