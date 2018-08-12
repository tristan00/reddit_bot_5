import pickle
import random
import gensim


path = r'/home/td/Documents/reddit_bot/'
path_out = r'/home/td/Documents/reddit_bot/comment_texts/'
c_splitter = ' c_splitter '

with open(path + 'posts.plk', 'rb') as f:
    posts = pickle.load(f)


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



random.shuffle(posts)
for count, i in enumerate(posts):
    post_title = i.title
    comment_text = []
    for j in i.comments._comments:
        comment_text.extend(get_comments(post_title, j))
    print(count, len(comment_text))

    for count2, j in enumerate(comment_text):
        with open(path_out + '{0}_{1}.txt'.format(count, count2), 'w') as f:
            f.write(j)



