import praw


client_id, client_secret, username, password = 'e67IU5dAiH-QRw', '6ma7vRs71fyh_69r9AmC9eSr6sw', 'dirty_cheeser', 'SVUhgJCTZrPBeN2U'


def get_praw_agent():
    reddit_agent = praw.Reddit(client_id=client_id,
                                   client_secret=client_secret,
                                   username=username,
                                   password=password,
                               user_agent='user_agent')
    return reddit_agent


def get_recent_comment_chains():
    pass


def get_possible_comments():
    pass


def get_highest_scoring_comment(comment_chains, possible_comments):
    pass


def main():
    bot = get_praw_agent()
    comment_chains = get_recent_comment_chains()
    possible_comments = get_possible_comments()
    get_highest_scoring_comment(comment_chains, possible_comments)



