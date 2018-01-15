import os
import tweepy
import time

from predict import model_prediction, load_rnn, load_model_meta
import config


if __name__ == '__main__':

    message = "first test tweet"
    num_prediction = config.num_prediction
    window_size = config.window_size

    CONSUMER_KEY = os.environ.get('BOT_CONSUMER_KEY')
    CONSUMER_SECRET = os.environ.get('BOT_CONSUMER_SECRET')
    ACCESS_KEY = os.environ.get('BOT_ACCESS_KEY')
    ACCESS_SECRET = os.environ.get('BOT_ACCESS_SECRET')
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = tweepy.API(auth)

    model_meta = load_model_meta(config.model_meta)
    model = load_rnn(config.model_path)

    while True:

        message = model_prediction(model, model_meta, window_size, num_prediction)

        api.update_status(message)

        time.sleep(3600)



