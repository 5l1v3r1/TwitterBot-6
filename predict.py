
import json
import random

import numpy as np
from keras.models import load_model


def load_rnn(model_path):

    try:
        model = load_model(model_path)
    except OSError:
        raise SystemExit("Unable to open model '{}'!".format(model_path))

    return model


def load_model_meta(meta_path):

    with open(meta_path, 'r', encoding='utf-8') as input:
        model_meta = json.load(input)

    return model_meta


def random_primer(primer_dict):

    chapter = random.choice(list(primer_dict.keys()))

    line = int(primer_dict[chapter]) + 1
    sub_chapter =  random.randint(line,line+50)

    return "{}:{} ".format(chapter, sub_chapter)


def predict_next_chars(model, model_meta, input_chars, window_size,
                       num_to_predict):

    int_to_chars = model_meta['text_encoder']
    chars_to_int = model_meta['text_decoder']

    num_chars = len(list(int_to_chars.keys()))

    predicted_chars = ''

    for i in range(num_to_predict):

        x_test = np.zeros((1, window_size, num_chars))
        for t, char in enumerate(input_chars):
            x_test[0, t, chars_to_int[char]] = 1.

        test_predict = model.predict(x_test,verbose = 0)[0]

        r = np.argmax(test_predict)  # predict class of each test input
        d = int_to_chars[str(r)]

        # update predicted_chars and input
        predicted_chars += d
        input_chars += d

        if len(input_chars) > window_size:
            input_chars = input_chars[1:]

    return predicted_chars


def model_prediction(model, model_meta, window_size, num_to_predict):

    input_chars = random_primer(model_meta['primer'])

    # use the prediction function
    predict_input = predict_next_chars(model, model_meta, input_chars,
                                       window_size, num_to_predict)

    return predict_input
