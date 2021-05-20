
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR

    Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION

LICENSE
    This script is in the public domain, free from copyrights or restrictions.
    Created: 12 December 2019
"""

# System/default
import sys
import os

# Arguments
import argparse

# Messaging/logging
import traceback
import time
import logging

# Data/plot
import numpy as np
import pandas as pd
import plotnine as p9

# Warning
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("error", RuntimeWarning)


# text parsing
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

###############################################################################
# global constants
###############################################################################
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]
BG_COLOR = '#FFFFFF'


def text_emotion(df, column):
    '''
    Takes a DataFrame and a specified column of text and adds 10 columns to the
    DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the text in that emotions
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns
    '''

    new_df = df.copy()

    filepath = ('./data/'
        '/FEEL-v1.csv')
    emolex_df = pd.read_csv(filepath,
                            names=["id","word","polarity","emotion","score","emotion_cat"]
                            sep='\t')
    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='score').reset_index()
    emotions = emolex_words.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    # stemmer = SnowballStemmer("french")

    
    # book = ''
    # chapter = ''
    
    # with tqdm(total=len(list(new_df.iterrows()))) as pbar:
    #     for i, row in new_df.iterrows():
    #         pbar.update(1)
    #         if row['book'] != book:
    #             print(row['book'])
    #             book = row['book']
    #         if row['chapter_title'] != chapter:
    #             print('   ', row['chapter_title'])
    #             chapter = row['chapter_title']
    #             chap = row['chapter_title']
    #         document = word_tokenize(new_df.loc[i][column])
    #         for word in document:
    #             word = stemmer.stem(word.lower())
    #             emo_score = emolex_words[emolex_words.word == word]
    #             if not emo_score.empty:
    #                 for emotion in list(emotions):
    #                     emo_df.at[i, emotion] += emo_score[emotion]

    # new_df = pd.concat([new_df, emo_df], axis=1)

    # return new_df


def main():
    """Main entry function
    """
    global args

    # Load score file
    df = pd.read_csv(args.score_file)
    print(df.head())
    # text_emotion(df, column):


###############################################################################
#  Envelopping
###############################################################################
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="")

        # Add options
        parser.add_argument("-l", "--log_file", default=None,
                            help="Logger file")
        parser.add_argument("-v", "--verbosity", action="count", default=0,
                            help="increase output verbosity")

        # Add arguments
        parser.add_argument("score_file")
        # Example : parser.add_argument("echo", help="description")
        # TODO

        # Parsing arguments
        args = parser.parse_args()

        # create logger and formatter
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Verbose level => logging level
        log_level = args.verbosity
        if (args.verbosity >= len(LEVEL)):
            log_level = len(LEVEL) - 1
            logger.setLevel(log_level)
            logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)
        else:
            logger.setLevel(LEVEL[log_level])

        # create console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # create file handler
        if args.log_file is not None:
            fh = logging.FileHandler(args.log_file)
            logger.addHandler(fh)

        # Debug time
        start_time = time.time()
        logger.info("start time = " + time.asctime())

        # Running main function <=> run application
        main()

        # Debug time
        logging.info("end time = " + time.asctime())
        logging.info('TOTAL TIME IN MINUTES: %02.2f' %
                     ((time.time() - start_time) / 60.0))

        # Exit program
        sys.exit(0)
    except KeyboardInterrupt as e:  # Ctrl-C
        raise e
    except SystemExit:  # sys.exit()
        pass
    except Exception as e:
        logging.error('ERROR, UNEXPECTED EXCEPTION')
        logging.error(str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(-1)



