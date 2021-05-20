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

###############################################################################
# global constants
###############################################################################
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]
BG_COLOR = '#FFFFFF'

###############################################################################
# Functions
###############################################################################
def plot_score(df, plot_fn):
    f = (
        p9.ggplot(df, p9.aes(x="emotion_cat", y="score"))
        + p9.geom_boxplot()
        + p9.labs(x="Model", y="EMOTION FEEL Score")
        + p9.theme_538()
        + p9.theme(legend_position="top",
                   legend_direction="horizontal",
                   figure_size=(10,5))
        + p9.theme(plot_background=p9.element_rect(fill=BG_COLOR, color=BG_COLOR, size=1))
    )
    f.save(plot_fn)

def plot_rank(df, plot_fn):
    f = (
        p9.ggplot(df, p9.aes(x="emotion_cat", y="ratio", fill="factor(rank)"))
        + p9.geom_bar(stat="identity")
        + p9.facet_wrap("cluster_labels_6", nrow=2)
        + p9.labs(x="Model", y="Proportion (%)", fill="Rank")
        + p9.theme_538()
        + p9.theme(legend_position="top",
                   legend_direction="horizontal",
                   figure_size=(10, 5))
        + p9.theme(plot_background=p9.element_rect(fill=BG_COLOR, color=BG_COLOR, size=1))
    )
    f.save(plot_fn)

def plot_rank_full(df, plot_fn):
    f = (
        p9.ggplot(df, p9.aes(x="emotion_cat", y="ratio", fill="factor(rank)"))
        + p9.geom_bar(stat="identity")
        + p9.facet_wrap("cluster_labels_6")
        + p9.labs(x="Model", y="Proportion (%)", fill="Rank")
        + p9.theme_538()
        + p9.theme(legend_position="top",
                   legend_direction="horizontal",
                   figure_size=(10,5))
        + p9.theme(plot_background=p9.element_rect(fill=BG_COLOR, color=BG_COLOR, size=1),
                   axis_text_x=p9.element_text(rotation=45, hjust=1))

    )
    f.save(plot_fn)

###############################################################################
# Main function
###############################################################################
def main():
    """Main entry function
    """
    global args

    # Load score file
    df = pd.read_csv(args.score_file)
    df["step"] = np.arange(len(df))

    doc2vec_featutes=['doc2vec_{}'.format(str(i).zfill(3)) for i in range(400)]





    # Reshape dataframe
    df = df.melt(["step",'cluster_labels_6' ,"utterance"]+doc2vec_featutes)
    df.columns = ["step",'cluster_labels_6', "utterance"]+doc2vec_featutes+["emotion", "score"]

    # Reorder models
    cat_list = ['joy','fear','sadness','anger','surprise','disgust']
    categories = pd.Categorical(df["emotion"], categories=cat_list)
    df = df.assign(emotion_cat=categories)
    # Plot the score
    plot_score(df, "emotion_distribution_boule_de_suif.png")

    # Compute ranking
    df["rank"] = df.groupby(["step", "cluster_labels_6", "utterance"])["score"].rank(ascending=False, method="dense")
    
    rank_list = np.arange(len(cat_list))
    rank_cat = pd.Categorical(df["rank"], categories=rank_list)
    df = df.assign(rank=rank_cat)

     # Compute proportion
    df_group = df.groupby(["cluster_labels_6", "emotion_cat"])["rank"].value_counts(normalize=True).reset_index(name='ratio')
    df_group["ratio"] = df_group["ratio"] * 100
    print(df_group.head())

    # Plot proportion per speakers
    plot_rank_full(df_group, "rank_result_full.svg")

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
