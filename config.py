import os
import pandas as pd


def split_train_test(filename, output_train, output_test, test_prop, sep=','):
    df = pd.read_csv(filename, sep=sep)
    test_size = int(len(df) * test_prop)
    df[:-test_size].to_csv(output_train, index_label=False)
    df[-test_size:].to_csv(output_test, index_label=False)


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    # general config
    output_path = "results/"
    models_path = output_path + "models/"
    log_path = output_path + "logs/"

    # embeddings
    we_dim = 300
    glove_filename = "word_embeddings/glove.6B/glove.6B.{}d_w2vformat.txt".format(we_dim)

    # data
    train_filename = "data/train.csv"
    dev_filename = "data/dev.csv"
    test_filename = "data/test.csv"
    train_save = "data/train.pkl"
    dev_save = "data/dev.pkl"
    test_save = "data/test.pkl"

    # vocab
    # TODO saving and quick reloading of dicts and formatted embeddings ???

    # training
    train_embeddings = False
    n_epochs = 10
    dropout = 0
    batch_size = 16
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    reload = False
    nepochs_no_improv = 3

    # hyperparameters
    hidden_size = 1024


if __name__ == "__main__":
    split_train_test("data/quora_duplicate_questions.tsv", "data/train.csv", "data/test.csv", 0.2, sep="\t")
