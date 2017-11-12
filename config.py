import os


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

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
    dropout = 0.
    batch_size = 64
    lr_method = "adam"
    fd_activation = "tanh"
    lr = 0.001
    lr_decay = 0.9
    lr_divide = 1
    reload = False
    nepochs_no_improv = 3

    # hyperparameters
    hidden_size = 256

    conf_dir = "hid-{}_lr-{}-{}-{}_bs-{}_drop-{}_tremb-{}_nep-{}/".format(hidden_size, lr_method, lr, fd_activation,
                                                                          batch_size, dropout, int(train_embeddings),
                                                                          n_epochs)

    # general config
    output_path = "results/" + conf_dir
    model_path = output_path + "model/"
    log_path = output_path + "logs/"
