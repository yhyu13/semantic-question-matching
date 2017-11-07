# semantic-question-matching

## TODO

### Preprocessing
- ~~Load Quora Dataset.~~
- Load PPDB paraphrase dataset.
- ~~Tokenizer (NLTK) + lower case.~~
- ~~Punctuation split ("good/bad" to ["good","/","bad"]).~~
- Clean symbols, non-ASCII characters, urls, digits, snippets, formulas...
- Spelling errors ?
- Min count / Max frequency (dictionary).
- ~~Load Word Embeddings: Glove 300D init_ (pretrained on Wikipedia & GigaWord 5).~~

### Baselines
- ~~WMD: Unseparable.~~ (starting_kit.py)
- ~~pLSI + MLP: 78% test accuracy.~~ (naive.py)
- Weighted (TF-IDF, BM25, SIF) sum of word Embedding (+ cosine, norm or clf). Remove 1st principal component (PCA).
- Other BOW model (like FastSent).
- Random Forest with ft engineering (lexical word overlap, Jaccard, BM25 score...).

### Training

#### Sequential Variational AutoEncoders
- Encoder module: bi-LSTMs vs. Self-Attention.
- ~~Decoder module: LSTMs.~~
- ~~AutoEncoder objective: -cross_entropy+kld.~~
- ~~AutoDecoder objective: Duplicate case Ok~~/ Not duplicate cases under progress.
- Results: Under progress for sent length <= 12.

#### Siamese/Interactive Network
- Encoder modules: bi-LSTMs vs. Self-Attention
- Representation-based (Siamese Net): Concatenate sentence representations (concat(u,v),uv,|u-v|) + FFN w/ softmax module (0/1)
- Interactive-based: Dense 3D tensor (u,v) + extract features (CNN) + FFN w/ softmax module (0/1)

### Tricks
- Dropout and Batch Normalization
- L1 Regularization
- Inject noise in input space (corrupt input // SDAE)
- Hyperparameter grid search


### Testing 
- Cross Validation
