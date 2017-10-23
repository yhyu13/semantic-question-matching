# semantic-question-matching

## TODO

### Preprocessing
- Read Quora Dataset or PPDB
- Clean sentences (NLTK tokenizer + lower case). punctuation, symbols, urls, digits, snippets, formulas.. ? stopwords ? non-ASCII characters?
- Porter Stemming?

### Statistical and NLP Features
- Word Embeddings: Glove 300D init_. Words not contained in GloVe?
- Word probabilitie0s? (Who, What, Where, When, How, Why + Does, Can, Should...)
- Word that appear only once? (spelling errors or too specific)

### Baselines
- WMD
- Weighted (TF-IDF, BM25, SIF) sum of word Embedding (+ cosine, euclidean_dist or clf). Remove 1st principal component (PCA).
- Other BOW model (like FastSent)
- Random Forest with ft engineering (lexical word overlap, Jaccard, BM25 score...)

### Training

#### AutoEncoders
- Sequential Denoising Auto Encoders (SDAE): inject noise in input
- Neural Variational Inference: inject noise in latent space
- Encoder modules: bi-LSTMs vs. Self-Attention
- Decoder modules (for AutoEncoders): LSTMs vs. Self-Attention

#### Traditional approach
- Encoder modules: bi-LSTMs vs. Self-Attention
- Representation-based (Siamese Net): Concatenate sentence representations (concat(u,v),uv,|u-v|) + FFN w/ softmax module (0/1)
- Interactive-based: Dense 3D tensor (u,v) + extract features (CNN) + FFN w/ softmax module (0/1)

### Tricks
- Dropout and Batch Normalization
- L1 Regularization


### Testing 
- Cross Validation
