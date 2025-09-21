# BBC News Text Classification with BiLSTM

A deep learning project that classifies BBC news articles into **business, entertainment, politics, sport, and tech** using a BiLSTM model and GloVe embeddings.

---

## Dataset

- **Training set:** 1,490 articles (`BBC News Train.csv`)
- **Test set:** 735 articles (`BBC News Test.csv`)
- **Category distribution:**

| Category      | Count |
| ------------- | ----- |
| sport         | 346   |
| business      | 336   |
| politics      | 274   |
| entertainment | 273   |
| tech          | 261   |

---

## Preprocessing

1. Clean text: remove HTML, links, special characters, stopwords
2. Apply stemming (Porter Stemmer)
3. Encode labels with `LabelEncoder`
4. Tokenize and pad sequences
5. Load **GloVe embeddings** and create embedding matrix

---

## Model Architecture

| Layer                  | Output Shape | Notes                           |
| ---------------------- | ------------ | ------------------------------- |
| Input                  | (150,)       | Padded sequences                |
| Embedding              | (150,100)    | GloVe embeddings, non-trainable |
| BiLSTM                 | (150,256)    | 128 units, bidirectional        |
| GlobalAveragePooling1D | (256,)       | -                               |
| Dense                  | (64,)        | ReLU activation                 |
| Output                 | (5,)         | Softmax for 5 classes           |

**Training:** 10 epochs, 20% validation split

**Performance on training set:**

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 95.77% |
| F1 Score | 95.78% |

---

## Dependencies

- Python 3.x
- numpy, pandas, re, nltk
- tensorflow / keras
- scikit-learn

---

## License

MIT License
