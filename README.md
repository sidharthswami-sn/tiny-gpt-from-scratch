# Tiny GPT from Scratch — Python & PyTorch

A hands-on implementation of a small GPT-style language model using Python and PyTorch, developed to understand the internal workings of Transformer-based Large Language Models (LLMs).

Rather than relying on a pretrained model, this project explores how a language model learns token relationships and generates text through Transformer layers, self-attention, and next-token prediction.

## 1. Project Objective

The objective is to build a foundational understanding of GPT architecture by implementing and training a small autoregressive language model.

The project explores:

- Byte Pair Encoding (BPE) tokenization using SentencePiece
- Token and positional embeddings
- Transformer blocks and multi-head self-attention
- Feed-forward neural networks within Transformer blocks
- Layer normalization
- Forward propagation and backpropagation
- Cross-entropy loss and AdamW optimization
- Autoregressive text generation using Softmax sampling

## 2. Model Architecture

The model processes text using the following stages:

```text
Input Text
    |
SentencePiece BPE Tokenizer
    |
Token IDs
    |
Token Embeddings + Positional Embeddings
    |
Transformer Block 1
    |
Transformer Block 2
    |
Final Layer Normalization
    |
Linear Projection to Vocabulary
    |
Logits
    |
Softmax + Multinomial Sampling
    |
Next Token Prediction
    |
Generated Text
```

The Transformer blocks are implemented in `transformer_blocks.py` and imported into the main model.

## 3. Model Configuration

| Parameter | Value |
|---|---|
| Model architecture | Decoder-only GPT-style Transformer |
| Tokenizer | SentencePiece BPE |
| Vocabulary size | 40 tokens |
| Context window | 6 tokens |
| Embedding dimension | 32 |
| Attention heads | 2 |
| Transformer layers | 2 |
| Learning rate | 0.001 |
| Training iterations | 1,500 |
| Batch size | 16 |
| Optimizer | AdamW |
| Loss function | Cross-entropy |
| Framework | PyTorch |

This is a deliberately small model intended for educational experimentation rather than production language generation.

## 4. Project Files

```text
L-3_tinygpt_adding_tokenizer/
├── tinygpt_tokenizer.py
├── transformer_blocks.py
├── corpus.txt
├── tokenizer.model
├── tokenizer.vocab
├── requirements.txt
├── README.md
└── .gitignore
```

**File descriptions:**

- `tinygpt_tokenizer.py` — Tokenization, TinyGPT architecture, training, and text generation
- `transformer_blocks.py` — Transformer block implementation
- `corpus.txt` — Input text used for training
- `tokenizer.model` — Trained SentencePiece tokenizer
- `tokenizer.vocab` — Tokenizer vocabulary
- `requirements.txt` — Python dependencies

## 5. Tokenization

The project uses SentencePiece with Byte Pair Encoding (BPE) to convert raw text into token IDs.

```python
spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=40,
    model_type="bpe"
)
```

The trained tokenizer encodes text into integer tokens, which become inputs to the language model.

## 6. Training Process

During training:

1. Text is tokenized into integer IDs.
2. Random input sequences of six tokens are selected.
3. The target sequences are shifted forward by one token.
4. The model predicts the next token at each position.
5. Cross-entropy loss measures prediction errors.
6. Backpropagation computes gradients.
7. AdamW updates the model parameters.

The training loop runs for 1,500 iterations and prints the loss every 300 steps.

## 7. Text Generation

After training, the model can generate additional tokens from an initial prompt.

Example:

```python
context = torch.tensor(
    [sp.encode("hello")],
    dtype=torch.long
)

out = model.generate(
    context,
    max_new_tokens=20
)

print(sp.decode(out[0].tolist()))
```

The model generates text autoregressively, predicting and sampling one token at a time.

Because the training corpus and vocabulary are small, generated text may be repetitive, grammatically inconsistent, or semantically limited.

## 8. Running the Project

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/tiny-gpt-from-scratch.git
cd tiny-gpt-from-scratch
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python tinygpt_tokenizer.py
```

Ensure `corpus.txt` and `transformer_blocks.py` are present in the project directory.

## 9. Device Compatibility

The project checks for Apple Metal Performance Shaders (MPS) availability.

```python
torch.backends.mps.is_available()
```

The current code tests MPS availability and performs a sample tensor operation on the selected device. The main training loop currently runs on CPU because the model and training batches are not explicitly moved to MPS.

## 10. Key Learnings

This project helps develop an understanding of:

- How language models transform text into numerical representations
- How token embeddings and positional embeddings work
- How Transformer blocks process token representations
- How language models learn through backpropagation
- How Softmax probabilities support token sampling
- How context-window size influences next-token prediction

## 11. Future Improvements

- Train the model using Apple Silicon MPS acceleration
- Experiment with larger training datasets
- Increase context-window length and vocabulary size
- Visualize training-loss progression
- Add model checkpoint saving and loading
- Experiment with generation temperature and top-k sampling
- Evaluate generated text quality

## Author

**Sidharth Swami**

A hands-on exploration of Generative AI, Transformer architecture, and Large Language Models through Python implementation.

*Disclaimer: This is an educational implementation of a small GPT-style model, not a pretrained or production-ready LLM.*
