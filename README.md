# PyTorch Transformer Implementation

A from-scratch implementation of the Transformer architecture as described in *"Attention Is All You Need"* (Vaswani et al., 2017). This project is designed to serve as an educational resource for understanding transformer models in deep learning.

---

## Features

- **Complete Transformer Architecture**: Encoder-decoder structure with multi-head self-attention.  
- **Multi-Head Self-Attention Mechanism**: Implements scaled dot-product attention with multiple attention heads.  
- **Positional Encoding**: Uses learned embeddings for positional information.  
- **Sequence Masking**: Handles padding and look-ahead masking for source and target sequences.  
- **Modular Design**: Reusable components including `SelfAttention`, `TransformerBlock`, `Encoder`, and `Decoder`.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/pytorch-transformer.git
   cd pytorch-transformer
   ```

2. **Install Requirements**:
   ```bash
   pip install torch numpy
   ```

---

## Usage

### Example Implementation
```python
import torch
from transformers import Transformer

# Initialize the model
model = Transformer(
    src_vocab_size=10000,
    trg_vocab_size=10000,
    src_pad_idx=0,
    trg_pad_idx=0,
    embed_size=256,
    num_layers=6,
    forward_expansion=4,
    heads=8,
    dropout=0.1,
    device="cuda"
).to("cuda")

# Sample input tensors (source and target)
src = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to("cuda")
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to("cuda")

# Forward pass
output = model(src, trg[:, :-1])
print(output.shape)  # Expected output shape: (batch_size, seq_len, trg_vocab_size)
```

---

## Key Components

- **`SelfAttention`**: Implements scaled dot-product attention with multiple heads.
- **`TransformerBlock`**: Combines self-attention and feed-forward networks with residual connections and layer normalization.
- **`Encoder`**: A stack of transformer blocks with positional encoding.
- **`Decoder`**: Similar to the encoder but with an additional masked attention layer to handle target sequence prediction.

---

## Code Structure

```plaintext
.
├── transformers.py         # Main implementation file
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation (this file)
```

---

## Results

The current implementation demonstrates the following functionalities:
- **Variable-Length Sequence Processing**: Handles input sequences of varying lengths.
- **Padding Mask Support**: Appropriately processes padded sequences.
- **Target Sequence Prediction**: Produces output logits for target token prediction.

---

## Future Improvements

Contributions are welcome! If you'd like to improve this implementation, please open an issue to discuss potential enhancements. Some suggested areas for improvement include:

- Implement sinusoidal positional encoding as described in the original paper.  
- Add weight initialization schemes for better training stability.  
- Include example training scripts to showcase practical usage.  
- Add unit tests to ensure code robustness.

---

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

---

## Acknowledgements

This implementation was inspired by the following resources:  
- *Vaswani et al. (2017), "Attention Is All You Need"*  
- [PyTorch Documentation and Tutorials](https://pytorch.org/docs/stable/index.html)