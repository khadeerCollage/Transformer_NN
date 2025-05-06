import numpy as np

# Helper function for layer normalization
def layer_norm(x, epsilon=1e-6):
    """
    Apply layer normalization to stabilize training.
    Args:
        x: Input array (shape: [L, d_model]).
        epsilon: Small constant to avoid division by zero.
    Returns:
        Normalized array of the same shape.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute scaled dot-product attention.
    Args:
        q: Query matrix (shape: [L, d_k]).
        k: Key matrix (shape: [L, d_k]).
        v: Value matrix (shape: [L, d_v]).
        mask: Optional mask to prevent attending to certain positions (shape: [L, L]).
    Returns:
        Output matrix (shape: [L, d_v]) and attention weights (shape: [L, L]).
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)  # Shape: [L, L]
    if mask is not None:
        scores += mask
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Softmax
    output = np.matmul(weights, v)  # Shape: [L, d_v]
    return output, weights

# Multi-Head Attention
def multi_head_attention(q, k, v, num_heads, d_model):
    """
    Compute multi-head attention by splitting into multiple attention heads.
    Args:
        q: Query matrix (shape: [L, d_model]).
        k: Key matrix (shape: [L, d_model]).
        v: Value matrix (shape: [L, d_model]).
        num_heads: Number of attention heads.
        d_model: Total dimension of the model (must be divisible by num_heads).
    Returns:
        Output matrix (shape: [L, d_model]).
    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads
    
    # Split into heads
    q_heads = np.split(q, num_heads, axis=-1)  # List of [L, head_dim]
    k_heads = np.split(k, num_heads, axis=-1)
    v_heads = np.split(v, num_heads, axis=-1)
    
    # Apply attention to each head
    head_outputs = []
    for q_h, k_h, v_h in zip(q_heads, k_heads, v_heads):
        out, _ = scaled_dot_product_attention(q_h, k_h, v_h)
        head_outputs.append(out)
    
    # Concatenate and apply linear transformation
    concat = np.concatenate(head_outputs, axis=-1)  # Shape: [L, d_model]
    W_o = np.random.randn(d_model, d_model)  # Output projection
    output = np.matmul(concat, W_o)
    return output

# Positional Encoding
def get_positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.
    Args:
        max_len: Maximum sequence length.
        d_model: Embedding dimension.
    Returns:
        Positional encoding matrix (shape: [max_len, d_model]).
    """
    pos_enc = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / 10000 ** (2 * i / d_model))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / 10000 ** (2 * i / d_model))
    return pos_enc

# Feed-Forward Neural Network
def feed_forward(x, d_ff=32, d_model=8):
    """
    Apply position-wise feed-forward network.
    Args:
        x: Input array (shape: [L, d_model]).
        d_ff: Dimension of the feed-forward hidden layer.
        d_model: Dimension of the input/output.
    Returns:
        Output array (shape: [L, d_model]).
    """
    W1 = np.random.randn(d_model, d_ff)
    W2 = np.random.randn(d_ff, d_model)
    x = np.matmul(np.maximum(0, np.matmul(x, W1)), W2)  # ReLU activation
    return x

# Encoder Layer
def encoder_layer(x, num_heads, d_model):
    """
    Single encoder layer with multi-head attention and feed-forward network.
    Args:
        x: Input array (shape: [L, d_model]).
        num_heads: Number of attention heads.
        d_model: Model dimension.
    Returns:
        Output array (shape: [L, d_model]).
    """
    # Multi-Head Self-Attention
    attn_output = multi_head_attention(x, x, x, num_heads, d_model)
    x = layer_norm(x + attn_output)  # Residual connection + normalization
    
    # Feed-Forward
    ff_output = feed_forward(x, d_ff=32, d_model=d_model)
    x = layer_norm(x + ff_output)  # Residual connection + normalization
    
    return x

# Decoder Layer
def decoder_layer(x, enc_output, num_heads, d_model):
    """
    Single decoder layer with masked self-attention, cross-attention, and feed-forward.
    Args:
        x: Input array (target sequence, shape: [L, d_model]).
        enc_output: Encoder output (shape: [L, d_model]).
        num_heads: Number of attention heads.
        d_model: Model dimension.
    Returns:
        Output array (shape: [L, d_model]).
    """
    # Masked Multi-Head Self-Attention
    mask = np.tril(np.ones((x.shape[0], x.shape[0])))
    mask[mask == 0] = -np.inf
    attn1 = multi_head_attention(x, x, x, num_heads, d_model)  # Simplified, needs mask
    x = layer_norm(x + attn1)
    
    # Multi-Head Cross-Attention with encoder output
    attn2 = multi_head_attention(x, enc_output, enc_output, num_heads, d_model)
    x = layer_norm(x + attn2)
    
    # Feed-Forward
    ff_output = feed_forward(x, d_ff=32, d_model=d_model)
    x = layer_norm(x + ff_output)
    
    return x

# Encoder
def encoder(input_seq, num_layers, num_heads, d_model):
    """
    Full encoder with multiple layers.
    Args:
        input_seq: Input sequence (shape: [L, d_model]).
        num_layers: Number of encoder layers.
        num_heads: Number of attention heads.
        d_model: Model dimension.
    Returns:
        Encoder output (shape: [L, d_model]).
    """
    x = input_seq + get_positional_encoding(input_seq.shape[0], d_model)
    for _ in range(num_layers):
        x = encoder_layer(x, num_heads, d_model)
    return x

# Decoder
def decoder(target_seq, enc_output, num_layers, num_heads, d_model):
    """
    Full decoder with multiple layers.
    Args:
        target_seq: Target sequence (shape: [L, d_model]).
        enc_output: Encoder output (shape: [L, d_model]).
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        d_model: Model dimension.
    Returns:
        Decoder output (shape: [L, d_model]).
    """
    x = target_seq + get_positional_encoding(target_seq.shape[0], d_model)
    for _ in range(num_layers):
        x = decoder_layer(x, enc_output, num_heads, d_model)
    return x

# Transformer
def transformer(input_seq, target_seq, num_layers, num_heads, d_model):
    """
    Full Transformer model combining encoder and decoder.
    Args:
        input_seq: Input sequence (shape: [L, d_model]).
        target_seq: Target sequence (shape: [L, d_model]).
        num_layers: Number of layers in encoder and decoder.
        num_heads: Number of attention heads.
        d_model: Model dimension.
    Returns:
        Transformer output (shape: [L, d_model]).
    """
    # Encoder
    enc_output = encoder(input_seq, num_layers, num_heads, d_model)
    
    # Decoder
    dec_output = decoder(target_seq, enc_output, num_layers, num_heads, d_model)
    
    # Final linear layer (simplified)
    W_final = np.random.randn(d_model, d_model)
    output = np.matmul(dec_output, W_final)
    return output

# Example Usage
if __name__ == "__main__":
    # Parameters
    L = 4  # Sequence length
    d_model = 8  # Model dimension
    num_layers = 2  # Number of layers
    num_heads = 2  # Number of attention heads
    
    # Random input and target sequences (simulating embedded tokens)
    input_seq = np.random.randn(L, d_model)
    target_seq = np.random.randn(L, d_model)
    
    # Run the Transformer
    output = transformer(input_seq, target_seq, num_layers, num_heads, d_model)
    print("Transformer Output Shape:", output.shape)  # Expected: (4, 8)
    print("Transformer Output:\n", output)