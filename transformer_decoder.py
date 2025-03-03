import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class TransformerDecoder(nn.Module):
    """
    Transformer decoder for the modified DETR.
    Works with the output of the dilated convolutional encoder.
    """
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, activation="relu", 
                 num_queries=20):
        super().__init__()
        
        # Create decoder layer and clone it for the required number of layers
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, memory, memory_key_padding_mask=None, pos_embed=None):
        """
        Forward pass through the transformer decoder.
        Args:
            memory: Output from the encoder [batch_size, HW/16, d_model]
            memory_key_padding_mask: Mask for the encoder output [batch_size, HW/16]
            pos_embed: Positional embeddings for the encoder output [batch_size, HW/16, d_model]
        
        Returns:
            Decoder output [batch_size, num_queries, d_model]
        """
        batch_size = memory.shape[0]
        
        # Get object queries and expand for batch
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1) # [batch_size, num_queries, d_model]
        
        # Initialize output with zeros
        tgt = torch.zeros_like(query_embed)
        
        # Process through decoder layers
        output = tgt
        
        for layer in self.layers:
            output = layer(
                output, 
                memory, 
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embed=pos_embed,
                query_pos=query_embed
            )
        
        # Apply normalization
        output = self.norm(output)
        
        return output


class TransformerDecoderLayer(nn.Module):
    """
    Single layer of the transformer decoder.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Activation function
        self.activation = _get_activation_fn(activation)
    
    def with_pos_embed(self, tensor, pos):
        """Add positional embeddings to the input tensor if provided."""
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory, memory_key_padding_mask=None, pos_embed=None, query_pos=None):
        """
        Forward pass through a single decoder layer.
        Args:
            tgt: Input tensor [batch_size, num_queries, d_model]
            memory: Output from the encoder [batch_size, HW/16, d_model]
            memory_key_padding_mask: Mask for the encoder output [batch_size, HW/16]
            pos_embed: Positional embeddings for the encoder output [batch_size, HW/16, d_model]
            query_pos: Positional embeddings for the queries [batch_size, num_queries, d_model]
        
        Returns:
            Layer output [batch_size, num_queries, d_model]
        """
        # For self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        
        # Self-attention
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # For cross-attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos_embed)
        
        # Cross-attention
        tgt2 = self.multihead_attn(
            q.transpose(0, 1), 
            k.transpose(0, 1), 
            memory.transpose(0, 1),
            key_padding_mask=memory_key_padding_mask
        )[0].transpose(0, 1)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Activation {activation} not supported.")
