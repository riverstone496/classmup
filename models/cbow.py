import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingWithConv2d(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingWithConv2d, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # 1x1の畳み込みを使用
        self.conv = nn.Conv2d(in_channels=vocab_size, out_channels=embed_dim, kernel_size=1)
        self.conv.base_fan_in=vocab_size
        self.conv.base_fan_out=64
        
    def forward(self, input_indices):
        # one-hot encoding
        one_hot = torch.nn.functional.one_hot(input_indices, num_classes=self.vocab_size).float()
        # Reshape one_hot: (batch_size * num_context_words, vocab_size, 1, 1)
        one_hot = one_hot.view(-1, self.vocab_size, 1, 1)
        # Apply 1x1 convolution
        embedding = self.conv(one_hot)
        # Reshape back to: (batch_size, num_context_words, embed_dim)
        embedding = embedding.squeeze(3).squeeze(2).view(input_indices.size(0), input_indices.size(1), self.embed_dim)
        return embedding
    
class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int, embed_dim = 300, embed_max_norm=None, bias=True):
        super(CBOW_Model, self).__init__()
        # self.input_layer = EmbeddingWithConv2d(
        #     vocab_size=vocab_size,
        #     embed_dim=embed_dim,
        # )
        self.input_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            max_norm=embed_max_norm,
        )
        self.output_layer = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
            bias=bias
        )
        self.input_layer.base_fan_in=vocab_size
        self.output_layer.base_fan_in=64
        self.input_layer.base_fan_out=64
        self.output_layer.base_fan_out=vocab_size

    def forward(self, inputs):
        embeds = torch.sum(self.input_layer(inputs), dim=1) # [200, 4, 50] => [200, 50]
        out = self.output_layer(embeds) # nonlinear + projection
        log_probs = F.log_softmax(out, dim=1) # softmax compute log probability
        return log_probs

class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int, embed_dim = 300, embed_max_norm=1, bias=True):
        super(SkipGram_Model, self).__init__()
        self.input_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            max_norm=embed_max_norm,
        )
        self.output_layer = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
            bias=bias
        )

    def forward(self, inputs_):
        x = self.input_layer(inputs_)
        x = self.output_layer(x)
        return x