import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_attention_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        attn_output, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=(attention_mask == 0)
            if attention_mask is not None
            else None,
        )
        x = self.layer_norm(x + self.dropout(attn_output))
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        output = self.linear2(self.gelu(self.linear1(x)))
        x = self.layer_norm(x + self.dropout(output))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super(EncoderLayer, self).__init__()
        self.attention_block = MultiHeadAttentionBlock(hidden_size, num_attention_heads)
        self.feed_forward_block = FeedForwardBlock(hidden_size, intermediate_size)

    def forward(self, x, attention_mask=None):
        x = self.attention_block(x, attention_mask)
        x = self.feed_forward_block(x)
        return x


class SmallBertForFeatureExtraction(nn.Module):
    def __init__(
        self,
        vocab_size=30,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=602,
    ):
        super(SmallBertForFeatureExtraction, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, num_attention_heads, intermediate_size)
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        output = embeddings
        for layer in self.encoder_layers:
            output = layer(output, attention_mask)

        return output


class ConcatMultiPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        mean_pool = torch.mean(x, dim=1)  # (batch, input_dim)
        max_pool = torch.max(x, dim=1).values  # (batch, input_dim)

        attn_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(
            1
        )  # (batch, input_dim)

        concat = torch.cat(
            [mean_pool, max_pool, attn_pool], dim=-1
        )  # (batch, 3 × input_dim)
        return concat  # (batch, 768 if input_dim=256)


class CNN_BiLSTM_AttentionBlock(nn.Module):
    def __init__(self, input_dim=512, cnn_filters=128, lstm_hidden_size=128):
        super(CNN_BiLSTM_AttentionBlock, self).__init__()
        self.cnn = nn.Conv1d(
            in_channels=input_dim, out_channels=cnn_filters, kernel_size=1
        )
        self.bn_cnn = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.bi_lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.bn_lstm = nn.BatchNorm1d(2 * lstm_hidden_size)
        self.pooling = ConcatMultiPooling(input_dim=2 * lstm_hidden_size)

    def forward(self, features):
        x = features.permute(0, 2, 1)
        x = self.relu(self.bn_cnn(self.cnn(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bi_lstm(x)
        lstm_out = self.bn_lstm(lstm_out.transpose(1, 2)).transpose(1, 2)
        pool_out = self.pooling(lstm_out)
        return pool_out


class FullFeatureExtractor(nn.Module):
    def __init__(self):
        super(FullFeatureExtractor, self).__init__()
        self.feature_extractor = SmallBertForFeatureExtraction()
        self.cnn_lstm_attention = CNN_BiLSTM_AttentionBlock()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        features = self.feature_extractor(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        extracted_features = self.cnn_lstm_attention(features)
        return extracted_features


if __name__ == "__main__":
    pass
