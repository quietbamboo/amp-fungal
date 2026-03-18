from utils.feature_extractor import *


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        q = q.unsqueeze(1)  # (batch, 1, dim)
        out, _ = self.attn(q, kv, kv)
        return self.norm(out.squeeze(1))  # (batch, dim)


class ClassificationHead(nn.Module):
    def __init__(self, input_size=768, hidden_sizes=None, output_size=2):
        super(ClassificationHead, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.hidden_layers = nn.ModuleList([])
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.dropout(self.relu(hidden_layer(x)))
        x = self.output_layer(x)
        return x


class PretrainModel(nn.Module):
    def __init__(self, feature_dim, hidden_sizes=None, output_size=2):
        super(PretrainModel, self).__init__()
        self.feature_extractor = SmallBertForFeatureExtraction()
        self.cnn_lstm_attention = CNN_BiLSTM_AttentionBlock()
        self.classifier = ClassificationHead(
            input_size=feature_dim, hidden_sizes=hidden_sizes, output_size=output_size
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        features = self.feature_extractor(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        deep_features = self.cnn_lstm_attention(features)
        logits = self.classifier(deep_features)
        return logits


class ConcatModel(nn.Module):
    def __init__(self, modal_list, input_size, hidden_sizes=None, output_size=2):
        super(ConcatModel, self).__init__()
        self.modal_list = modal_list
        self.feature_sizes = {
            "bert": 768,
            "unirep": 1900,
            "esm2": 1280,
            "prott5": 1024,
            "esmc": 1152,
        }

        # 初始化模块（只在包含 bert 时启用）
        if "bert" in modal_list:
            self.feature_extractor = SmallBertForFeatureExtraction()
            self.cnn_lstm_attention = CNN_BiLSTM_AttentionBlock()

        self.classifier = ClassificationHead(
            input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size
        )

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs
    ):
        """
        这里所有bert相关输入作为单独参数接收，其他模态特征通过 kwargs 传入。
        """
        embedded = []

        for i, name in enumerate(self.modal_list):
            if name == "bert":
                # 用单独的bert输入参数调用提取器
                x = self.feature_extractor(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                x = self.cnn_lstm_attention(x)  # (batch, 768)
            else:
                # 其他模态通过kwargs传入，key即模态名
                if name not in kwargs:
                    raise ValueError(f"Missing input for modality {name}")
                x = kwargs[name]  # (batch, raw_dim)

            embedded.append(x)

        concat_feat = torch.cat(embedded, dim=-1)
        logits = self.classifier(concat_feat)
        return logits


class CrossAttentionModel(nn.Module):
    def __init__(self, modal_list, hidden_sizes=None, output_size=2):
        super(CrossAttentionModel, self).__init__()
        self.modal_list = modal_list
        self.feature_sizes = {
            "bert": 768,
            "unirep": 1900,
            "esm2": 1280,
            "prott5": 1024,
            "esmc": 1152,
        }

        # 初始化模块（只在包含 bert 时启用）
        if "bert" in modal_list:
            self.feature_extractor = SmallBertForFeatureExtraction()
            self.cnn_lstm_attention = CNN_BiLSTM_AttentionBlock()

        # 为每个模态建立投影层（原始维度 → 512）
        self.projectors = nn.ModuleDict(
            {name: nn.Linear(self.feature_sizes[name], 512) for name in modal_list}
        )

        # 每个模态都作为 Q，其余为 K/V → CrossAttention
        self.attn_blocks = nn.ModuleList(
            [CrossAttentionBlock(dim=512) for _ in modal_list]
        )

        self.classifier = ClassificationHead(
            input_size=512 * len(modal_list),
            hidden_sizes=hidden_sizes,
            output_size=output_size,
        )

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs
    ):
        """
        这里所有bert相关输入作为单独参数接收，其他模态特征通过 kwargs 传入。
        """
        embedded = []

        for i, name in enumerate(self.modal_list):
            if name == "bert":
                # 用单独的bert输入参数调用提取器
                x = self.feature_extractor(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                x = self.cnn_lstm_attention(x)  # (batch, 768)
            else:
                # 其他模态通过kwargs传入，key即模态名
                if name not in kwargs:
                    raise ValueError(f"Missing input for modality {name}")
                x = kwargs[name]  # (batch, raw_dim)

            projected = self.projectors[name](x)  # (batch, 512)
            embedded.append(projected)

        fused = []
        for i in range(len(embedded)):
            q = embedded[i]
            kv = torch.stack(
                [embedded[j] for j in range(len(embedded)) if j != i], dim=1
            )
            fused.append(self.attn_blocks[i](q, kv))

        concat_feat = torch.cat(fused, dim=-1)
        logits = self.classifier(concat_feat)
        return logits
