from __future__ import annotations

import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        embedding_weights: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
        bidirectional: bool = True,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_weights is not None:
            if embedding_weights.shape != self.embedding.weight.shape:
                raise ValueError("embedding_weights shape mismatch")
            self.embedding.weight.data.copy_(embedding_weights)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.dropout(out)


class POSTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        embedding_weights: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
        dropout: float = 0.5,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
            embedding_weights=embedding_weights,
            freeze_embeddings=freeze_embeddings,
            bidirectional=bidirectional,
            pad_idx=pad_idx,
        )
        self.classifier = nn.Linear(self.encoder.out_dim, num_tags)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids, lengths)
        return self.classifier(h)


class CRF(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return mean negative log-likelihood over the batch."""
        log_partition = self._compute_log_partition(emissions, mask)
        gold_score = self._compute_gold_score(emissions, tags, mask)
        nll = log_partition - gold_score
        return nll.mean()

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        if num_tags != self.num_tags:
            raise ValueError("Emission/tag size mismatch")

        score = self.start_transitions + emissions[:, 0]  # [B, C]
        for t in range(1, seq_len):
            emit_t = emissions[:, t].unsqueeze(1)  # [B, 1, C]
            score_t = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t  # [B, C, C]
            next_score = torch.logsumexp(score_t, dim=1)  # [B, C]
            mask_t = mask[:, t].unsqueeze(1)
            score = torch.where(mask_t, next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _compute_gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape

        first_emit = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        score = self.start_transitions[tags[:, 0]] + first_emit
        for t in range(1, seq_len):
            prev_tags = tags[:, t - 1]
            curr_tags = tags[:, t]
            trans_score = self.transitions[prev_tags, curr_tags]
            emit_score = emissions[:, t].gather(1, curr_tags.unsqueeze(1)).squeeze(1)
            score = score + (trans_score + emit_score) * mask[:, t]

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size, device=tags.device), lengths]
        score = score + self.end_transitions[last_tags]
        return score

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        batch_size, seq_len, _ = emissions.shape

        score = self.start_transitions + emissions[:, 0]  # [B, C]
        history: list[torch.Tensor] = []

        for t in range(1, seq_len):
            score_t = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_path = torch.max(score_t, dim=1)
            best_score = best_score + emissions[:, t]
            mask_t = mask[:, t].unsqueeze(1)
            score = torch.where(mask_t, best_score, score)
            history.append(best_path)

        score = score + self.end_transitions
        best_last_score, best_last_tag = torch.max(score, dim=1)
        del best_last_score  # value not required after argmax

        results: list[list[int]] = []
        lengths = mask.long().sum(dim=1)

        for b in range(batch_size):
            seq_end = int(lengths[b].item()) - 1
            best_tag = int(best_last_tag[b].item())
            path = [best_tag]

            for hist_t in range(seq_end - 1, -1, -1):
                best_tag = int(history[hist_t][b, best_tag].item())
                path.append(best_tag)

            path.reverse()
            results.append(path)

        return results


class NERTaggerCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        embedding_weights: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
        dropout: float = 0.5,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
            embedding_weights=embedding_weights,
            freeze_embeddings=freeze_embeddings,
            bidirectional=bidirectional,
            pad_idx=pad_idx,
        )
        self.classifier = nn.Linear(self.encoder.out_dim, num_tags)
        self.crf = CRF(num_tags=num_tags)

    def emissions(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids, lengths)
        return self.classifier(h)

    def loss(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        em = self.emissions(input_ids, lengths)
        return self.crf(em, tags, mask)

    def decode(self, input_ids: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        em = self.emissions(input_ids, lengths)
        return self.crf.decode(em, mask)


class NERTaggerSoftmax(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        embedding_weights: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
        dropout: float = 0.5,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
            embedding_weights=embedding_weights,
            freeze_embeddings=freeze_embeddings,
            bidirectional=bidirectional,
            pad_idx=pad_idx,
        )
        self.classifier = nn.Linear(self.encoder.out_dim, num_tags)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids, lengths)
        return self.classifier(h)
