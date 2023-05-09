import torch
import torch.nn.functional as F
from typeguard import check_argument_types


class CTC(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        blank_id: int = 0,
        dropout_rate: float = 0.0,
        reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id, reduction=reduction_type)

    def probs(self, hs_pad, T=2.0):
        '''
        return probs in frame dim
        hs_pad: batch of padded hidden state sequences (B, seq_len, D)
        mask: tell which frame is pad, [batch_size, seq_len], True is to be padded
        T: temperature, which is to soft the prob
        '''
        # [batch_size, seq_len, n_vocab]
        logit = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate)) / T
        # [batch_size, n_vocab, seq_len]
        probs = logit.permute(0, 2, 1).softmax(-1)
        return probs

    def logits(self, hs_pad, T=2.0):
        '''
        return probs in frame dim
        hs_pad: batch of padded hidden state sequences (B, seq_len, D)
        mask: tell which frame is pad, [batch_size, seq_len], True is to be padded
        T: temperature, which is to soft the prob
        '''
        # [batch_size, seq_len, n_vocab]
        logits = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate)) / T
        # [batch_size, n_vocab, seq_len]
        logits = logits.permute(0, 2, 1)
        return logits



    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


class MaskedCTC(torch.nn.Module):
    """MaskedCTC module"""
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        batch_size = hs_pad.size(0)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # [B, Nvocab, L]
        ys_hat = ys_hat.permute(0, 2, 1)
        mask = torch.ones_like(ys_hat).bool()
        # pad_idx = -1, so may the <eos> will be contain, but i think its ok
        mask[mask.new_tensor(range(batch_size), dtype=torch.long).unsqueeze(-1), ys_pad] = False
        # also contain <blank> and <unk>
        mask[:, 0:2] = False
        ys_hat.masked_fill_(mask, -float('inf'))
        # ys_hat: (L, B, D)
        ys_hat = ys_hat.permute(2, 0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

