import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, enc_layers=2, dropout=0.3, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim, enc_hidden, num_layers=enc_layers,
                           dropout=dropout, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        outputs, (h, c) = self.rnn(emb)
        return outputs, (h, c)

class Attention(nn.Module):
    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.attn = nn.Linear(enc_hidden*2 + dec_hidden, dec_hidden)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, dec_hidden, dec_layers=4, dropout=0.3, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim + enc_hidden*2, dec_hidden, num_layers=dec_layers,
                           dropout=dropout, batch_first=True)
        self.attention = Attention(enc_hidden, dec_hidden)
        self.fc_out = nn.Linear(dec_hidden + enc_hidden*2 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, encoder_outputs):
        emb = self.dropout(self.embedding(input_step))
        top_hidden = hidden[0][-1]  # last layer hidden
        context, _ = self.attention(top_hidden, encoder_outputs)
        context = context.unsqueeze(1)
        rnn_input = torch.cat((emb, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        emb_s = emb.squeeze(1)
        concat = torch.cat((output, context.squeeze(1), emb_s), dim=1)
        preds = self.fc_out(concat)
        return preds, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, max_len = tgt.size()
        vocab_size = self.decoder.fc_out.out_features

        encoder_outputs, _ = self.encoder(src)

        # Initialize decoder hidden state
        dec_h = torch.zeros(self.decoder.rnn.num_layers, batch_size, 
                           self.decoder.rnn.hidden_size).to(self.device)
        dec_c = torch.zeros_like(dec_h)
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        input_step = tgt[:,0].unsqueeze(1)  # BOS token

        for t in range(1, max_len):
            preds, (dec_h, dec_c) = self.decoder(input_step, (dec_h, dec_c), encoder_outputs)
            outputs[:,t,:] = preds
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = preds.argmax(1).unsqueeze(1)
            input_step = tgt[:,t].unsqueeze(1) if teacher_force else top1

        return outputs