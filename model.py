import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pywt


class DiscreteWaveletTransform(nn.Module):
    def __init__(self, wavelet='db4', levels=3):
        super(DiscreteWaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
    
    def forward(self, x):
        B, T, N, _ = x.shape
        coeffs_list = []
        
        for b in range(B):
            for n in range(N):
                signal = x[b, :, n, 0].detach().cpu().numpy()
                coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels)
                coeffs_list.append(coeffs)
        
        level_coeffs = []
        for level in range(self.levels + 1):
            level_data = []
            for b in range(B):
                node_data = []
                for n in range(N):
                    idx = b * N + n
                    if level == 0:
                        coeff = coeffs_list[idx][0]
                    else:
                        coeff = coeffs_list[idx][level]
                    
                    coeff_tensor = torch.from_numpy(coeff).float().to(x.device)
                    if len(coeff_tensor) < T:
                        coeff_tensor = F.interpolate(
                            coeff_tensor.unsqueeze(0).unsqueeze(0), 
                            size=T, mode='linear', align_corners=False
                        ).squeeze()
                    elif len(coeff_tensor) > T:
                        coeff_tensor = coeff_tensor[:T]
                    
                    node_data.append(coeff_tensor)
                level_data.append(torch.stack(node_data, dim=0))
            level_coeffs.append(torch.stack(level_data, dim=0))
        
        frequency_series = torch.stack(level_coeffs, dim=-1)
        return frequency_series


class MSATemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=2, dropout=0.1):
        super(MSATemporalEncoder, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, src):
        B, T, N, D = src.shape
        src = src.view(B*N, T, D)
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.layer_norm(output)
        output = output + self.feed_forward(output)
        output = output.view(B, T, N, D)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class FrequencyGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(FrequencyGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('adj_matrix', adj_matrix)
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.adj_matrix, support.transpose(2, 3)).transpose(2, 3)
        return output + self.bias


class MultiScaleFrequencyGraphLearning(nn.Module):
    def __init__(self, d_model, num_nodes, num_scales=3, dropout=0.1):
        super(MultiScaleFrequencyGraphLearning, self).__init__()
        self.num_scales = num_scales
        self.d_model = d_model
        
        self.graph_convs = nn.ModuleList()
        for i in range(num_scales):
            adj = self.create_scale_adj_matrix(num_nodes, scale=i+1)
            self.graph_convs.append(FrequencyGraphConvolution(d_model, d_model, adj))
        
        self.gcn_layer = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.scale_constraint = ScaleIndependenceConstraint(num_scales)
    
    def create_scale_adj_matrix(self, num_nodes, scale=1):
        adj = torch.eye(num_nodes)
        for i in range(num_nodes):
            for j in range(max(0, i-scale), min(num_nodes, i+scale+1)):
                if i != j:
                    adj[i, j] = 1.0 / (abs(i-j) + 1)
        
        d = torch.sum(adj, dim=1)
        d_sqrt_inv = torch.pow(d, -0.5)
        d_sqrt_inv[torch.isinf(d_sqrt_inv)] = 0.
        adj_normalized = torch.matmul(torch.matmul(torch.diag(d_sqrt_inv), adj), torch.diag(d_sqrt_inv))
        
        return adj_normalized
    
    def forward(self, x):
        scale_outputs = []
        
        for i, graph_conv in enumerate(self.graph_convs):
            scale_out = graph_conv(x)
            scale_outputs.append(scale_out)
        
        multi_scale_features = torch.cat(scale_outputs, dim=-1)
        output = self.gcn_layer(multi_scale_features)
        constraint_loss = self.scale_constraint(scale_outputs)
        
        return output, constraint_loss


class ScaleIndependenceConstraint(nn.Module):
    def __init__(self, num_scales):
        super(ScaleIndependenceConstraint, self).__init__()
        self.num_scales = num_scales
    
    def forward(self, scale_outputs):
        constraint_loss = 0.0
        for i in range(len(scale_outputs)):
            for j in range(i+1, len(scale_outputs)):
                hi = scale_outputs[i]
                hj = scale_outputs[j]
                
                similarity = F.cosine_similarity(hi.flatten(1), hj.flatten(1), dim=1).mean()
                constraint_loss += similarity
        
        return constraint_loss


class FrequencyIndependentPredictor(nn.Module):
    def __init__(self, d_model, output_len, dropout=0.1):
        super(FrequencyIndependentPredictor, self).__init__()
        self.output_len = output_len
        
        self.wavelet_aggregate_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.idwt_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, target_time_series=None):
        B, T, N, D = x.shape
        
        x_reshaped = x.view(B*N, T, D)
        lstm_out, _ = self.wavelet_aggregate_lstm(x_reshaped)
        idwt_out = self.idwt_layer(lstm_out)
        idwt_out = idwt_out.view(B, T, N, D)
        
        predictions = []
        current_input = idwt_out[:, -1:, :, :]
        
        for _ in range(self.output_len):
            pred_step = self.predictor(current_input)
            predictions.append(pred_step)
            current_input = torch.cat([current_input[:, :, :, 1:], pred_step], dim=-1)
        
        predictions = torch.cat(predictions, dim=1)
        return predictions


class WaveletFrequencyGraphModel(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, d_model=64, 
                 nhead=8, num_encoder_layers=2, num_scales=3, 
                 wavelet_levels=3, dropout=0.1):
        super(WaveletFrequencyGraphModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        
        self.dwt = DiscreteWaveletTransform(levels=wavelet_levels)
        self.input_embedding = nn.Linear(wavelet_levels + 1, d_model)
        
        self.temporal_encoder = MSATemporalEncoder(
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        self.freq_graph_learning = MultiScaleFrequencyGraphLearning(
            d_model=d_model,
            num_nodes=num_nodes,
            num_scales=num_scales,
            dropout=dropout
        )
        
        self.predictor = FrequencyIndependentPredictor(
            d_model=d_model,
            output_len=output_len,
            dropout=dropout
        )
    
    def forward(self, x):
        frequency_series = self.dwt(x)
        embedded = self.input_embedding(frequency_series)
        temporal_encoded = self.temporal_encoder(embedded)
        graph_output, constraint_loss = self.freq_graph_learning(temporal_encoded)
        predictions = self.predictor(graph_output)
        
        return predictions, constraint_loss
    
    def compute_loss(self, predictions, targets, constraint_loss, 
                    lambda_constraint=0.1, lambda_wwmse=1.0):
        mse_loss = F.mse_loss(predictions, targets)
        total_loss = lambda_wwmse * mse_loss + lambda_constraint * constraint_loss
        
        return total_loss, mse_loss, constraint_loss