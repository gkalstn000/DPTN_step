import copy
import torch
from torch import nn
from models.dptn_networks import modules

class PoseTransformerModule(nn.Module):
    """
    Pose Transformer Module (PTM)
    :param d_model: number of channels in input
    :param nhead: number of heads in attention module
    :param num_CABs: number of CABs
    :param num_TTBs: number of TTBs
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, opt):
        super(PoseTransformerModule, self).__init__()
        self.opt = opt
        d_model = opt.ngf * opt.mult
        encoder_layer = CAB(d_model, opt.nhead, d_model,
                                                opt.activation, opt.affine, opt.norm)
        if opt.norm == 'batch':
            encoder_norm = None
            decoder_norm = nn.BatchNorm1d(d_model, affine=opt.affine)
        elif opt.norm == 'instance':
            encoder_norm = None
            decoder_norm = nn.InstanceNorm1d(d_model, affine=opt.affine)

        self.encoder = CABs(encoder_layer, opt.num_CABs, encoder_norm)

        decoder_layer = TTB(d_model, opt.nhead, d_model,
                                                opt.activation, opt.affine, opt.norm)

        self.decoder = TTBs(decoder_layer, opt.num_TTBs, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = opt.nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, val, pos_embed=None):
        # src: key
        # tgt: query
        # val: value
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        tgt = tgt.flatten(2).permute(2, 0, 1)
        val = val.flatten(2).permute(2, 0, 1)
        if pos_embed != None:
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src, pos=pos_embed)
        hs, first_attn_weights, last_attn_weights = self.decoder(tgt, memory, val, pos=pos_embed) # query, key, value 순서
        return hs.view(bs, c, h, w), first_attn_weights, last_attn_weights


class CABs(nn.Module):
    """
    Context Augment Blocks (CABs)
    :param encoder_layer: CAB
    :param num_CABS: number of CABs
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, encoder_layer, num_CABs, norm=None):
        super(CABs, self).__init__()
        self.layers = _get_clones(encoder_layer, num_CABs)
        self.norm = norm

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        if self.norm is not None:
            output = self.norm(output.permute(1, 2, 0)).permute(2, 0, 1)

        return output


class TTBs(nn.Module):
    """
    Texture Transfer Blocks (TTBs)
    :param decoder_layer: TTB
    :param num_layers: number of TTBs
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, decoder_layer, num_TTBs, norm=None):
        super(TTBs, self).__init__()
        self.layers = _get_clones(decoder_layer, num_TTBs)
        self.norm = norm

    def forward(self, tgt, memory, val, pos = None):
        output = tgt
        weight_list = []
        for layer in self.layers:
            output, attn_output_weights = layer(output, memory, val, pos=pos)
            weight_list.append(attn_output_weights.cpu())

        if self.norm is not None:
            output = self.norm(output.permute(1, 2, 0))
        return output, weight_list[0], weight_list[-1]


class CAB(nn.Module):
    """
    Context Augment Block (CAB)
    :param d_model: number of channels in input
    :param nhead: number of heads in attention module
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 activation="LeakyReLU", affine=True, norm='instance'):
        super(CAB, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if norm == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model, affine=affine)
            self.norm2 = nn.BatchNorm1d(d_model, affine=affine)
        else:
            self.norm1 = nn.InstanceNorm1d(d_model, affine=affine)
            self.norm2 = nn.InstanceNorm1d(d_model, affine=affine)

        self.activation = modules.get_nonlinearity_layer(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + src2
        src = self.norm1(src.permute(1, 2, 0)).permute(2, 0, 1)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src.permute(1, 2, 0)).permute(2, 0, 1)
        return src


class DecomposeNetwork(nn.Module):
    def __init__(self, opt, d_model):
        super(DecomposeNetwork, self).__init__()
        self.opt = opt

        attn_layer = AttentionNetwork(d_model, opt.nhead)
        self.attn_layers = _get_clones(attn_layer, opt.num_attns)
    def forward(self, query, key, value):
        output = query
        for layer in self.attn_layers:
            output = layer(output, key, value)

        return output

class AttentionNetwork(nn.Module):
    """
    Texture Transfer Block (TTB)
    :param d_model: number of channels in input
    :param nhead: number of heads in attention module
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, affine=True):
        super(AttentionNetwork, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first = True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.InstanceNorm1d(d_model, affine=affine)
        self.norm2 = nn.InstanceNorm1d(d_model, affine=affine)

        self.activation = nn.LeakyReLU(0.2, False)

    def forward(self, query, key, value ):
        '''
        B: batch size (N)
        C: channel depth (d_model)
        HxW: pixel length (L)
        :param query: size is (B, C, HxW)
        :param key: size is (B, C, HxW)
        :param value: size is (B, C, HxW)
        :return:
        '''

        # Input shape (B, HxW, C)
        attn_output, _ = self.multihead_attn(query=query.permute(0, 2, 1),
                                                               key=key.permute(0, 2, 1),
                                                               value=value.permute(0, 2, 1))
        attn_output = attn_output + query.permute(0, 2, 1) # shape (B, HxW, C)
        attn_output = self.norm1(attn_output.permute(0, 2, 1)).permute(0, 2, 1) # shape (B, HxW, C)
        output = self.linear2(self.activation(self.linear1(attn_output))) # shape (B, HxW, C)
        output = output + attn_output # shape (B, HxW, C)
        output = self.norm2(output.permute(0, 2, 1)) # shape (B, C, HxW)
        return output






def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


