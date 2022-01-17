import torch
import torch.nn as nn
import random
import torch.nn.utils as nnu
import torch.nn.functional as F
from collections import OrderedDict
try:
    from core import Model, LayerNorm
    from modules import *
except ImportError:
    from .core import Model, LayerNorm
    from .modules import *

# BEWARE: PyTorch >= 0.4.1 REQUIRED
from torch.nn.utils.spectral_norm import spectral_norm

from .modules_pase import FeBlock, FeResBlock, format_frontend_chunk, format_frontend_output, build_rnn_block,  Model as PASE_Model

class WaveFe(PASE_Model):
    """ Convolutional front-end to process waveforms
        into a decimated intermediate representation 
    """
    def __init__(self, num_inputs=1,
                 sincnet=True,
                 kwidths=[251, 10, 5, 5, 5, 5, 5, 5],
                 strides=[1, 10, 2, 1, 2, 1, 2, 2],
                 dilations=[1, 1, 1, 1, 1, 1, 1, 1],
                 fmaps=[64, 64, 128, 128, 256, 256, 512, 512],
                 norm_type='bnorm',
                 pad_mode='reflect', sr=16000,
                 emb_dim=256,
                 rnn_dim=None,
                 activation=None,
                 rnn_pool=False,
                 rnn_layers=1,
                 rnn_dropout=0,
                 rnn_type='qrnn',
                 vq_K=None,
                 vq_beta=0.25,
                 vq_gamma=0.99,
                 norm_out=False,
                 tanh_out=False,
                 resblocks=False,
                 denseskips=False,
                 densemerge='sum',
                 name='WaveFe'):
        super().__init__(name=name)
        # apply sincnet at first layer
        self.sincnet = sincnet
        self.kwidths = kwidths
        self.strides = strides
        self.fmaps = fmaps
        self.densemerge = densemerge
        if denseskips:
            self.denseskips = nn.ModuleList()
        self.blocks = nn.ModuleList()
        assert len(kwidths) == len(strides)
        assert len(strides) == len(fmaps)
        concat_emb_dim = emb_dim
        ninp = num_inputs
        for n, (kwidth, stride, dilation, fmap) in enumerate(zip(kwidths,
                                                                 strides,
                                                                 dilations,
                                                                 fmaps),
                                                             start=1):
            if n > 1:
                # make sure sincnet is deactivated after first layer
                sincnet = False
            if resblocks and not sincnet:
                feblock = FeResBlock(ninp, fmap, kwidth,
                                     downsample=stride,
                                     act=activation,
                                     pad_mode=pad_mode, norm_type=norm_type)
            else:
                feblock = FeBlock(ninp, fmap, kwidth, stride,
                                  dilation,
                                  act=activation,
                                  pad_mode=pad_mode,
                                  norm_type=norm_type,
                                  sincnet=sincnet,
                                  sr=sr)
            self.blocks.append(feblock)
            if denseskips and n < len(kwidths):
                # add projection adapter
                self.denseskips.append(nn.Conv1d(fmap, emb_dim, 1, bias=False))
                if densemerge == 'concat':
                    concat_emb_dim += emb_dim
            ninp = fmap
        # last projection
        if rnn_pool:
            if rnn_dim is None:
                rnn_dim = emb_dim
            self.rnn = build_rnn_block(fmap, rnn_dim // 2,
                                       rnn_layers=rnn_layers,
                                       rnn_type=rnn_type,
                                       bidirectional=True,
                                       dropout=rnn_dropout)
            self.W = nn.Conv1d(rnn_dim, emb_dim, 1)
        else:
            self.W = nn.Conv1d(fmap, emb_dim, 1)
        self.emb_dim = concat_emb_dim
        self.rnn_pool = rnn_pool
        self.quantizer = None
        # ouptut vectors are normalized to norm^2 1
        if norm_out:
            if norm_type == 'bnorm':
                self.norm_out = nn.BatchNorm1d(self.emb_dim, affine=False)
            else:
                self.norm_out = nn.InstanceNorm1d(self.emb_dim)
        self.tanh_out = tanh_out

    def fuse_skip(self, input_, skip):
        #print('input_ shape: ', input_.shape)
        #print('skip shape: ', skip.shape)
        dfactor = skip.shape[2] // input_.shape[2]
        if dfactor > 1:
            #print('dfactor: ', dfactor)
            # downsample skips
            # [B, F, T]
            maxlen = input_.shape[2] * dfactor
            skip = skip[:, :, :maxlen]
            bsz, feats, slen = skip.shape
            skip_re = skip.view(bsz, feats, slen // dfactor, dfactor)
            skip = torch.mean(skip_re, dim=3)
            #skip = F.adaptive_avg_pool1d(skip, input_.shape[2])
        if self.densemerge == 'concat':
            return torch.cat((input_, skip), dim=1)
        elif self.densemerge == 'sum':
            return input_ + skip
        else:
            raise TypeError('Unknown densemerge: ', self.densemerge)

    def forward(self, batch, device=None, mode=None):
        # batch possible chunk and contexts, or just forward non-dict tensor
        x, data_fmt = format_frontend_chunk(batch, device)
        h = x
        denseskips = hasattr(self, 'denseskips')
        if denseskips:
            dskips = None
            dskips = []
        for n, block in enumerate(self.blocks):
            h = block(h)
            if denseskips and (n + 1) < len(self.blocks):
                # denseskips happen til the last but one layer
                # til the embedding one
                proj = self.denseskips[n]
                dskips.append(proj(h))
                """
                if dskips is None:
                    dskips = proj(h)
                else:
                    h_proj = proj(h)
                    dskips = self.fuse_skip(h_proj, dskips)
                """
        if self.rnn_pool:
            h = h.transpose(1, 2).transpose(0, 1)
            h, _ = self.rnn(h)
            h = h.transpose(0, 1).transpose(1, 2)
            #y = self.W(h)
        #else:
        y = self.W(h)
        if denseskips:
            for dskip in dskips:
                # sum all dskips contributions in the embedding
                y = self.fuse_skip(y, dskip)
        if hasattr(self, 'norm_out'):
            y = self.norm_out(y)
        if self.tanh_out:
            y = torch.tanh(y)

        if self.quantizer is not None:
            qloss, y, pp, enc = self.quantizer(y)
            if self.training:
                return qloss, y, pp, enc
            else:
                return y

        return format_frontend_output(y, data_fmt, mode)

class Discriminator(Model):

    def __init__(self, ninputs, fmaps,
                 kwidth, poolings,
                 pool_type='none',
                 pool_slen=None,
                 norm_type='bnorm',
                 bias=True,
                 phase_shift=None,
                 sinc_conv=False,pase_net=False):
        super().__init__(name='Discriminator')
        # If PASE should be used for feature extraction, setup a 
        # PASE feature extractor
        if pase_net:
            self.pase = WaveFe(kwidths=[251, 20, 11, 11, 11, 11, 11, 11],
                            strides=[1, 10, 2, 1, 2, 1, 2, 2],
                            fmaps=[64, 64, 128, 128, 256, 256, 512, 512],
                            emb_dim=100,
                            norm_out=True)
            #self.pase.load_pretrained('/usr/home/cheema/Downloads/FE_e199.ckpt', load_last=True, verbose=True)
        
        # phase_shift randomly occurs within D layers
        # as proposed in https://arxiv.org/pdf/1802.04208.pdf
        # phase shift has to be specified as an integer
        self.phase_shift = phase_shift
        if phase_shift is not None:
            assert isinstance(phase_shift, int), type(phase_shift)
            assert phase_shift > 1, phase_shift
        if pool_slen is None:
            raise ValueError('Please specify D network pool seq len '
                             '(pool_slen) in the end of the conv '
                             'stack: [inp_len // (total_pooling_factor)]')
        ninp = ninputs
        # SincNet as proposed in
        # https://arxiv.org/abs/1808.00158
        if sinc_conv:
            # build sincnet module as first layer
            self.sinc_conv = SincConv(fmaps[0] // 2,
                                      251, 16e3, padding='SAME')
            ninp = fmaps[0]
            fmaps = fmaps[1:]
        self.enc_blocks = nn.ModuleList()
        for pi, (fmap, pool) in enumerate(zip(fmaps,
                                              poolings),
                                          start=1):
            enc_block = GConv1DBlock(
                ninp, fmap, kwidth, stride=pool,
                bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap
        self.pool_type = pool_type
        if pool_type == 'none':
            # resize tensor to fit into FC directly
            pool_slen *= fmaps[-1]
            if self.pase:
                pool_slen = 200 * 102 #todo - calculate size from pase model.
            self.fc = nn.Sequential(
                nn.Linear(pool_slen, 256),
                nn.PReLU(256),
                nn.Linear(256, 128),
                nn.PReLU(128),
                nn.Linear(128, 1)
            )
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc[0])
                torch.nn.utils.spectral_norm(self.fc[2])
                torch.nn.utils.spectral_norm(self.fc[3])
        elif pool_type == 'conv':
            self.pool_conv = nn.Conv1d(fmaps[-1], 1, 1)
            self.fc = nn.Linear(pool_slen, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.pool_conv)
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gmax':
            self.gmax = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gavg':
            self.gavg = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Conv1d(fmaps[-1], fmaps[-1], 1),
                nn.PReLU(fmaps[-1]),
                nn.Conv1d(fmaps[-1], 1, 1)
            )
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.mlp[0])
                torch.nn.utils.spectral_norm(self.mlp[1])
        else:
            raise TypeError('Unrecognized pool type: ', pool_type)

    def forward(self, x):
        h = x
        if hasattr(self, 'sinc_conv'):
            h_l, h_r = torch.chunk(h, 2, dim=1)
            h_l = self.sinc_conv(h_l)
            h_r = self.sinc_conv(h_r)
            h = torch.cat((h_l, h_r), dim=1)
        # store intermediate activations
        int_act = {}
        if self.pase:
            x_l, x_r = torch.chunk(x, 2, dim=1)
            x_l = self.pase(x_l)
            x_r = self.pase(x_r)
            h = torch.cat((x_l, x_r), dim=1)
        else:
            for ii, layer in enumerate(self.enc_blocks):
                if self.phase_shift is not None:
                    shift = random.randint(1, self.phase_shift)
                    # 0.5 chance of shifting right or left
                    right = random.random() > 0.5
                    # split tensor in time dim (dim 2)
                    if right:
                        sp1 = h[:, :, :-shift]
                        sp2 = h[:, :, -shift:]
                        h = torch.cat((sp2, sp1), dim=2)
                    else:
                        sp1 = h[:, :, :shift]
                        sp2 = h[:, :, shift:]
                        h = torch.cat((sp2, sp1), dim=2)
                h = layer(h)
                int_act['h_{}'.format(ii)] = h
        if self.pool_type == 'conv':
            h = self.pool_conv(h)
            h = h.view(h.size(0), -1)
            int_act['avg_conv_h'] = h
            y = self.fc(h)
        elif self.pool_type == 'none':
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gmax':
            h = self.gmax(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gavg':
            h = self.gavg(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'mlp':
            y = self.mlp(h)
        int_act['logit'] = y
        return y, int_act


if __name__ == '__main__':
    # pool_slen = 16 because we have input len 16384
    # and we perform 5 pooling layers of 4, so 16384 // (4 ** 5) = 16
    disc = Discriminator(2, [64, 128, 256, 512, 1024],
                         31, [4] * 5, pool_type='none',
                         pool_slen=16)
    print(disc)
    print('Num params: ', disc.get_n_params())
    x = torch.randn(1, 2, 16384)
    y, _ = disc(x)
    print(y)
    print('x size: {} -> y size: {}'.format(x.size(), y.size()))
