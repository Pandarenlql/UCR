import torch
from torch import nn
import math
import models.densenet as densenet
import numpy as np
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# Character Feature Reconstruction module
class CFR(nn.Module):
    # Input Tensor shape is (b, max len, attn_dim)
    def __init__(self, dim):
        super(CFR, self).__init__()

        self.agg_fc = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, caption_lengths, train=True):
        weights = self.agg_fc(x)
        attn_feature = x * weights

        if not train:
            aggregation_feature = torch.mean(attn_feature, dim=1)
        else:
            aggregation_feature = None
            for i, real_len in enumerate(caption_lengths):
                # 去掉start位
                real_len = real_len-1
                _agg_feature =  torch.mean(attn_feature[i, :real_len, :], dim=0).unsqueeze(0)
                
                if aggregation_feature is None:
                    aggregation_feature = _agg_feature
                else:
                    aggregation_feature = torch.cat((aggregation_feature, _agg_feature), dim=0)

        return aggregation_feature


class PositionalEncoder(nn.Module):
    """
    1D PE for Transformer-based decoder
    """
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len]
        if x.is_cuda:
            pe.cuda()
        return pe


class TwoDimPositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    
    Comes form: https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    """
    def __init__(self, num_pos_feats=64, patch_h=8, patch_w=8, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.heigh_num = patch_h
        self.width_num = patch_w

    def forward(self, x):
        x = rearrange(x, 'b (h w) d -> b d h w', h=self.heigh_num, w=self.width_num)
        b, d, h, w = x.shape
        ones = torch.ones(b, h, w)
        y_embed = ones.cumsum(1, dtype=torch.float32).to(x.device)
        x_embed = ones.cumsum(2, dtype=torch.float32).to(x.device)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = rearrange(pos, 'b d h w -> b (h w) d', h=self.heigh_num, w=self.width_num)
        return pos


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, pos):
        q = self.to_q(x + pos)
        k = self.to_k(x + pos)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MaskedAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            dots = dots.masked_fill(mask == 0, -1e9)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DecoderAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, encoder_out, encoder_pos):
        q = self.to_q(x)
        k = self.to_k(encoder_out + encoder_pos)
        v = self.to_v(encoder_out)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.encoder = nn.ModuleList([])
        for _ in range(depth):
            self.encoder.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        
        self.decoder = nn.ModuleList([])
        for _ in range(depth):
            self.decoder.append(nn.ModuleList([
                MaskedAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                DecoderAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pos, ids_embedding, pos_decoder, ids_mask):
        for attn, ff in self.encoder:
            x = attn(self.norm(x), pos) + x
            x = ff(self.norm(x)) + x
        encoder_out = x
        x = ids_embedding
        for mask_attn, decoder_attn, decoder_ff in self.decoder:
            x = mask_attn(self.norm(x), ids_mask) + x
            x = decoder_attn(self.norm(x), encoder_out, pos) + x
            x = decoder_ff(self.norm(x)) + x
        return encoder_out, x
    

class ViTIds(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, ids_class, dim, depth, heads, mlp_dim, 
                 pool='mean', channels=1200, dim_head=64, max_seq_len=50, dropout=0., emb_dropout=0.):
        super().__init__()       
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches equal the seqLen
        self.heigh_num = image_height // patch_height
        self.width_num = image_width // patch_width
        num_patches = self.heigh_num * self.width_num
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        dim_2 = dim // 2
        self.encoder_pos_embedding = TwoDimPositionEmbedding(dim_2, self.heigh_num, self.width_num, normalize=True)
        self.decoder_embed = Embedding(ids_class, dim)
        self.decoder_pos_embedding = PositionalEncoder(dim, max_seq_len)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.spa = SpatialAttention()
        self.cfr = CFR(dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.global_classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.ids_classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ids_class)
        )
        self.softmax = nn.Softmax(dim=2)

        # Summary the params.
        print("-"*60)
        print("Init I/O")
        print("SeqLen: {0}; Patch dim (N^2*C): {1}; Embedding dim: {2}; Input Dropout: {3}; Out cls: {4}; IDS cls {5}\n".format(num_patches, patch_dim, dim, emb_dropout, num_classes, ids_class))
        print("Init ViT:")
        print("Encoder Layer num: {0}; heads: {1}; dim in every head: {2}; FeedForward hidden dim: {3}; Dropout: {4}".format(depth, heads, dim_head, mlp_dim, dropout))
        print("Decoder Layer num: {0}; heads: {1}; dim in every head: {2}; FeedForward hidden dim: {3}; Dropout: {4}".format(depth, heads, dim_head, mlp_dim, dropout))
        print("-"*60)

    def forward(self, cnn_out, ids_seq, ids_mask, caption_lengths=[10,], train=True):
        # --------------------------------------------------
        # sort the input ids sequence, only in training
        if train:
            caption_lengths, sort_ind = caption_lengths.sort(descending=True)
            ids_seq = ids_seq[sort_ind]
            cnn_out = cnn_out[sort_ind]
        
        # --------------------------------------------------
        # IDS Decoder Embedding
        ids_embedding = self.decoder_embed(ids_seq)
        pos_decoder = self.decoder_pos_embedding(ids_seq)
        ids_embedding = ids_embedding + pos_decoder
        
        # Image Encoder feature Embedding
        x = self.to_patch_embedding(cnn_out)
        _, n, _ = x.shape
        pos_encoder = self.encoder_pos_embedding(x)[:, :n]

        # --------------------------------------------------
        global_feature, pred_ids = self.transformer(x, pos_encoder, ids_embedding, pos_decoder, ids_mask)

        #-----------------Global Feature Aggregation Module-------------------
        global_feature = rearrange(global_feature, 'b (h w) d -> b d h w', h=self.heigh_num, w=self.width_num)
        spa = self.spa(global_feature)
        global_feature = spa*global_feature
        global_feature = rearrange(global_feature, 'b d h w -> b (h w) d', h=self.heigh_num, w=self.width_num)
        global_feature = global_feature.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        global_feature = self.to_latent(global_feature)

        #-----------------Character Feature Reconstruction Module-------------------
        if train:
            char_reconstruct = self.cfr(pred_ids, caption_lengths)
        else:
            char_reconstruct = self.cfr(pred_ids, None, train=False)

        if train:
            # global_feature and char_reconstruct are used to calculate the align loss
            return self.global_classifier(global_feature), self.ids_classifier(pred_ids), global_feature,\
                   char_reconstruct, sort_ind
        else:
            return self.global_classifier(global_feature), self.ids_classifier(pred_ids)


class UCR(nn.Module):
    def __init__(self, backbone_name="densenet135_ucr", backbone_vit_name='vit_base', clsss_num=1000, 
                 ids_class=10, patch_size=1):
        super(UCR, self).__init__()
        
        self.cnn = self._cnn_model_select(backbone_name)
        self.vit = self._vit_model_select(backbone_vit_name, clsss_num, ids_class, patch_size)

    def _cnn_model_select(self, name):
        if name == 'densenet135_ucr':
            print("CNN is DenseNet135 UCR")
            cnn = densenet.densenet135_ucr()
        elif name == 'densenet157_ucr':
            print("CNN is DenseNet157 UCR")
            cnn = densenet.densenet157_ucr()
        else:
            raise ModuleNotFoundError
        return cnn

    def _vit_model_select(self, vit_name, clsss_num, ids_class, patch_size):
        if vit_name == 'vit_base':
            model = vit_base(image_size=8, patch_size=patch_size,
                             num_classes=clsss_num, ids_class=ids_class)
        elif vit_name == 'vit_large':
            model = vit_large(image_size=8, patch_size=patch_size,
                              num_classes=clsss_num, ids_class=ids_class)
        else:
            raise ModuleNotFoundError
        return model

    def forward(self, images, ids_seq, ids_mask, caption_lengths, train=True):
        cnn_out = self.cnn(images)

        if train:
            global_pred, local_pred, global_feature, char_reconstruct, sort_ind = \
                self.vit(cnn_out, ids_seq, ids_mask, caption_lengths, train=train)
            return global_pred, local_pred, global_feature, char_reconstruct, sort_ind
        else:
            global_pred, local_pred = self.vit(cnn_out, ids_seq, ids_mask, caption_lengths, train=train)
            return global_pred, local_pred


def vit_base(image_size=8, patch_size=1, num_classes=1000,
             dropout=0.1, emb_dropout=0.1, ids_class=100):
    layer_num = 12; hidden_size = 768; mlp_size = 3072; multi_heads = 12

    return ViTIds(image_size = image_size, patch_size = patch_size, num_classes = num_classes, 
            ids_class = ids_class, dim = hidden_size, depth = layer_num, heads = multi_heads, 
            mlp_dim = mlp_size, dropout = dropout, emb_dropout = emb_dropout)

def vit_large(image_size=8, patch_size=1, num_classes=1000,
             dropout=0.1, emb_dropout=0.1, ids_class=100):
    layer_num = 24; hidden_size = 1024; mlp_size = 4096; multi_heads = 16

    return ViTIds(image_size = image_size, patch_size = patch_size, num_classes = num_classes, 
            ids_class = ids_class, dim = hidden_size, depth = layer_num, heads = multi_heads, 
            mlp_dim = mlp_size, dropout = dropout, emb_dropout = emb_dropout)

def ucr_base(backbone_name='densenet135_ucr', backbone_vit_name='vit_base', clsss_num=1000, ids_class=10):
    return UCR(backbone_name, backbone_vit_name, clsss_num, ids_class)

def ucr_large(backbone_name='densenet157_ucr', backbone_vit_name='vit_base', clsss_num=1000, ids_class=10):
    return UCR(backbone_name, backbone_vit_name, clsss_num, ids_class)


def _create_masks(trg):
    # trg shape is [batch, seqLen]
    # 把非pad位全部置为1, 即True
    trg_mask = (trg != 0).unsqueeze(-2)
    size = trg.size(1) # get seq_len for matrix

    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    np_mask = np_mask.cuda()
    trg_mask = trg_mask.cuda()
    trg_mask = trg_mask & np_mask
    return trg_mask

def ucr_test():
    ucr = ucr_large(clsss_num=3580, ids_class=571).cuda()

    img = torch.randn(1, 3, 32, 32).cuda()
    ids_seq = torch.linspace(1, 100, steps=20).int().unsqueeze(0).cuda()
    caption_lengths = torch.Tensor([20,]).int()
    ids_masks = _create_masks(ids_seq)
    
    global_preds, local_preds, sort_ind = ucr(img, ids_seq, ids_masks, caption_lengths)
    print("global_preds shape is ", global_preds.shape)
    print("local_preds shape is ", local_preds.shape)


if __name__ == '__main__':
    ucr_test()
