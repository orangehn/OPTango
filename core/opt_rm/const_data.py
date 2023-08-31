import os
from typing import Optional

import torch
import torch.nn as nn

from core.opt_rm.dist_utils import Dist
from third.vit_pytorch.vit_pytorch.vit import Transformer
from core.opt_rm.utils import init_weight


class TransformerEmbedding(nn.Module):
    def __init__(self, max_num_token, num_embeddings, embedding_dim=768, with_token_e=True, with_pos_e=True):
        super().__init__()
        self.with_token_e = with_token_e
        self.with_pos_e = with_pos_e
        self.max_token_len = max_num_token
        if with_token_e:
            self.token_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        if with_pos_e:
            self.position_embedding = nn.Embedding(num_embeddings=max_num_token, embedding_dim=embedding_dim)
        self.transformer = Transformer(dim=embedding_dim, depth=1, heads=12, dim_head=64, mlp_dim=3072, dropout=0.1)

    def forward(self, token_ids):
        # token_ids: (..., max_len) / (..., max_len, emb_dim)
        # output: (..., emb_dim)
        if self.with_token_e:
            tokens_e = self.token_embedding(token_ids)
            token_ids_shape = token_ids.shape  # (..., max_len)
        else:
            tokens_e = token_ids
            token_ids_shape = token_ids.shape[:-1]  # (..., max_len)
        if self.with_pos_e:
            # positions: (..., max_len)
            positions = torch.arange(token_ids_shape[-1], device=token_ids.device).expand(token_ids_shape)
            tokens_e = tokens_e + self.position_embedding(positions)  # (..., max_len, emb_dim)
        tokens_e = tokens_e.reshape(-1, *tokens_e.shape[-2:])  # (..., max_len, emb_dim) => (b, max_len, emb_dim)
        tokens_e = self.transformer(tokens_e)  # (b, max_len, emb_dim)
        tokens_e = tokens_e.reshape(*token_ids_shape, tokens_e.shape[-1])  # (..., max_len, emb_dim)
        return tokens_e.mean(dim=-2)  # (..., emb_dim)


class FcEmbedding(nn.Module):
    def __init__(self, max_num_token, num_embeddings, embedding_dim=768):
        super().__init__()
        self.linear1 = nn.Linear(max_num_token, embedding_dim)
        self.norm1 = nn.LayerNorm(max_num_token)
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_ids):
        # token_ids: (..., max_len)
        x = self.norm1(token_ids.float())  # (..., max_len)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x  # (..., embed_dim)


class StrEmbedding(nn.Module):
    """
        given a str list
        1. pad or clip each str to max_len_str.
        2. map each char of str to ASCII code id, use code id map to embedding vector
        3. aggregate all embedding vectors to a single embedding vector with embed_cls
    """
    def __init__(self, embed_cls='TransformerEmbedding', embed_kwargs={}):
        super().__init__()
        self.max_len_str = embed_kwargs["max_num_token"]
        self.embedding = eval(embed_cls)(**embed_kwargs)

    @staticmethod
    def preprocess_str(strs, max_len=10):
        """
            strs: (b, num_data) list of str, list of list str
        """
        str_ids = []
        if isinstance(strs[0], (list, tuple)):
            masks = []
            for s in strs:
                s_ids, s_masks = StrEmbedding.preprocess_str(s, max_len)
                str_ids.append(s_ids)
                masks.append(s_masks)
        elif isinstance(strs[0], str):
            empty_len = [0] * max_len  # 0 for empty
            masks = np.zeros((len(strs), max_len))
            for i, s in enumerate(strs):
                ord_ids = [ord(e) for e in s[::-1]]  # reverse char of str
                str_ids.append((ord_ids + empty_len)[:max_len])
                masks[i, :len(ord_ids)] = 1
        else:
            raise TypeError
        return str_ids, masks

    def forward(self, strs, device):
        # (...) => # (..., max_len)
        str_ids, masks = StrEmbedding.preprocess_str(strs, self.max_len_str)
        str_ids = torch.tensor(str_ids).to(device)
        masks = torch.tensor(np.array(masks)).to(device)
        return self.embedding(str_ids)


class ConstStrEmbedding(StrEmbedding):
    """
    give a info list
    1. get const info from infos by key asm_const_src, clip or pad it's length to max_num_str
    2. get data and position idx from each info, turn data to str
    3. use StrEmbedding turn each data str to embedding vec
    """
    def __init__(self, asm_str_src, max_num_str=20, embed_cls='TransformerEmbedding', embed_kwargs={}):
        super().__init__(embed_cls, embed_kwargs)
        disasm_name = asm_str_src.split('_')[0]
        self.asm_const_src = disasm_name + "_asm_consts"
        self.max_num_str = max_num_str

    @staticmethod
    def preprocess_strs(infos, asm_const_src, max_num_str):
        info_const_data = [info[asm_const_src] for info in infos]

        max_empty_data = [('\x00', 0, 0)] * max_num_str  # ord(str('\x00')) == ord('\x00') == 0
        masks = np.zeros((len(info_const_data), max_num_str))
        for i, ds in enumerate(info_const_data):  # (b, max_str_len, 3)
            masks[i, :len(ds)] = 1
        info_const_data_pad = [(ds + max_empty_data)[:max_num_str] for ds in
                               info_const_data]  # (b, max_str_len, 3)
        const_str = [[str(d[0]) for d in ds] for ds in info_const_data_pad]  # (b, max_str_len)
        const_pos = [[d[1] for d in ds] for ds in info_const_data_pad]  # (b, max_str_len)
        return const_str, const_pos, masks

    def forward(self, infos, device):
        """
        return: const_str_emb, const_pos_ids, atte_masks
        """
        const_strs, const_pos_ids, masks = ConstStrEmbedding.preprocess_strs(infos, self.asm_const_src, self.max_num_str)
        str_embeds = super(ConstStrEmbedding, self).forward(const_strs, device)
        return str_embeds, torch.tensor(const_pos_ids).to(device), torch.tensor(masks).to(device)


class ConstDataBlock(nn.Module):

    def __init__(self, asm_str_src,
                 # num_group=2, num_const=5, const_embed_dim=2, num_embed=512, out_dim=768,
                 str_embed_cls='TransformerEmbedding',
                 str_embed_kwargs=dict(max_num_token=10, num_embeddings=256, embedding_dim=768),
                 str_list_embed_cls='TransformerEmbedding',
                 str_list_embed_kwargs=dict(
                     max_num_token=20, num_embeddings=256, embedding_dim=768, with_token_e=False, with_pos_e=False),
                 out_type='const_data'):
        super().__init__()
        self.asm_str_src = asm_str_src
        disasm_name = asm_str_src.split('_')[0]
        self.asm_const_src = disasm_name + "_asm_consts"
        self.out_type = out_type

        if self.out_type.startswith('const_emb'):
            max_num_str = str_list_embed_kwargs.get("max_num_token", 20)
            self.str_embedding = ConstStrEmbedding(
                asm_str_src, max_num_str, embed_cls=str_embed_cls, embed_kwargs=str_embed_kwargs)
            self.str_list_embedding = eval(str_list_embed_cls)(**str_list_embed_kwargs)
        elif self.out_type == 'const_data':
            pass
        else:
            raise ValueError()

    def init_weight(self):
        init_weight(self)

    def forward(self, fn_e, infos, fn_token_es=None):
        """
        """
        if self.out_type == 'const_data':
            info_const_data = [info[self.asm_const_src] for info in infos]
            return fn_e, {'const_f': info_const_data}

        device = fn_e.device
        const_str_emb, const_pos, masks = self.str_embedding(infos, device)
        if fn_token_es is not None:  # (b, seq_len, emb_dim)
            b_idx = torch.arange(const_pos.shape[1]).unsqueeze(dim=0)  # (1, max_str_len)
            left_pos = (const_pos - 1).clamp(0., fn_token_es.shape[1])
            right_pos = (const_pos + 1).clamp(0., fn_token_es.shape[1])
            neighbor_emb = fn_token_es[b_idx, left_pos] + fn_token_es[b_idx, right_pos]  # (b, max_str_len, emb_dim)
            const_str_emb = (const_str_emb + neighbor_emb) / 3
        const_e = self.str_list_embedding(const_str_emb.relu())  # (b, emb_dim)
        if self.out_type == 'const_emb':
            return fn_e, {"const_f": const_e}
        elif self.out_type == 'const_emb:add':
            return fn_e + const_e, {"const_f": const_e}
        else:
            raise ValueError


from third.jTrans.data_func import gen_funcstr
from math import ceil, floor
import pickle
import difflib
import numpy as np


def is_none_list(alist):
    for a in alist:
        if a is not None:
            return False
    return True


class ConstDataHelper(object):
    # scripts/analysis/const_same_json.py
    ida_feat_dir = 'data/data-bsca/feat/jtrans_ida_feat_split/'
    max_len_const_f = 0

    @staticmethod
    def get_const_root_dir(bin_feat_dir):
        es = [e.strip() for e in bin_feat_dir.split('/') if len(e.strip()) > 0]
        bench, opt, bin_name = es[-3:]
        feat_dir = '/'.join(es[:-3])
        return ConstDataHelper.get_const_dir2(feat_dir)

    @staticmethod
    def get_const_dir(bin_feat_dir, const_root_dir=None):
        es = [e.strip() for e in bin_feat_dir.split('/') if len(e.strip()) > 0]
        bench, opt, bin_name = es[-3:]
        feat_dir = '/'.join(es[:-3])
        const_root_dir = ConstDataHelper.get_const_dir2(feat_dir, const_root_dir)
        return os.path.join(const_root_dir, bench, opt, bin_name)

    @staticmethod
    def get_const_dir2(feat_dir, const_root_dir=None):
        if const_root_dir is not None:
            return const_root_dir
        es = [e.strip() for e in feat_dir.split('/') if len(e.strip()) > 0]
        es[-1] = es[-1] + "_const"
        return "/".join(es)

    @staticmethod
    def clip_const_data(const_fs, max_len=20):
        len_const_f = max([len(const_f) for const_f in const_fs])
        if len_const_f > ConstDataHelper.max_len_const_f:
            ConstDataHelper.max_len_const_f = len_const_f
            Dist.print(f"[const_data]: max const data length = {len_const_f}, clip to max_len={max_len}")
        return [const_f[:max_len] for const_f in const_fs]

    @staticmethod
    def const_match(similar_matrix, query_meta, index_meta, max_const_len=20, eps=1e-8):
        if 'const_f' not in query_meta or 'const_f' not in index_meta:
            return similar_matrix
        if is_none_list(query_meta['const_f']) or is_none_list(index_meta['const_f']):  # do not use const_match
            return similar_matrix
        assert (len(query_meta['const_f']), len(index_meta['const_f'])) == similar_matrix.shape

        Dist.print(f"[const_data]: evaluate with const_data")
        if isinstance(query_meta['const_f'][0], torch.Tensor) and isinstance(index_meta['const_f'][0], torch.Tensor):
            raise NotImplementedError
        elif isinstance(query_meta['const_f'][0], list) and isinstance(index_meta['const_f'][0], list):
            max_score, max_index_idx = similar_matrix.max(dim=-1)
            tie_topk = (similar_matrix - max_score[:, None]).abs() < eps
            num_tie_topk = tie_topk.sum(dim=-1)

            # const_length limit
            query_meta['const_f'] = ConstDataHelper.clip_const_data(query_meta['const_f'], max_const_len)
            index_const_fs = index_meta['const_f']
            for q_idx, q_const_f in enumerate(query_meta['const_f']):
                if num_tie_topk[q_idx] <= 1:
                    continue
                tie_topk_indexs = torch.where(tie_topk[q_idx])[0]
                tie_topk_index_const_fs = [index_const_fs[i_idx] for i_idx in tie_topk_indexs]
                # const_length limit
                tie_topk_index_const_fs = ConstDataHelper.clip_const_data(tie_topk_index_const_fs, max_const_len)

                if len(q_const_f) == 0:
                    diff_s = [1 if len(i_const_f) == 0 else 0 for i_const_f in tie_topk_index_const_fs]
                else:
                    diff_s = []
                    for i_const_f in tie_topk_index_const_fs:
                        score = 0 if len(i_const_f) == 0 else ConstDataHelper.diff_score(q_const_f, i_const_f)
                        diff_s.append(score)
                diff_s = 0.5 + torch.tensor(diff_s) / 2  # (0.5, 1.0)  # take there are no-zero score for empty const
                min_diff_s = diff_s.min()
                similar_matrix[q_idx] *= min_diff_s  # diff_s.min() * cos_similarity
                similar_matrix[q_idx, tie_topk_indexs] = diff_s  # diff_s
        else:
            raise TypeError(f"{query_meta['const_f'][0]} and {index_meta['const_f'][0]}")
        return similar_matrix

    @staticmethod
    def load_asm_str(ida_feat_dir, full_offset):
        bob, offset = full_offset[:3], full_offset[3]  # (bench, opt, bin_name, offset)
        pkl_data = os.path.join(ida_feat_dir, '/'.join(bob), f'{hex(offset)}.pkl')
        #     print(bob, hex(offset))
        #     if not os.path.exists(pkl_data):
        #         return None
        data = pickle.load(open(pkl_data, 'rb'))
        # asm = data[1]
        #     print(asm)
        func_str, info = gen_funcstr(data, True, with_info=True)
        return func_str.strip(), info

    @staticmethod
    def diff_score(const_data1, const_data2):
        list1 = [d[0] for d in const_data1]
        list2 = [d[0] for d in const_data2]
        if len(list1) == 0 and len(list2) == 0:
            return 1.
        diff = list(difflib.ndiff([str(e) for e in list1], [str(e) for e in list2]))
        I = [e for e in diff if e.startswith(' ')]
        return len(I) / len(diff)
        # return (len(list1) + len(list2) - len(diff)) / len(diff)

    # wrong_annotation


if __name__ == '__main__':
    CH = ConstDataHelper

    # dataset
    # _, info1 = CH.load_asm_str(CH.ida_feat_dir, (""))
    # _, info2 = CH.load_asm_str(CH.ida_feat_dir, (""))

    cb = ConstDataBlock('ida_asm_str', out_type='const_emb')
    cb.eval()
    x = cb(torch.zeros(2, 768), [
        {"ida_asm_consts": [(100, 10, 0), (1, 12, 1)]},
        {"ida_asm_consts": [(100, 10, 0), (1, 12, 1)]}]
           )
    print(x)
    # f, others = cb(torch.zeros(2, 768), [info1, info2])
    # const_f = others['const_f']
    #
    # CH.diff_score(info1['consts'], info2['consts'])
