import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch

from core.opt_rm.model import OptRemoveBertModel
from third.jTrans.data_func import gen_funcstr
import time
import argparse


def cos_similarity_(v1, v2):
    return (v1 @ v2.T / (v1.norm(dim=-1)[:, None] * v2.norm(dim=-1)[None, :]))


def cosine_similarity(v1, v2, device=0):
    # with torch.no_grad():
    #     step1, step2 = 80000, 10000
    #     result = []
    #     for s1 in range(0, len(v1), step1):
    #         res = []
    #         for s2 in range(0, len(v2), step2):
    #             s = cos_similarity_(v1[s1:s1 + step1].clone().to(device), v2[s2:s2 + step2].clone().to(device))
    #             res.append(s.detach().cpu())
    #         result.append(torch.cat(res, dim=1).detach().cpu())
    #     return torch.cat(result, dim=0)
    return cos_similarity_(v1, v2)


def remove_parallel_prefix(checkpoint):
    if 'model' in checkpoint:
        ws = {}
        for k, w in checkpoint['model'].items():
            if k.startswith("module."):
                k = k[len("module."):]
                ws[k] = w
        checkpoint['model'] = ws
    else:
        for mk, mv in checkpoint.items():
            ws = {}
            for k, w in mv.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                    ws[k] = w
            checkpoint[mk] = ws
    return checkpoint


class FunctionEmbeddingSet(object):
    def __init__(self, opt_to_bin_pkl_paths, limit=None):
        self.func_embeddings, self.func_names = {}, {}
        for opt, bin_pkl_paths in opt_to_bin_pkl_paths.items():
            self.func_embeddings[opt] = []
            self.func_names[opt] = []
            for pkl_path in bin_pkl_paths:
                pkl = pickle.load(open(pkl_path, 'rb'))
                if 'embedding' in pkl:
                    self.func_embeddings[opt].append(pkl['embedding'])
                    [self.func_names[opt].append((pkl_path, fun_name)) for fun_name in pkl['name']]
                else:
                    feats = []
                    for offset, (offset, feat, hash, fun_name) in pkl.items():
                        feats.append(feat)
                        self.func_names[opt].append((pkl_path, fun_name))
                    self.func_embeddings[opt].append(torch.stack(feats))
            self.func_embeddings[opt] = torch.cat(self.func_embeddings[opt])

        self.all_func_names = []
        for opt in self.func_embeddings:
            self.all_func_names.extend(self.func_names[opt])

    def query(self, feat, opt=None):
        if opt is None:
            scores = []
            for opt in self.func_embeddings:
                score = cosine_similarity(feat.unsqueeze(dim=0), self.func_embeddings[opt])[0]
                scores.append(score)
            scores = torch.cat(scores)
            max_id = scores.argmax()
            return self.all_func_names[max_id]
        else:
            scores = cosine_similarity(feat.unsqueeze(dim=0), self.func_embeddings[opt])[0]
            max_id = scores.argmax()
            return self.func_names[opt][max_id]

    def __len__(self):
        return len(self.all_func_names)


class FullModel(object):
    def __init__(self, device="cuda:0", with_gp=True, checkpoint_dir="model_weight/"):
        checkpoint = torch.load(f"{checkpoint_dir}/model_release.pt", map_location=device)
        checkpoint = remove_parallel_prefix(checkpoint)

        model = OptRemoveBertModel(feat_source='opt_rm').to(device)
        model.load_state_dict(checkpoint["bert"])
        model.eval()
        self.model_bert = model

        model = OptRemoveBertModel(feat_source='bsc', sub_modules="const_data",
                                   const_data_kwargs=dict(out_type='const_emb:add')).to(device)
        model.load_state_dict(checkpoint["const"])
        model.eval()
        self.model_const_data = model

        if with_gp:
            model = OptRemoveBertModel(feat_source='bsc', sub_modules='group_pred').to(device)
            model.load_state_dict(checkpoint["group_pred"])
            model.eval()
            self.model_group_pred = model
        self.with_gp = with_gp

        self.bsc_feat = torch.zeros(0, ).to(device)  # just for given device, not used
        self.t = {"gp": 0}

    def __call__(self, infos):
        with torch.no_grad():
            feats, other_out = self.model_bert(self.bsc_feat, None, None, infos)
            feats, other_out = self.model_const_data(feats, None, None, infos)
            tic = time.time()
            if self.with_gp:
                feats, other_out = self.model_group_pred(feats, None, None, None)
                other_out["pred_opt"] = [self.model_group_pred.group_predictor.class_names[i] for i in other_out["group_pred_idx"]]
            toc = time.time()
            self.t['gp'] += toc - tic
            return feats, other_out


def load_funcs(ida_path):
    func_names = []
    func_infos = []
    data = pickle.load(open(ida_path, 'rb'))
    for func_name, fun_info in data.items():
        asm_str, asm_info = gen_funcstr(fun_info, convert_jump=True, with_info=True)
        func_names.append(func_name)
        info = {
            "ida_asm_str": asm_str,
            'ida_asm_consts': asm_info['consts'],
        }
        func_infos.append(info)
    return func_names, func_infos


def extract_embeddings(model, func_infos):
    feats, pred_outs = [], []
    for f_info in tqdm(func_infos):
        _feats, other_out = model([f_info])
        pred_out = other_out['pred_opt'][0] if 'pred_opt' in other_out else None
        feats.append(_feats[0].cpu())
        pred_outs.append(pred_out)
    return torch.stack(feats), pred_outs


def matched_gt(func_names1, func_names2):
    match = [-1] * len(func_names1)
    for i1, f1 in enumerate(func_names1):
        for i2, f2 in enumerate(func_names2):
            if f1 == f2:
                match[i1] = i2
                break
    return torch.tensor(match)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-gp", default='True')
    args = parser.parse_args()
    print(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = FullModel(device, with_gp=eval(args.with_gp))

    # 1. load asm info extracted by ida of two compiler option
    ida_dir = 'data/data-bsca/feat/ida_feat/'
    ida_path = os.path.join(ida_dir, 'openssl', 'gcc-O2', 'openssl_extract.pkl')
    func_names, func_infos = load_funcs(ida_path)
    f1_name, f1_info = func_names[1], func_infos[1]   # get the first function

    ida_path = os.path.join(ida_dir, 'openssl', 'gcc-O3', 'openssl_extract.pkl')
    func_names2, func_infos2 = load_funcs(ida_path)
    f2_names, f2_infos = func_names2[1:10], func_infos2[1:10]   # get first 10 function

    # 2. key step: extract function embedding for given functions
    f1_feat, _ = extract_embeddings(model, [f1_info])
    f2_feats, _ = extract_embeddings(model, f2_infos)

    # 3. calculate similarity score
    similarity = cosine_similarity(f1_feat, f2_feats)
    for i, s in enumerate(similarity[0].cpu().numpy()):
        print(f'cosine similarity({f1_name}, {f2_names[i]}) = {s}')


main()
