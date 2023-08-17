import os
import pickle
from collections import defaultdict

import torch

from core.opt_rm.dist_utils import Dist
from core.opt_rm.sampler import OptGroupSampler
from core.opt_rm.loss import cosine_similarity
from tqdm import tqdm

from package.simple_utils.simple_log import printl


class OptGroupManager(object):
    # templates_name = ["gcc-O0", "gcc-O1", "gcc-O2", "gcc-O3"]
    # templates_name = ["gcc-O0", "gcc-O1", "gcc-O3"]
    # please change in framework args -ogt
    templates_name = ["gcc-O0", "gcc-O2"]  # ["gcc-O0", "gcc-O2"]

    def __init__(self, dataset, batch_size, device, model, re_cal_tid_epoch=1, start_after_epoch=1, group_iters=[], **group_kwargs):
        self.re_cal_tid_epoch = re_cal_tid_epoch
        self.device = device
        self.model = model
        self.start_after_epoch = start_after_epoch
        self.group_iters = group_iters
        self.group_kwargs = group_kwargs

        self.dataset = dataset
        self.fid2template_data_idxs = defaultdict(list)
        self.fid2template_tids = defaultdict(list)
        self.fid2data_idxs = defaultdict(list)
        for data_idx, fid in enumerate(dataset.data_infos["fid"]):
            base_tid = dataset.data_infos["base_tid"][data_idx]
            # if base_tid >= 0:  # template data
            #     self.fid2data_idxs[fid][base_tid] = data_idx
            # else:   # non-template data
            self.fid2data_idxs[fid].append(data_idx)
            if base_tid >= 0:
                self.fid2template_data_idxs[fid].append(data_idx)
                self.fid2template_tids[fid].append(base_tid)

        no_template_count = 0
        for fid, data_idxs in self.fid2data_idxs.items():
            t_data_idxs = self.fid2template_data_idxs[fid]
            if len(t_data_idxs) == 0:
                no_template_count += 1
        printl(f'[opt_group] {no_template_count} fids have no templates')

        self.feats_cache = {}

        # # for re-cal tid
        # opt_group_sampler = Dist.sampler(OptGroupSampler(dataset), dataset)
        #
        # # 保证每条数据有且仅有一次访问
        # self.iter_data_loader = torch.utils.data.DataLoader(
        #     dataset, batch_size, num_workers=0, collate_fn=dataset.collate_fn, sampler=opt_group_sampler)

    def extract_feat(self):
        self.model.eval()
        M = len(self.iter_data_loader.dataset)
        features, features_fid, features_base_tid = [None] * M, [None] * M, [None] * M
        fid2template_data_idxs = defaultdict(list)
        fid2template_tids = defaultdict(list)
        fid2data_idxs = defaultdict(list)
        for data in tqdm(self.iter_data_loader, desc="group"):
            with torch.no_grad():
                # 1. forward model extract feat
                data_idxs, bsc_feat, fids, infos, asm_path, asm_token_idx, label, coarse_label, x_weight, graph_dict = data
                graph_dict = {key: graph_dict[key].to(self.device) for key in graph_dict}
                bsc_feat = bsc_feat.to(self.device)
                asm_token_idx = asm_token_idx.to(self.device)
                feats = self.model(bsc_feat, asm_token_idx, graph_dict, infos)
                feats = feats.detach().cpu()

                # 2. save feat and tid from info['base_tid']
                fids = fids.detach().cpu().numpy().tolist()
                for i, idx in enumerate(data_idxs):
                    fid, base_tid = fids[i], infos[i]["base_tid"]
                    features[idx] = feats[i]
                    features_fid[idx] = fid
                    features_base_tid[idx] = base_tid
                    fid2data_idxs[fid].append(idx)
                    if base_tid >= 0:
                        fid2template_data_idxs[fid].append(idx)
                        fid2template_tids[fid].append(base_tid)
        self.model.train()

    def load_features(self, data_idxs, feat_dir):
        feats = []
        for data_idx in data_idxs:
            if data_idx in self.feats_cache:
                feat = self.feats_cache[data_idx]
            else:
                bench, opt, bin_name, offset = self.dataset.data_infos['full_offset'][data_idx]
                feat_pkl = os.path.join(feat_dir, bench, opt, bin_name, f"{hex(offset)}.pkl")
                offset, feat, hash, name = pickle.load(open(feat_pkl, 'rb'))
                self.feats_cache[data_idx] = feat
            feats.append(feat)
        return torch.stack(feats, dim=0)

    def get_template_scores(self, feat_dir=None):
        assert feat_dir is not None, ""
        Dist.print(f"[opt_group] load feat from {feat_dir}.")
        # template_scores, is_which_template = , []
        # 3. calculate cosine similarity of each function
        num_t = len(OptGroupManager.templates_name)
        score_map = torch.empty((len(self.dataset), num_t))
        score_map[:] = 1. / num_t  # init to average

        for fid, data_idxs in tqdm(self.fid2data_idxs.items(), desc='group'):
            t_data_idxs = self.fid2template_data_idxs[fid]
            if len(t_data_idxs) == 0:
                continue
            feats = self.load_features(data_idxs, feat_dir)
            t_feats = self.load_features(t_data_idxs, feat_dir)
            self.feats_cache = {}  # clear feat_cache
            cos = cosine_similarity(feats.to(self.device), t_feats.to(self.device))
            scores = ((cos + 1) / 2).detach().cpu().clone()
            tids = self.fid2template_tids[fid]
            for i, data_idx in enumerate(data_idxs):
                score_map[data_idx, tids] = scores[i]

        base_tid_map = torch.tensor(self.dataset.data_infos["base_tid"], dtype=torch.short)
        return score_map, base_tid_map

    def calculate_tid(self, score_map):
        method = self.group_kwargs.get('group_method', 'max')
        if method == 'max':
            return score_map.max(dim=-1)[1]
        elif method == 'balance':
            num_group = score_map.shape[1]
            assert num_group == len(OptGroupManager.templates_name)
            assert num_group == 2, "only support 2 group for now"
            tids = torch.zeros(score_map.shape[0], device=score_map.device, dtype=torch.long)
            # sort by score of second group
            _, idx = torch.sort(score_map[:, 1], descending=True)
            idx = idx[:int(len(score_map)/num_group)]
            tids[idx] = 1
            return tids
        elif method == 'prob':
            pass
        else:
            raise ValueError("")
