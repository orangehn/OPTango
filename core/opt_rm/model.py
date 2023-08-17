import os
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
from core.opt_rm.opt_group import OptGroupManager

from core.asm2vec.datatype import Tokens
from core.opt_cls.model import FuncEmbedding
from core.opt_rm.dist_utils import Dist
from package.embedding_plus.CombineEmbedding import CombineEmbedding
from core.opt_rm.loss import TripletLoss
from package.simple_utils.simple_log import printl
from core.opt_rm.const_data import ConstDataBlock, StrEmbedding, ConstStrEmbedding
from core.opt_rm.utils import init_weight
from core.opt_rm.bin_bert_model import BinBertModel, BertTokenizer


def load_tokens(path, device='cpu'):
    import core.asm2vec as asm2vec
    checkpoint = torch.load(path, map_location=device)
    tokens = Tokens()
    tokens.load_state_dict(checkpoint['tokens'])
    return tokens


class OptRemoveModel(nn.Module):
    def __init__(self, tokens, embedding_dim, bsc_feat_dim, feat_source, **func_args):
        super().__init__()
        token_embeddings = CombineEmbedding(
            len(tokens.atom_name_to_atom_index) + 1, tokens.index_to_atom_index, embedding_dim,
            _weight=torch.randn(len(tokens.atom_name_to_atom_index) + 1, embedding_dim)
        )
        self.func_embeddings = FuncEmbedding(token_embeddings, **func_args)
        func_embedding_dim = embedding_dim * 2

        self.other_loss = 0.
        self.fc = nn.Linear(func_embedding_dim, bsc_feat_dim)
        self.loss_func = TripletLoss(margin=None, mid_weight=0.)

        self.post_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bsc_feat_dim, bsc_feat_dim//2),
            nn.Tanh(),
            nn.Linear(bsc_feat_dim//2, bsc_feat_dim)
        )

        self.feat_source = [e.strip() for e in feat_source.split(",") if len(e.strip()) > 0]
        for fs in self.feat_source:
            assert fs in ["bsc", "opt_rm"], (feat_source, fs)

    def init_weight(self):
        # for param in self.parameters():
        #     nn.init.normal_(param, std=0.01)  # std too big will cause inf and nan
        self.func_embeddings.init_weight()   #
        init_weight(self)

    def combine_feat(self, fn_embedding, bsc_feat):
        w1 = 1 if "opt_rm" in self.feat_source else 0
        w2 = 1 if "bsc" in self.feat_source else 0
        return w1 * fn_embedding + w2 * bsc_feat

    def forward(self, bsc_feat, asm_tokens, graph_dict, infos=None, fids=None, tids=None, coarse_labels=None):
        if self.training:
            return self.loss(bsc_feat, asm_tokens, graph_dict, infos, fids, tids, coarse_labels)
        fn_embedding, _, _, _ = self.func_embeddings(asm_tokens, graph_dict)
        # do some solving, add a fc
        fn_embedding = self.fc(fn_embedding)
        feat = self.combine_feat(fn_embedding, bsc_feat)
        return self.post_block(feat)

    def forward2(self, bsc_feat, asm_tokens, graph_dict, infos=None):
        fn_embedding, _, _, _ = self.func_embeddings(asm_tokens, graph_dict)
        # do some solving, add a fc
        fn_embedding = self.fc(torch.relu(fn_embedding))
        feat = self.combine_feat(fn_embedding, bsc_feat)
        return self.post_block(feat), fn_embedding, bsc_feat

    def loss(self, bsc_feat, asm_tokens, graph_dict, infos, fids, tids, coarse_labels):
        batch_feat, f1, f2 = self.forward2(bsc_feat, asm_tokens, graph_dict, infos)
        loss, dist_ap, dist_am, dist_an = self.loss_func(batch_feat, fids, tids, normalize_feature=True)
        norm1, norm2 = torch.norm(f1, p=2, dim=-1).mean(), torch.norm(f2, p=2, dim=-1).mean()
        return {"tri_loss": loss, "dist_ap": dist_ap.mean(), "dist_am": dist_am.mean(), "dist_an": dist_an.mean(),
                "norm1": norm1, "norm2": norm2}

    # def loss(self, anchor, pos, neg, margin=1.0):
    #     anchor_feat = self(*anchor)  # (num_func, feat_dim)
    #     pos_feat = self(*pos)  # (num_func, )
    #     neg_feat = self(*neg)
    #
    #     loss = - cosine_similarity(anchor_feat, pos_feat) + cosine_similarity(anchor_feat, neg_feat) + margin
    #     return (loss > 0).float() * loss


class DynamicFc(nn.Module):
    """
    dynamic fc: usage for network that parameter need dynamic changed
    """
    def __init__(self, pf_dim, f_dim, low_dim=128, f_mid_dim=None):
        """
        pg:    d * l * l // 4 * 2
        proj: d * l * 3

        1. no proj: d * d * d // 2 = d**3//2
        2. with proj: l ** 3 // 2 + d * l * 3 = d**3 // 2(k**3) + d*d/k*3
           d = k*l
           for low_dim = 128 => k = 6, the equation = d**3 // (2 * 216) + d*d/6*3  ~= d*d*2
        """
        super().__init__()

        self.proj_f = nn.Linear(f_dim, low_dim)
        self.proj_pf = nn.Linear(pf_dim, low_dim)
        self.proj_f2 = nn.Linear(low_dim, f_dim)

        if f_mid_dim is None:
            f_mid_dim = low_dim // 4
        self.param_generator = nn.Linear(low_dim, low_dim*f_mid_dim + f_mid_dim*low_dim)

        self.f_mid_dim = f_mid_dim
        self.low_dim = low_dim

    def init_weight(self):
        init_weight(self)

    def forward(self, f, pf):
        """
        pf(b, d) => p (b, d*d//4+d//4*d) => p1 (b, d, d//4), p2 (b, d//4, d)
        f (b, d) => f (b, 1, d) @ p1 => (b, 1, d//4) @ p2 => (b, 1, d)
        """
        assert f.shape[0] == pf.shape[0]
        proj_f = self.proj_f(f).unsqueeze(dim=1)                   # (b, 768) => (b, 1, 128)
        proj_pf = self.proj_pf(pf)

        batch_size = pf.shape[0]
        params = self.param_generator(proj_pf)                     # (b, 128) => (b, 128 * 32 + 32 *128)
        dim = self.low_dim * self.f_mid_dim
        params1 = params[:, :dim].reshape(batch_size, self.low_dim, self.f_mid_dim)
        params2 = params[:, dim:].reshape(batch_size, self.f_mid_dim, self.low_dim)
        # (b, 1, 128) => (b, 1, 32) => (b, 1, 128)
        proj_f = torch.bmm(torch.relu(torch.bmm(proj_f, params1)), params2)
        f = self.proj_f2(proj_f.squeeze(dim=1)) + f + pf
        return f


class OBaseClassifier(nn.Module):
    """
    对输入的特征进行分类
    """
    def __init__(self, num_features, class_names, tid_w=0.5, coarse_w=0.5):
        super().__init__()
        self.class_names = class_names
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, len(class_names))
        self.softmax = nn.Softmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()
        assert abs(tid_w + coarse_w - 1) < 1e-6, "tid + coarse_w must equal 1.0"
        self.tid_w = tid_w
        self.coarse_w = coarse_w

        self.coarse_label_to_cid = self.get_OBase_to_tid_map()

    def get_OBase_to_tid_map(self):
        """
        [O0, O2], [O0, O3] => [(O0, O1), (O2, O3)] => low, high
        [O0, O1, O2], [O0, O1, O3] => [O0, O1, (O2, O3)]
        """
        coarse_label_to_cid = torch.tensor([-1] * 4, dtype=torch.long)
        opt_priority_map = {0: [0], 1: [1, 0], 2: [2, 3], 3: [3, 2]}
        for i in range(4):
            for j in opt_priority_map[i]:
                name = f"gcc-O{j}"
                if name in self.class_names:
                    coarse_label_to_cid[i] = self.class_names.index(name)
        for cid in coarse_label_to_cid:
            assert cid >= 0, self.class_names
        return coarse_label_to_cid

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def loss(self, x, tids, coarse_labels):
        """
        x: [batch_size, num_classes]
        y：[batch_size]
        """
        x = self(x)
        loss = {}
        if self.tid_w > 0 and tids is not None:
            tid_loss = self.loss_func(x, tids) * self.tid_w
            loss["Obase_tid_loss"] = tid_loss
        if self.coarse_w > 0:
            self.coarse_label_to_cid = self.coarse_label_to_cid.to(coarse_labels.device)
            coa_tids = self.coarse_label_to_cid[coarse_labels]
            loss["Obase_coa_loss"] = self.loss_func(x, coa_tids) * self.coarse_w
        return loss


class GroupPredictor(nn.Module):
    def __init__(self, num_features, class_names, **kwargs):
        super().__init__()
        self.class_names = class_names
        self.fc1 = nn.Linear(num_features, num_features*2)
        self.fc2 = nn.Linear(num_features*2, len(class_names))
        self.softmax = nn.Softmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()

    def init_weight(self):
        init_weight(self)

    def forward(self, x, infos):
        """
        x: (b, feat_dim)
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def loss(self, x, infos, tid):
        """
         x: (b, feat_dim)
         tid: (b,)
        """
        losses = {}
        x = self(x, infos)
        result = x.argmax(dim=-1)
        if tid is not None:
            # losses["gp_true"] = (result.long() == tid.long()).float().sum().item()
            # losses["gp_sum"] = len(result)
            losses["gp_acc"] = (result.long() == tid.long()).float().mean().item()
            losses["gp_loss"] = self.loss_func(x, tid.long())
        return {"group_pred_idx": result}, losses

    def predict_or_loss(self, x, infos, tids=None):
        if self.training:
            return self.loss(x, infos, tids)

        x = self(x, infos)
        result = x.argmax(dim=-1)
        return {"group_pred_idx": result}, {}


class OptRemoveBertModel(nn.Module):
    def __init__(self, *args, init_method='j-pretrain', lr_factors=None, wd_factors=None, freeze_cnt=-1, freeze_emb=False,
                 bsc_feat_dim=768, feat_source="opt_rm, bsc", with_post_block=False, opt_rm_method='dynamic_fc',
                 asm_str_src='ida_asm_str',
                 sub_modules="bert",
                 with_const_data=False, with_data_embed_in_bert=False, const_data_kwargs={},
                 obase_cls_kwargs=None, group_pred_kwargs={},
                 tokens=None, embed_dim=None, graph=None  # for general API
                 ):
        super().__init__()
        self.init_method = init_method
        self.lr_factors = lr_factors
        self.wd_factors = wd_factors
        self.freeze_cnt = freeze_cnt
        self.freeze_emb = freeze_emb
        self.asm_str_src = asm_str_src
        self.sub_module_names = set([m.strip() for m in sub_modules.split(',') if len(m.strip()) > 0])  # 'bert,const_data,group_pred'
        if with_const_data:
            self.sub_module_names.add("const_data")

        self.feat_source = [e.strip() for e in feat_source.split(",") if len(e.strip()) > 0]
        for fs in self.feat_source:
            assert fs in ["bsc", "opt_rm"], (feat_source, fs)
        self.with_post_block = with_post_block
        self.opt_rm_method = opt_rm_method

        feat_dim = bsc_feat_dim

        if 'bert' in self.sub_module_names:
            if init_method in ['scratch', 'j-pretrain']:
                jTrans_path = os.path.join(os.path.split(__file__)[0], "../../projects/jTrans")
                self.tokenizer = BertTokenizer.from_pretrained(os.path.join(jTrans_path, "./jtrans_tokenizer/"))
                init_path = os.path.join(jTrans_path, "./models/jTrans-pretrain")
                self.bert = BinBertModel.from_pretrained(init_path)
            else:
                init_path = init_method   # "bert-base-uncased"
                self.tokenizer = BertTokenizer.from_pretrained(init_path)
                self.bert = BinBertModel.from_pretrained(init_path)
            if with_data_embed_in_bert:
                self.bert.build_data_embedding(asm_str_src, **const_data_kwargs)

            # post_forward
            if with_post_block:
                self.post_block = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(feat_dim, feat_dim // 2),
                    nn.Tanh(),
                    nn.Linear(feat_dim // 2, feat_dim)
                )
            if len(self.feat_source) > 1 and opt_rm_method == 'dynamic_fc':
                self.opt_rm_block = DynamicFc(feat_dim, feat_dim)
        else:
            assert 'opt_rm' not in self.feat_source, "when only_const_data_module=True, we can not obtain opt_rm feat"

        if 'const_data' in self.sub_module_names:
            self.with_const_data = True
            self.const_block = ConstDataBlock(asm_str_src, **const_data_kwargs)
        else:
            self.with_const_data = False

        self.loss_func = TripletLoss(margin=None, mid_weight=0.)

        # obase classifier
        if obase_cls_kwargs is not None:
            assert 'opt_rm' in self.feat_source, self.feat_source
            self.obase_classifier = OBaseClassifier(feat_dim, OptGroupManager.templates_name, **obase_cls_kwargs)
        else:
            self.obase_classifier = None

        if 'group_pred' in self.sub_module_names:
            self.group_predictor = GroupPredictor(feat_dim, OptGroupManager.templates_name, **group_pred_kwargs)
        else:
            assert group_pred_kwargs is None or len(group_pred_kwargs)==0, "group_pred not in sub_module_names while group_pred_kwargs is not None"
            self.group_predictor = None

        # Dist.print(self)

    def init_weight(self):
        if self.init_method == 'scratch':
            init_weight(self)
        # elif self.init_method == 'j-pretrain':
        else:
            for name, x in self.named_parameters():
                if name in ["bert.pooler.dense.weight", "bert.pooler.dense.bias"]:
                    nn.init.normal_(x, std=0.01)
                elif 'bert.' in name:
                    pass
        # else:
            # for name, x in self.named_parameters():
            #     if not name.startswith('bert.'):
            #         continue
            #     name = name[len("bert."):]
            #     if name in ['cls.seq_relationship.bias', 'cls.predictions.decoder.weight',
            #                 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias',
            #                 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight',
            #                 'cls.predictions.transform.dense.bias', 'cls.predictions.bias']:
            #         nn.init.normal_(x, std=0.01)

        for m in self.modules():
            if isinstance(m, ConstDataBlock):
                m.init_weight()
            elif isinstance(m, OBaseClassifier):
                m.init_weight()
            elif isinstance(m, DynamicFc):
                m.init_weight()
            elif isinstance(m, GroupPredictor):
                m.init_weight()
            # elif  # post_block initial

        if hasattr(self, 'bert'):
            self.freeze_parameters()

    def freeze_parameters(self):
        if self.freeze_emb:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        freeze_layer_count = self.freeze_cnt
        if freeze_layer_count != -1:
            for layer in self.bert.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

    def get_param_lr_setting(self, base_lr, base_wd, eps=1e-8):
        # split param to group
        pos_embed, embed, bert_norm, bert_bias, bert_left, no_bert = [], [], [], [], [], []
        names = defaultdict(list)
        for name, param in self.named_parameters():
            if 'bert.' in name:
                if 'bert.embeddings.position_embeddings' in name:
                    # 0 / 1
                    #  'bert.embeddings.position_embeddings.weight'
                    pos_embed.append(param)
                    names["pos_embed"].append(name)
                elif 'bert.embeddings' in name and '_embeddings' in name:
                    # 2
                    # ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight',
                    embed.append(param)
                    names["embed"].append(name)
                elif 'LayerNorm.' in name:
                    # 2*2*12 + 2 = 50
                    # 'bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias'
                    # i=0, 1, ...11
                    #   bert.encoder.layer.i.attention.output.LayerNorm.weight,
                    #   bert.encoder.layer.i.attention.output.LayerNorm.bias,
                    #   bert.encoder.layer.i.output.LayerNorm.weight,
                    #   bert.encoder.layer.i.output.LayerNorm.bias
                    bert_norm.append(param)
                    names["bert_norm"].append(name)
                elif 'pooler.dense' in name or 'data_embeddings' in name:
                    # 'bert.pooler.dense.bias'
                    # 'bert.pooler.dense.weight'
                    no_bert.append(param)
                    names['no_bert'].append(name)
                elif '.bias' in name:
                    # 12 * 6 = 72
                    # i=0, 1, ...11
                    #  'bert.encoder.layer.i.attention.self.query.bias',
                    #  'bert.encoder.layer.i.attention.self.key.bias',
                    #  'bert.encoder.layer.i.attention.self.value.bias',
                    #  'bert.encoder.layer.i.attention.output.dense.bias',
                    #  'bert.encoder.layer.i.intermediate.dense.bias',
                    #  'bert.encoder.layer.i.output.dense.bias',
                    bert_bias.append(param)
                    names["bert_bias"].append(name)
                else:
                    # 12 * 6 = 72
                    # i=0, 1, ...11
                    #   'bert.encoder.layer.i.attention.self.query.weight',
                    #   'bert.encoder.layer.i.attention.self.key.weight',
                    #   'bert.encoder.layer.i.attention.self.value.weight',
                    #   'bert.encoder.layer.i.attention.output.dense.weight',
                    #   'bert.encoder.layer.i.intermediate.dense.weight',
                    #   'bert.encoder.layer.i.output.dense.weight'
                    bert_left.append(param)
                    names["bert_left"].append(name)
            else:
                no_bert.append(param)
                names['no_bert'].append(name)
        param_set = [pos_embed, embed, bert_norm, bert_bias, bert_left, no_bert]
        print(names)

        # set lr factor for each group
        lr_factors, wd_factors = [1.0] * len(param_set), [1.0] * len(param_set)
        if self.lr_factors is not None:
            for i, lrf in enumerate(self.lr_factors):
                lr_factors[i] = lrf
        if self.wd_factors is not None:
            for i, wdf in enumerate(self.wd_factors):
                wd_factors[i] = wdf

        # obtain lr setting
        lr_settings = []
        for ps, lr_f, wd_f in zip(param_set, lr_factors, wd_factors):
            if abs(lr_f*base_lr) < eps:
                if abs(wd_f*base_wd) < eps:
                    for p in ps:
                        p.require_grad = False
                else:
                    warnings.warn("[model.lr_set]: lr is set to 0 while wd is not 0, it will leads to zero weight.")
                    printl("[model.lr_set]: lr is set to 0 while wd is not 0, it will leads to zero weight.")
            else:
                lr_settings.append({"params": ps, "lr": base_lr*lr_f, 'weight_decay': base_wd*wd_f})
        Dist.print("[model.lr_set]:", [{k: len(v) if isinstance(v, list) else v for k, v in lrs.items()} for lrs in lr_settings])
        return lr_settings

    def tokenizer_call(self, func_strs, device):
        """
            func_strs: List(str), each str represent a function
        """
        ret1 = self.tokenizer(
            func_strs, add_special_tokens=True, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt')  # tokenize them
        return ret1["input_ids"].to(device), ret1["attention_mask"].to(device)

    def forward(self, bsc_feat, asm_tokens, graph_dict, infos,
                fids=None, tids=None, coarse_label=None, with_obase=False):
        if self.training:
            return self.loss(bsc_feat, asm_tokens, graph_dict, infos, fids, tids, coarse_label)

        fn_embeddings = self.pre_forward(bsc_feat.device, asm_tokens, graph_dict, infos)  # obtain fn_embedding
        group_result, _ = self.group_pred_forward(bsc_feat, fn_embeddings, infos, None)
        bsc_feat, fn_embeddings, _other_out = self.const_data_forward(bsc_feat, fn_embeddings, infos)
        f = self.post_forward(bsc_feat, fn_embeddings)                                     # perform option remove
        # return predict obase
        other_out = {}
        other_out.update(_other_out)
        other_out.update(group_result)
        if with_obase:
            obase = self.obase_classifier(fn_embeddings)
            other_out['obase'] = self.obase_classifier.class_names[obase]
        return f, other_out

    def pre_forward(self, device, asm_tokens, graph_dict, infos):
        """
        obtain fn_embedding from bert
        """
        if 'opt_rm' in self.feat_source:
            ida_asm_str = [info[self.asm_str_src] for info in infos]
            input_ids, attention_mask = self.tokenizer_call(ida_asm_str, device)
            output = self.bert(infos, input_ids, attention_mask)
            fn_embeddings = output.pooler_output
            return fn_embeddings
        else:
            return None

    def const_data_forward(self, bsc_feat, fn_embeddings, infos):
        """
            add const embedding
        """
        other_out = {}
        if self.with_const_data:
            if 'bert' not in self.sub_module_names:  # only_const_data_module:
                assert 'opt_rm' not in self.feat_source and fn_embeddings is None
                bsc_feat, _other_out = self.const_block(bsc_feat, infos)
            else:
                fn_embeddings, _other_out = self.const_block(fn_embeddings, infos)
            other_out.update(_other_out)
        return bsc_feat, fn_embeddings, other_out

    def group_pred_forward(self, bsc_feat, fn_embeddings, infos, tids):
        """
            when fn_embeddings is None, means no bert model load, it is multi-stage training policy
        """
        if self.group_predictor is not None:
            if 'bert' not in self.sub_module_names:
                assert 'opt_rm' not in self.feat_source and fn_embeddings is None
                feat = bsc_feat
            else:
                feat = fn_embeddings
            pred_result, loss = self.group_predictor.predict_or_loss(feat, infos, tids)
            return pred_result, loss
        return {}, {}

    def post_forward(self, bsc_feat, fn_e):
        def option_remove(self, bsc_feat, fn_e):
            if self.opt_rm_method == 'add':
                return bsc_feat + fn_e
            elif self.opt_rm_method == 'dynamic_fc':
                return self.opt_rm_block(bsc_feat, fn_e)
            else:
                raise ValueError(self.opt_rm_method)

        if len(self.feat_source) == 1:
            if self.feat_source[0] == 'bsc':
                f = bsc_feat
            elif self.feat_source[0] == 'opt_rm':
                f = fn_e
            else:
                raise ValueError(self.feat_source)
        else:
            f = option_remove(self, bsc_feat, fn_e)

        if self.with_post_block:
            f = self.post_block(f)
        return f

    def loss(self, bsc_feat, asm_tokens, graph_dict, infos, fids, tids, coarse_label):
        norm = {}
        if bsc_feat is not None:
            norm['norm_b'] = bsc_feat.norm(p=2, dim=-1).mean()
        fn_embeddings = self.pre_forward(bsc_feat.device, asm_tokens, graph_dict, infos)    # obtain fn_embedding
        if fn_embeddings is not None:
            norm['norm_f'] = fn_embeddings.norm(p=2, dim=-1).mean()
        _, loss_group = self.group_pred_forward(bsc_feat, fn_embeddings, infos, tids)
        bsc_feat, fn_embeddings, other_out = self.const_data_forward(bsc_feat, fn_embeddings, infos)  # with const_data

        batch_feat = self.post_forward(bsc_feat, fn_embeddings)                             # perform option remove
        loss, dist_ap, dist_am, dist_an = self.loss_func(batch_feat, fids, tids, normalize_feature=True)
        loss_infos = {"tri_loss": loss, "dist_ap": dist_ap.mean(), "dist_am": dist_am.mean(),
                      "dist_an": dist_an.mean()}
        loss_infos.update(loss_group)

        # norm1, norm2 = torch.norm(f1, p=2, dim=-1).mean(), torch.norm(f2, p=2, dim=-1).mean()
        if 'const_f' in other_out and isinstance(other_out['const_f'], torch.Tensor):
            norm['norm_c'] = other_out['const_f'].norm(p=2, dim=-1).mean()
        norm['norm'] = torch.norm(batch_feat, p=2, dim=1).mean()
        loss_infos.update(norm)

        if self.obase_classifier is not None:
            obase_loss = self.obase_classifier.loss(fn_embeddings, tids, coarse_label)
            loss_infos.update(obase_loss)
        return loss_infos


def build_model(model_class, *args, **kwargs):
    return eval(model_class)(*args, **kwargs)


if __name__ == '__main__':
    device = torch.device('cuda:0')

    model = build_model("OptRemoveBertModel", init_method='j-pretrain', freeze_cnt=10, freeze_emb=True,
                        # feat_source='opt_rm',
                        feat_source='bsc',
                        sub_modules='group_pred',
                        # with_data_embed_in_bert=True,
                        # with_const_data=True, only_const_data_module=True,
                        # const_data_kwargs=dict(out_type='const_emb:add'),
                        ).to(device)
    model.init_weight()
    print(model)

    from core.opt_rm.dataset import _create_unit_test_dataset
    tokens, dataset, data_loader = _create_unit_test_dataset(device)

    optimizer = torch.optim.Adam(model.get_param_lr_setting(0.01, 5e-4))
    tids = None

    # model = build_model("OptRemoveBertModel", obase_cls_kwargs=dict()).to(device)
    print([p.mean() for name, p in model.named_parameters() if '_embeddings' in name])
    print([p.mean() for name, p in model.named_parameters() if '11' in name])
    for idx, bsc_feat, fids, infos, asm_path, asm_token_idx, label, coarse_label, x_weight, graph_dict, asm_info in data_loader:
        graph_dict = {key: graph_dict[key].to(device) for key in graph_dict}
        bsc_feat = bsc_feat.to(device)
        asm_token_idx = asm_token_idx.to(device)
        coarse_label = coarse_label.to(device)
        model.eval()
        model(bsc_feat, asm_token_idx, graph_dict, infos)
        model.train()
        fids = fids.to(device)
        tids = torch.zeros(len(fids)).to(device)
        losses = model(bsc_feat, asm_token_idx, graph_dict, infos, fids, tids, coarse_label)
        print(losses)
        loss = sum([v for key, v in losses.items() if ('_loss' in key)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        res = model(bsc_feat, asm_token_idx, graph_dict, infos, fids, tids, coarse_label)
        print(res)
        model.train()
        break
    print([p.mean() for name, p in model.named_parameters() if '_embeddings' in name])
    print([p.mean() for name, p in model.named_parameters() if '11' in name])

    graph = dict(with_cfg_cdg=True)
    model = OptRemoveModel(tokens, 100, 768, "bsc,opt_rm", graph=graph).to(device)
    # for epoch in range(num_epochs):
    for idx, bsc_feat, fids, infos, asm_path, asm_token_idx, label, coarse_label, x_weight, graph_dict, asm_info in data_loader:
        graph_dict = {key: graph_dict[key].to(device) for key in graph_dict}
        bsc_feat = bsc_feat.to(device)
        asm_token_idx = asm_token_idx.to(device)
        coarse_label = coarse_label.to(device)
        model.eval()
        model(bsc_feat, asm_token_idx, graph_dict, infos)
        model.train()
        tids = None
        losses = model(bsc_feat, asm_token_idx, graph_dict, infos, fids, tids, coarse_label)
        print(losses)
        break
