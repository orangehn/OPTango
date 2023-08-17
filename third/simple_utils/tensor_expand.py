import torch


class TensorList(object):
    def __init__(self, tensors):
        self.tensors = tensors
        self.len = [len(t) for t in tensors]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index]

    def tensors_len(self):
        return torch.LongTensor([len(t) for t in self.tensors]).to(self.tensors[0].device)

    def to(self, *args, **kwargs):
        tensors = [tensor.to(*args, **kwargs) for tensor in self.tensors]
        return TensorList(tensors)

    def cat(self):
        return torch.cat(self.tensors)

    def split(self, inst_embeddings):
        return torch.split(inst_embeddings, self.len, dim=0)


class DeepTensorList(object):
    """
    data: [d0, d1, d2..., dk-2, (dk-1, ....)]
    """
    def __init__(self, tensors, dim_name=None):
        self.tensors = tensors
        self.dim_len = self._dim_len(tensors)  # [d1, d2, d3, ...., dk-1]
        self.dim_name = dim_name

    def _dim_len(self, tensors):
        tensor_len = [len(inner_tensors) for inner_tensors in tensors]
        if isinstance(tensors[0], list):
            dk_dims = self._dim_len(tensors[0])
            for inner_tensors in tensors[1:]:
                for i, d in enumerate(self._dim_len(inner_tensors)):
                    dk_dims[i].extend(d)
            return [tensor_len] + dk_dims
        elif isinstance(tensors[0], torch.Tensor):
            return tensor_len,
        else:
            raise TypeError("")

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, *index):
        t = self.tensors
        for i in index:
            t = t[i]
        return t

    def _to(self, tensors, *args, **kwargs):
        new_tensors = []
        for inner_tensors in tensors:
            if isinstance(inner_tensors, list):
                new_tensors.append(self._to(inner_tensors, *args, **kwargs))
            elif isinstance(inner_tensors, torch.Tensor):
                new_tensors.append(inner_tensors.to(*args, **kwargs))
            else:
                raise TypeError()
        return DeepTensorList(new_tensors)

    def to(self, *args, **kwargs):
        return self._to(self.tensors, *args, **kwargs)

    def _cat(self, tensors, tensors_depth, depth):
        tensors = [self._cat(inner_tensors, tensors_depth-1, depth) for inner_tensors in tensors]
        if tensors_depth <= depth:
            tensors = torch.cat(tensors, dim=0)
        return tensors

    def cat(self, depth=-1):
        """
        depth = 1:  data: [d0, d1, d2..., ((dk-2)*+(dk-1), ....)]
        """
        if depth < 0:
            depth = len(self.dim_len)
        return self._cat(self.tensors, len(self.dim_len), depth)

    def _split_tensor_or_list(self, data, len_list):
        if isinstance(data, torch.Tensor):
            return torch.split(data, len_list, dim=0)
        elif isinstance(data, list):
            split_data = []
            start_idx = 0
            for l in len_list:
                split_data.append(
                    data[start_idx:start_idx+l]
                )
                start_idx += l
            return split_data
        else:
            raise TypeError("")

    def split(self, data, start_dim=-1, end_dim=0):
        if start_dim < 0: start_dim = start_dim % len(self.dim_len)
        if end_dim < 0: end_dim = end_dim % len(self.dim_len)
        for len_list in self.dim_len[start_dim:end_dim-1:-1]:
            data = self._split_tensor_or_list(data, len_list)
        return data


