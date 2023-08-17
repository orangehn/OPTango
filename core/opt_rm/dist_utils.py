# for parallel
import datetime
import math
import os
import warnings

import torch.utils.data as data
import torch.distributed as dist
import torch.nn.parallel as dp
from torch.utils.data.distributed import DistributedSampler
import torch
from package.simple_utils.simple_log import printl


class DistGeneralSampler(DistributedSampler):
    def __init__(self, sampler, dataset, *args, seed=0, log=False, repeat_for_gpu=True, sub_sample_step=1, **kwargs):
        super().__init__(dataset, *args, seed=seed, **kwargs)
        self.collate_callable()

        self.sub_sample_step = sub_sample_step
        self.sampler = sampler
        l = len(dataset)
        if self.sampler is not None:
            l = len(self.sampler)
            if self.drop_last:  # after drop last the batch_size should be a multiple of num_gpu(self.num_replicas)
                assert l % (self.num_replicas * self.sub_sample_step) == 0, (l, self.num_replicas, self.sub_sample_step)
            if not hasattr(self.sampler, 'g'):
                warnings.warn(f"sampler {sampler} have no attr g(torch.Generator), there should have no random")
        self.num_samples = int(math.ceil(l / (self.num_replicas * self.sub_sample_step))) * self.sub_sample_step
        self.repeat_samples = 0
        if not self.drop_last:
            if repeat_for_gpu:
                self.repeat_samples = self.num_samples * self.num_replicas - l
            else:
                warnings.warn(f"length of sampler is not a multiple of num_gpu, please set repeat_for_gpu=True, "
                              f"otherwise do not use Dist.gather that may cause keep waiting problem.")
        # if self.rank < l % self.num_replicas:  # if some gpu have no data, cause keep waiting problem when gather
        #     self.num_samples += 1
        self.log = log

    def __iter__(self):
        # super().__iter__()  # fixed sampler seed
        # deterministically shuffle based on epoch and seed
        if self.sampler is not None:
            if hasattr(self.sampler, 'g'):
                self.sampler.g.manual_seed(self.seed + self.epoch)
            indices = list(self.sampler.__iter__())
        else:
            indices = self.default_iter()
        # repeat to make sure every gpu have data, make gather no problem
        indices = indices + indices[:self.repeat_samples]
        # subsample
        indices = self.sub_sample(indices)
        if self.log:
            print(f"(sample index of gpu) {self.rank}/{self.num_replicas}:", indices)
        assert len(indices) == self.num_samples, (self.rank, len(indices), self.num_samples)
        self.set_epoch(self.epoch + 1)   # change epoch to change seed of next epoch
        return iter(indices)

    def default_iter(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        return indices

    def sub_sample(self, indices):
        # if self.sub_sample_step == 1:
        #     new_indices = indices[self.rank:len(indices):self.num_replicas]
        # else:
        indices_set = []
        i = 0
        while i < len(indices):
            indices_set.append(indices[i:i+self.sub_sample_step])
            i += self.sub_sample_step
        indices_set = indices_set[self.rank:len(indices):self.num_replicas]
        new_indices = []
        for ins in indices_set:
            new_indices.extend(ins)
        return new_indices

    def collate_callable(self):
        for name, value in self.__dict__.items():
            if callable(value):
                setattr(self, name, value)


class Dist(object):
    """
    1. setup, cleanup
    2. model, sampler, loss gather if need interact across batch.
    4. operator can only do in a single process:
        print,printl => Dist.print
        clear directory that need to save our result => Dist.check
    5. sync for operation need all gpu finished work:
        evaluate read from inference saved result of all GPU. => Dist.sync
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:10000 learning_test/bert_example/bert_example1_2.py
    """
    open = False

    @staticmethod
    def take_mem(n):
        if Dist.open and n > 0:
            rank = Dist.get_rank()
            d = torch.randn(int(20000*n), int(10000/5.3*6))
            d.to(rank)
            return d

    @staticmethod
    def release_mem(d):
        if d is not None:
            del d

    @staticmethod
    def setup():
        if Dist.open:
            rank = int(os.environ["RANK"])
            torch.cuda.set_device(rank % torch.cuda.device_count())
            # local_rank = int(os.environ["LOCAL_RANK"])
            # device = torch.device("cuda", local_rank)
            print(rank, torch.cuda.device_count())
            os.environ["MASTER_ADDR"] = "localhost"
            # os.environ["MASTER_PORT"] = f"{port}"
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(seconds=7200)
                # init_method=f"tcp://localhost:{port}",
                # rank=rank,
                # world_size=torch.cuda.device_count()
            )

    @staticmethod
    def model(m):
        if Dist.open:
            rank = Dist.get_rank()
            device_id = rank % torch.cuda.device_count()
            m = m.to(device_id)
            m = dp.DistributedDataParallel(m, device_ids=[device_id])
        return m

    @staticmethod
    def sampler(s, dataset, log=False, shuffle=True, drop_last=False, **kwargs):
        if Dist.open:
            s = DistGeneralSampler(s, dataset, log=log, shuffle=shuffle, drop_last=drop_last, **kwargs)
        return s

    @staticmethod
    def get_model(m):
        if isinstance(m, dp.DistributedDataParallel):
            return m.module
        elif isinstance(m, torch.nn.Module):
            return m
        else:
            raise TypeError

    @staticmethod
    def get_sampler(d):
        if isinstance(d, torch.utils.data.DataLoader):
            d = d.sampler
        if isinstance(d, DistGeneralSampler):
            d = d.sampler
        return d

    @staticmethod
    def gather(*args):
        if Dist.open:
            gather_args = []
            for arg in args:
                arg_list = [torch.zeros_like(arg) for _ in range(dist.get_world_size())]
                dist.all_gather(arg_list, arg)
                arg = torch.cat(arg_list, dim=0)
                gather_args.append(arg)
            if len(args) == 1:
                gather_args = gather_args[0]
            return gather_args
        return args

    @staticmethod
    def cleanup():
        if Dist.open:
            dist.destroy_process_group()

    @staticmethod
    def get_rank(rank=None):
        if Dist.open:
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
        else:
            rank = 0
        return rank

    @staticmethod
    def print(*args, **kwargs):
        if Dist.check():
            printl(*args, **kwargs)

    @staticmethod
    def check():
        """
        make sure something that only single process can do.
        """
        if Dist.open:
            rank = Dist.get_rank()
            if rank == 0:
                return True
            else:
                return False
        return True

    @staticmethod
    def sync():
        if Dist.open:
            return Dist.gather(torch.tensor([Dist.get_rank()]).to(Dist.get_rank()))

    @staticmethod
    def num_gpu():
        if Dist.open:
            return dist.get_world_size()
        return 1


if __name__ == '__main__':
    """
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:10002 core/opt_rm/dist_utils.py
    """
    from tqdm import tqdm

    class TempDataset(object):
        def __init__(self):
            self.data = [torch.arange(13).float().reshape(-1, 1) for _ in range(3)]
            self.e_data = [{"base_tid": i // 3 - 1} for i in range(13)]

        def __getitem__(self, idx):
            return *[d[idx] for d in self.data], self.e_data[idx]

        def __len__(self):
            return len(self.e_data)
    Dist.open = False
    Dist.setup()
    device = Dist.get_rank()

    dataset = TempDataset()
    model = torch.nn.Linear(1, 1).to(device)
    model = Dist.model(model)

    from core.opt_rm.sampler import TemplateTripletSampler, OptGroupSampler
    sampler = TemplateTripletSampler(dataset, sampler_n=2, num_instances=3)
    sampler = Dist.sampler(sampler, dataset, log=True, sub_sample_step=3)
    data_loader = torch.utils.data.DataLoader(dataset, 2*3, num_workers=0, sampler=sampler, drop_last=True)
    print(Dist.get_sampler(data_loader).get_tids)

    # sampler = OptGroupSampler(dataset)
    # # sampler = Dist.sampler(sampler, dataset, shuffle=False, log=True, repeat_for_gpu=False)  # use Dist.gather cause keep waiting problem
    # sampler = Dist.sampler(sampler, dataset, shuffle=False, log=True, repeat_for_gpu=True)
    # data_loader = torch.utils.data.DataLoader(dataset, 2*3, num_workers=0, sampler=sampler)

    # sampler = Dist.sampler(None, dataset, shuffle=False, drop_last=False, log=True)
    # data_loader = torch.utils.data.DataLoader(dataset, 2*3, num_workers=0, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    model.train()
    print(Dist.get_rank(), ":", "first epoch ------------------------------")
    Dist.sync()
    l = 0
    for data, _, _, _ in tqdm(data_loader):
        print(Dist.get_rank(), ":", data.T)
        data = data.to(device)
        optimizer.zero_grad()
        x = model(data)
        # x = Dist.gather(x)
        x.sum().backward()
        optimizer.step()
        print(Dist.get_rank(), ":", x)
        l+=len(data)
    Dist.sync()
    print(Dist.get_rank(), ":", l)

    print(Dist.get_rank(), ":", "second epoch ------------------------------")
