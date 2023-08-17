from .fmt_print import *
from math import ceil


def print_table(data_list, tab_len=4):
    data_str_list = [[str(d) for d in data] for data in data_list]
    num_col = len(data_list[0])
    num_tabs = [ceil(max([len(data[j]) for data in data_str_list]) // tab_len) for j in range(num_col)]
    for data in data_str_list:
        s = "\t".join([d + '\t' * (num_tabs[j] - len(d) // tab_len) for j, d in enumerate(data)])
        print(s)


class PrettyPrinter(object):
    def __init__(self):
        self.buffer = []

    def add_print(self, *args):
        self.buffer.append(args)

    def print_table(self):
        print_table(self.buffer)
        self.buffer = []



