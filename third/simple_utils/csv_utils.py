import collections
import csv
import os
from copy import deepcopy
from collections import defaultdict, OrderedDict
# a += b
# csv.reader('Input/inst_set_input/v/rvv(1).csv')


class DefaultValue(object):
    def __init__(self, default_value):
        self.default_value = default_value
        if isinstance(default_value, dict):
            self.__call__ = self.get_value_dict
        elif callable(default_value):
            self.__call__ = self.default_value
        else:
            self.__call__ = self.get_value_normal

    def get_value_dict(self, k):
        return self.default_value[k]

    def get_value_normal(self, k):
        return self.default_value


class CSVObject(object):
    def __init__(self, data=[], title_to_lower=False):
        self.filename = ""
        if isinstance(data, str):
            self.filename = data
            self.row_data = self.read(data)  # filename
        elif isinstance(data, list):
            self.row_data = data
        elif isinstance(data, (dict, OrderedDict)):
            self.row_data = self.dict_data_to_list_data(data)
        else:
            raise ValueError(f"not support data tupe {type(data)}")
        if title_to_lower:
            self.row_data[0] = [d.lower() for d in self.row_data[0]]

        if len(self.row_data) == 0:  # empty csv
            self.row_data.append([])  # add a empty title

    # query: selection function ########################################################
    def select_columns_by_key(self, keys, ignore_miss=False):
        """
        key
        """
        attr2idx = self.get_attr2idx()
        if ignore_miss:
            keys = [key for key in keys if key in attr2idx]
        new_csv_data = []
        for row in self.row_data[1:]:
            data = [row[attr2idx[key]] for key in keys]
            new_csv_data.append(data)
        new_csv_data = [keys] + new_csv_data
        return CSVObject(deepcopy(new_csv_data))

    def select_rows_by_key(self, key, values, ignore_miss=False):
        """
        row[key] in values
        """
        title = self.row_data[0]
        csv_dict = self.key_attr_dict(key)

        if not ignore_miss:
            ret_csv_data = [title] + [csv_dict[v] for v in values]
        else:
            ret_csv_data = [title] + [csv_dict[v] for v in values if v in csv_dict]
        return CSVObject(deepcopy(ret_csv_data))

    def select_rows(self, condition_func):
        title = self.row_data[0]
        rows = [row for row in self.row_data[1:] if condition_func(row, title)]
        return CSVObject(deepcopy([title] + rows))

    def __getitem__(self, items):
        if isinstance(items, tuple):
            row_idx, col_idx = items
        else:
            row_idx, col_idx = items, None

        if row_idx is None:                                 # csv_obj[None]
            row_data = self.row_data[1:]
        elif isinstance(row_idx, slice):                    # csv_obj[:]
            assert row_idx.start is None and row_idx.stop is None and row_idx.step is None
            row_data = self.row_data[1:]
        elif isinstance(row_idx, list):                     # csv_obj[['3', '5', '8']]
            csv_dict = self.key_attr_dict(self.row_data[0][0])
            row_data = [csv_dict[d] for d in row_idx]
        elif isinstance(row_idx, dict):  # csv_obj[{"age": "3"}], csv_obj[{"age": [3, 5]}]
            assert len(row_idx) == 1
            key, row_idx = list(row_idx.items())[0]
            csv_dict = self.key_attr_dict(key)
            if isinstance(row_idx, str):                   # csv_obj[{"age": "3"}]
                row_data = csv_dict[row_idx]
            elif isinstance(row_idx, (tuple, list)):       # csv_obj[{"age": [3, 5]}]
                print(row_idx)
                row_data = [csv_dict[d] for d in row_idx]
            else:
                raise TypeError()
        else:
            raise IndexError()

        if col_idx is None:
            data = row_data
            title = self.row_data[0]
        elif isinstance(col_idx, str):
            key_idx = self.get_attr2idx()[col_idx]  # key
            data = [row[key_idx] for row in row_data]
            title = [col_idx]
        elif isinstance(col_idx, (tuple, list)):
            attr2idx = self.get_attr2idx()
            keyidxs = [attr2idx[key] for key in col_idx]
            data = [[row[key_idx] for key_idx in keyidxs] for row in row_data]
            title = list(col_idx)
        else:
            raise IndexError()
        # ret_data = [title] + data
        # return CSVObject(deepcopy(ret_data))
        return data

    # update: update the csv file ########################################################
    def update_title(self, titles_map):
        for i, old_title in enumerate(self.row_data[0]):
            if old_title in titles_map:
                self.row_data[0][i] = titles_map[old_title]

    def update_column(self, update_func, col_name=None, col_index=None):
        if col_name is not None:
            assert col_index is None
            col_index = self.row_data[0].index(col_name)
        assert col_index is not None

        for i in range(1, len(self.row_data)):
            self.row_data[i][col_index] = update_func(self.row_data[i][col_index])

    # add: add new column or new rows ########################################################
    def add_columns(self, attrs: collections.OrderedDict, allow_repeated=False):
        """
        :param attrs: dict, {"column_name": default_value}
        :param allow_repeated: whether the new added column name can repeat with origin columns.
        """
        if allow_repeated:
            attrs_names = list(attrs.keys())
        else:
            attrs_names = [col for col in attrs if col not in self.row_data[0]]  # keep the order, can not use set sub
        attrs_values = [attrs[name] for name in attrs_names]
        self.row_data[0].extend(attrs_names)
        for row in self.row_data[1:]:
            row.extend(attrs_values)

    def add_rows(self, csv_obj, default_value={}):
        row_data = self._get_rows(csv_obj, default_value)
        self.row_data.extend(row_data)

    def __add__(self, csv_obj):
        row_data = deepcopy(self.row_data + self._get_rows(csv_obj))
        return CSVObject(row_data)

    def __iadd__(self, csv_obj):
        self.add_rows(csv_obj)
        return self

    def merge(self, csv_obj, default_value=None):
        """
        if two csv files have different columns and rows, we will
        1. extend self with the additional columns
        2. extend csv_obj with the additional columns
        3. append csv_obj's rows to self
        :param csv_obj:
        :param default_value: None, any type or dict give a map
        :return:
        """
        if self.is_empty():
            self.row_data = deepcopy(csv_obj.row_data)
            return
        if csv_obj.is_empty():
            return

        title = self.row_data[0] + csv_obj.row_data[0]
        attr_default_value = collections.OrderedDict()
        if isinstance(default_value, dict):
            for col in title:
                if col not in attr_default_value:
                    attr_default_value[col] = default_value[col] if col in default_value else None
        else:
            for col in title:
                if col not in attr_default_value:
                    attr_default_value[col] = default_value

        self.add_columns(attr_default_value, allow_repeated=False)   # add new column from csv_obj to self
        self.add_rows(csv_obj, attr_default_value)         # add new column from self to csv_obj and then append to self

    # join column  ########################################################
    def join(self, csv_obj, key):
        if self.is_empty():  # empty csv
            return deepcopy(csv_obj)
        csv_dict_b = csv_obj.to_dict(key)
        csv_dict_a = self.key_attr_dict(key)
        title_a = self.row_data[0]
        title_b = csv_obj.row_data[0]

        new_csv_data = [title_a + title_b]
        for key in csv_dict_a:
            if key in csv_dict_b:
                new_csv_data.append(csv_dict_a[key] + csv_dict_b[key])
            else:
                new_csv_data.append(csv_dict_a[key] + [None] * len(title_b))
        for key in set(csv_dict_b.keys()) - set(csv_dict_a.keys()):
            new_csv_data.append([None] * len(title_a) + csv_dict_b[key])
        return CSVObject(new_csv_data)

    def __or__(self, csv_obj):
        title = self.row_data[0]    # join by title[0]
        return self.join(csv_obj, title[0])

    # function ######################################################################
    def save(self, filename, attr_names=None):
        out_dir, name = os.path.split(filename)
        os.makedirs(out_dir, exist_ok=True)

        writer = csv.writer(open(filename, 'w', newline=''))  # newline=''不加这个，每条记录后会有多余的空行
        if attr_names is None:
            attr_names = self.row_data[0]
        writer.writerow(attr_names)

        attr2idx = self.get_attr2idx()
        new_csv_data = []
        for row in self.row_data[1:]:
            data = [row[attr2idx[key]] for key in attr_names]
            data = [('None' if d is None else d) for d in data]
            new_csv_data.append(data)
        writer.writerows(new_csv_data)

    def tolist(self, keepdim=True):
        data = self.row_data[1:]
        if len(data) == 1 and not keepdim:
            data = data[0]
            if len(data) == 1 and not keepdim:
                data = data[0]
        return data

    def __str__(self):
        return "\n".join([",\t".join(row) for row in self.row_data])

    def __repr__(self):
        return self.__str__()

    def shape(self):
        return len(self.row_data)-1, len(self.row_data[0])

    # utils functions
    def read(self, filename):
        row_data = []
        f = open(filename)
        reader = csv.reader(f)
        for row in reader:
            row_data.append(row)
        f.close()
        return row_data

    def get_attr2idx(self):
        attr2idx = {}
        for i, key in enumerate(self.row_data[0]):
            attr2idx[key] = i
        return attr2idx

    def key_attr_dict(self, key_attr):
        idx = self.get_attr2idx()[key_attr]
        data = {}
        for row in self.row_data[1:]:
            assert row[idx] not in data, f"there are multi rows whose {key_attr}={row[idx]}"
            data[row[idx]] = row
        return data

    def group_by(self, attrs):
        if isinstance(attrs, str):
            attrs = [attrs]
        attr2idx = self.get_attr2idx()
        idxs = [attr2idx[attr] for attr in attrs]
        data = defaultdict(list)
        for row in self.row_data[1:]:
            key = tuple([row[idx] for idx in idxs])
            data[key].append(row)
        return data

    def to_dict_data(self):
        """
        [
            ["attr1", "attr2"],
            [attr11, attr21],
            [attr12, attr22]
        ]
        =>
        {
            "attr1": [attr11, attr12 ...],
            "attr2": [attr21, attr22 ...],
        }
        """
        title = self.row_data[0]
        col_datas = [[] for _ in range(len(title))]
        for row in self.row_data[1:]:
            for col_i in range(len(row)):
                col_datas[col_i].append(row[col_i])
        return {t: col_datas[col_i] for col_i, t in enumerate(title)}

    def dict_data_to_list_data(self, key_dict):
        """
        {
            "attr1": [attr11, attr12 ...],
            "attr2": [attr21, attr22 ...],
        }
        =>
        [
            ["attr1", "attr2"],
            [attr11, attr21],
            [attr12, attr22]
        ]
        """
        title = list(key_dict.keys())
        if not isinstance(key_dict, OrderedDict):
            title = sorted(title)

        # check valid: all attrs have same column size
        if len(title) > 0:
            len_data = len(key_dict[title[0]])
            for t in title[1:]:
                assert len(key_dict[t]) == len_data, \
                    f"{t} ({len(key_dict[t])}) have not same length as {title[0]} ({len_data})"
        else:
            len_data = 0

        rows = [title]
        for i in range(len_data):
            row = [key_dict[t][i] for t in title]
            rows.append(row)
        return rows

    def is_empty(self):
        return len(self.row_data[0]) == 0

    # utils function for add row  ########################################################
    def _get_rows(self, csv_obj, default_value={}):
        if csv_obj.is_empty():
            return []
        if self.is_empty():  # empty csv
            return deepcopy(csv_obj.row_data)

        other_attr2idx = csv_obj.get_attr2idx()
        title = self.row_data[0]

        def get_v1(data, key):  # for default_value is a dict
            try:
                if key in other_attr2idx:
                    return data[other_attr2idx[key]]
                if key in default_value:
                    return default_value[key]
                raise KeyError(key)
            except BaseException as e:
                print(self.filename)
                raise e

        def get_v2(data, key):  # for default_value is not a dict
            return data[other_attr2idx[key]] if key in other_attr2idx else default_value

        get_v = get_v1 if isinstance(default_value, dict) else get_v2
        ret_data = [[get_v(data, key) for key in title] for data in csv_obj.row_data[1:]]
        return ret_data


# csv中获得指定行
# 两个csv按照key


if __name__ == '__main__':
    # for inst_set in ["a", "b", "c", "d", "f", "m", "I", "zfh"]:
    #     print(inst_set)
    #     csv_obj = CSVObject(f"Input/inst_set_input/{inst_set}/{inst_set}_from_json.csv", title_to_lower=False)
    #     csv_obj_32 = CSVObject(f"Input/inst_set_input_32/{inst_set}/{inst_set}_32.csv", title_to_lower=True)
    #     rname_upper_32 = csv_obj_32[:, "rname_upper"]
    #     sub_csv_obj = csv_obj.select_rows_by_key("rname_upper", rname_upper_32)
    #     sub_csv_obj.save(f"Input/inst_set_input_32/{inst_set}/{inst_set}_from_json_32.csv")
    #
    #     csv_obj = CSVObject(f"Input/inst_set_input/{inst_set}/{inst_set}_inst_match.csv", title_to_lower=False)
    #     csv_obj_32 = CSVObject(f"Input/inst_set_input_32/{inst_set}/{inst_set}_32.csv", title_to_lower=True)
    #     rname_upper_32 = csv_obj_32[:, "rname_upper"]
    #     sub_csv_obj = csv_obj.select_rows_by_key("rname_upper", rname_upper_32, ignore_miss=True)
    #     sub_csv_obj.save(f"Input/inst_set_input_32/{inst_set}/{inst_set}_inst_match_32.csv")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("files")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    if args.task == 'merge_row':
        import glob
        csv_obj = None
        for f in args.files.split(","):
            files = glob.glob(f.strip())
            for sub_f in files:
                print("add", sub_f)
                if csv_obj is None:
                    csv_obj = CSVObject(sub_f)
                else:
                    csv_obj = csv_obj + CSVObject(sub_f)
        if csv_obj is not None:
            csv_obj.save(args.out)
