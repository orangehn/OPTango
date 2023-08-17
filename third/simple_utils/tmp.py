import time
import csv


class UpdateFileReader(object):
    def __init__(self, file_path, end_res=None, check_interval=10):
        """
            file_path:
            end_res: can be callable or str
            check_interval: how many seconds to read the file
        """
        self.file_path = file_path
        self.end_res = end_res
        self.check_interval = check_interval
        self._read_results = None

    def _update_read_loop(self):
        start_idx = 0
        end = False
        while not end:
            time.sleep(self.check_interval)

            # read file
            with open(self.file_path) as f:
                for i, res in enumerate(self._read_results(f)):
                    if i < start_idx:
                        continue
                    start_idx += 1
                    if self._is_end(res):
                        end = True
                        break
                    yield res

    def lines(self):
        if self.end_res is None:  # end with empty line
            self.end_res = "\n"
        self._read_results = lambda f: f.readlines()
        for res in self._update_read_loop():
            yield res

    def csv_lines(self):
        if self.end_res is None:  # end with empty csv line
            self.end_res = lambda res: len(res) == 0
        self._read_results = lambda f: csv.reader(f)
        for res in self._update_read_loop():
            yield res

    def _is_end(self, res):
        if isinstance(self.end_res, res.__class__):
            return res == self.end_res
        elif callable(self.end_res):
            return self.end_res(res)
        else:
            raise TypeError(f"{self.end_res}({type(self.end_res)}) vs {res}({type(res)})")

