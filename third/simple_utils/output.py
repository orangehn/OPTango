import sys
import os


class RedirectStd(object):
    def __init__(self, log_file="log/angr_graph/std_out_err.log", mode='a'):
        log_dir, log_name = os.path.split(log_file)
        os.makedirs(log_dir, exist_ok=True)

        # files
        self.log_file_path = log_file
        self._log_file = open(self.log_file_path, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        # state
        self.have_redirect = False
        self._closed = False
        self.mode = mode

    def redirect(self):
        assert not self._closed, "the redirector have been closed, try reopen it by redirectStd.reopen()"
        sys.stdout = self._log_file
        sys.stderr = self._log_file
        self.have_redirect = True
        return self

    def redirect_back(self):
        assert not self._closed, "the redirector have been closed, try reopen it by redirectStd.reopen()"
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.have_redirect = False
        return self

    def close(self):
        assert not self._closed, "the redirector is already closed"
        self.redirect_back()
        self._log_file.close()
        self._closed = True
        return self

    def reopen(self, mode=None):
        if mode is not None:
            self.mode = mode
        self._log_file = open(self.log_file_path, self.mode)
        self._closed = False
