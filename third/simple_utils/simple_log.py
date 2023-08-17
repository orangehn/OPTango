
class Logger(object):
    file = None
    strict_mode = False

    @staticmethod
    def set_log_file(file_name, mode='w'):
        Logger.file = open(file_name, mode)


logger = Logger()

print_old = print


def printl(*args, **kwargs):
    print_old(*args, **kwargs)
    if logger.file is None:
        if logger.strict_mode:
            raise ValueError("logger.file_name have not been set, please use logger.set_log_file(...) to set.")
    else:
        kwargs.pop("file", None)
        print_old(*args, **kwargs, file=logger.file)
        logger.file.flush()
