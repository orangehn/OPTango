from __future__ import print_function  # for python2.x compatible


def print_red(msg, no_color_msg="", end="\n"):
    print("\033[31;1m" + msg + "\033[0m", end="")
    print(no_color_msg, end=end)


def print_green(msg, no_color_msg="", end="\n"):
    print("\033[32m" + msg + "\033[0m", end="")
    print(no_color_msg, end=end)
