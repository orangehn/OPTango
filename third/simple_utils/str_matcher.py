import re


class MatcherVar(object):
    def __init__(self, name):
        self.name = name


class Matcher(object):
    def __init__(self, regex_list, pure_str=True):
        self.compiled_regex_list = []
        self.vars = []
        for regex in regex_list:
            if isinstance(regex, MatcherVar):
                last_reg_i = len(self.compiled_regex_list) - 1
                self.vars.append([regex, last_reg_i])
            else:
                if pure_str:
                    # Use the escape function to escape special characters in the regex string
                    regex = re.escape(regex)
                # Compile the regular expressions
                self.compiled_regex_list.append(re.compile(regex))

    def __call__(self, input_str):
        # Loop through the compiled regular expressions and find matches in the input string
        # matches = []
        not_full_match = False
        match_range, last_match_end = [], 0
        for regex in self.compiled_regex_list:
            match = regex.search(input_str[last_match_end:])
            if match is None:
                not_full_match = True
                break
            match_range.append((last_match_end + match.start(), last_match_end+match.end()))
            last_match_end = match_range[-1][1]

        if not_full_match:
            return None

        # Extract the middle part of the input string based on the matches
        results = {}
        for var, last_reg_i in self.vars:
            start_i = match_range[last_reg_i][1] if last_reg_i >= 0 else None
            end_i = match_range[last_reg_i+1][0] if last_reg_i + 1 < len(match_range) else None
            results[var.name] = (input_str[start_i:end_i])
        return results


from time import time


class Diff(object):
    times = {'map': 0, 'unmap': 0, 'diff': 0}
    @staticmethod
    def _map_list_to_str(*alists, e2id={}):
        for alist in alists:
            for e in alist:
                if e not in e2id:
                    e2id[e] = len(e2id)
        return ["".join([chr(e2id[e]) for e in alist]) for alist in alists], e2id

    @staticmethod
    def _unmap_diff_result_func(diff, e2id):
        id2e = {idx: e for e, idx in e2id.items()}
        # # tag: 0=>keep, 1=>add, -1=>del
        diff = [(tag, [id2e[ord(s)] for s in ss]) for tag, ss in diff]
        return diff

    @staticmethod
    def diff_lists(list1, list2, unmap_diff_result=True, dmp=None):
        if dmp is None:
            from diff_match_patch import diff_match_patch
            dmp = diff_match_patch()

        tic = time()
        (list1, list2), e2id = Diff._map_list_to_str(list1, list2)
        Diff.times['map'] += time() - tic

        tic = time()
        diff = dmp.diff_main(list1, list2)
        Diff.times['diff'] += time() - tic

        if unmap_diff_result:
            tic = time()
            diff = Diff._unmap_diff_result_func(diff, e2id)
            Diff.times['unmap'] += time() - tic
        return diff

    @staticmethod
    def diff_order_iou(list1, list2, dmp=None):
        diff = Diff.diff_lists(list1, list2, unmap_diff_result=False, dmp=dmp)
        keep_tokens, all_tokens = 0, 0
        for tag, d in diff:
            if tag == 0:
                keep_tokens += len(d)
            all_tokens += len(d)
        return keep_tokens / all_tokens


if __name__ == '__main__':
    matcher = Matcher(["start", MatcherVar('content'), '/', MatcherVar('content2'), 'end'])
    print(matcher("start, hello, /, /, xxxxx end"))

    from diff_match_patch import diff_match_patch
    list1 = ['apple', 'banana', 'orange', 'a']
    list2 = ['apple', 'banana', 'pear', 'a', 'b']

    print(Diff.diff_lists(list1, list2))
    dmp = diff_match_patch()
    print(Diff.diff_lists(list1, list2, dmp=dmp))
    print(Diff.diff_lists(list1, list2, unmap_diff_result=False))
    print(Diff.diff_order_iou(list1, list2, dmp=dmp))