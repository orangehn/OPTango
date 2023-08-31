# add by hng for support third invoke
import os
import sys
import warnings

path = os.path.split(__file__)[0]
if path not in sys.path:
    sys.path.insert(0, path)
import readidadata

MAXLEN = 512


def r2_asm_presolve(func):
    for inst in func.insts:
        if inst.op == 'cjmp':
            arg0 = inst.args[0]
            idx = inst.args[0].find(' ')
            inst.op = arg0[:idx]
            args = arg0[idx + 1:]
            idx = args.rfind(' ')
            jmp_addr = args[:idx]
            if jmp_addr.startswith("0x"):
                jmp_addr = 'loc_' + jmp_addr
            inst.args[0] = jmp_addr
            assert len(inst.args[1]) == 0
        inst.args = [arg.strip() for arg in inst.args if len(arg) > 0]
    return func


def gen_funcstr_from_r2asm(func, convert_jump, parse_kwargs={}, with_info=False):
    code_lst = []
    map_id = {}
    info = {"consts": []}
    import copy
    func = r2_asm_presolve(copy.deepcopy(func))

    for inst in func.insts:
        code = (inst.op + " " + ",".join(inst.args)).strip()
        if with_info:
            operator, operand1, operand2, operand3, annotation, inst_info = readidadata.parse_asm(
                code, parse_kwargs, with_info=True)
            record_const([operator, operand1, operand2, operand3], len(code_lst), inst_info, info)
        else:
            operator, operand1, operand2, operand3, annotation = readidadata.parse_asm(code, parse_kwargs)
        map_id[inst.addr()] = len(code_lst)
        code_lst.append(operator)
        if operand1 != None:
            code_lst.append(operand1)
        if operand2 != None:
            code_lst.append(operand2)
        if operand3 != None:
            code_lst.append(operand3)
    for c in range(len(code_lst)):
        op = code_lst[c]
        if op.startswith('hex_'):
            jumpaddr = int(op[4:], base=16)
            if map_id.get(jumpaddr):
                jumpid = map_id[jumpaddr]
                if jumpid < MAXLEN:
                    code_lst[c] = 'JUMP_ADDR_{}'.format(jumpid)
                else:
                    code_lst[c] = 'JUMP_ADDR_EXCEEDED'
            else:
                code_lst[c] = 'UNK_JUMP_ADDR'
            if not convert_jump:
                code_lst[c] = 'CONST'
    func_str = ' '.join(code_lst)
    if with_info:
        return func_str, info
    else:
        return func_str


def get_arch_from_name(name):
    for arch in ['x86', 'riscv', 'arm']:
        if arch in name:
            return arch
    return None


def gen_funcstr(f, convert_jump, parse_kwargs={}, with_info=False, arch=""):
    if arch == 'x86' or len(arch) == 0:
        return gen_funcstr_x86(f, convert_jump, parse_kwargs, with_info, arch)
    else:
        raise ValueError(f"unexcepted arch={arch}")


def gen_funcstr_x86(f, convert_jump, parse_kwargs={}, with_info=False, arch=""):
    cfg = f[3]
    # print(hex(f[0]))
    bb_ls, code_lst, map_id = [], [], {}
    info = {"consts": []}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()
    for bx in range(len(bb_ls)):
        bb = bb_ls[bx]
        asm = cfg.nodes[bb]['asm']
        map_id[bb] = len(code_lst)
        for code in asm:
            if with_info:
                operator, operand1, operand2, operand3, annotation, inst_info = readidadata.parse_asm(
                    code, parse_kwargs, with_info=True)
                record_const([operator, operand1, operand2, operand3], len(code_lst), inst_info, info)
            else:
                operator, operand1, operand2, operand3, annotation = readidadata.parse_asm(code, parse_kwargs)
            code_lst.append(operator)
            if operand1 != None:
                code_lst.append(operand1)
            if operand2 != None:
                code_lst.append(operand2)
            if operand3 != None:
                code_lst.append(operand3)
    for c in range(len(code_lst)):
        op = code_lst[c]
        if op.startswith('hex_'):
            jumpaddr = int(op[4:], base=16)
            if map_id.get(jumpaddr):
                jumpid = map_id[jumpaddr]
                if jumpid < MAXLEN:
                    code_lst[c] = 'JUMP_ADDR_{}'.format(jumpid)
                else:
                    code_lst[c] = 'JUMP_ADDR_EXCEEDED'
            else:
                code_lst[c] = 'UNK_JUMP_ADDR'
            if not convert_jump:
                code_lst[c] = 'CONST'
    func_str = ' '.join(code_lst)
    if with_info:
        return func_str, info
    else:
        return func_str


def record_const(fmt_inst, operator_token_idx, inst_info, return_info):
    def const_str_to_int(s):
        if s.startswith('0x'):
            if ',' in s: s = s.split(',')[0]
            return int(s, 16), 0
        if s.endswith('h'):  # TODO: need mofidy as hex
            try:
                return int(s[:-1], 16), 0
            except ValueError as e:
                pass
        if s.isdigit():
            return int(s), 1
        else:  # string const TODO: need convert to data
            # warnings.warn(f"found non-const str {s}")
            # num, type = None, None
            num, type = s, 2
            return num, type

    consts = []
    for i, (op_info, op) in enumerate(zip(inst_info['op_info'], fmt_inst[1:])):
        if op_info is not None:
            for const_str in op_info['CONST'] + op_info['SUB_CONST'] + op_info['CALL']:
                num, const_type = const_str_to_int(const_str)
                if num is not None:
                    consts.append((num, operator_token_idx + 1 + i, const_type))
    return_info['consts'].extend(consts)
