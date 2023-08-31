import pickle
import networkx
import time


def parse_operand(operator, location, operand1, keep_sub=False):
    info = {"CONST": [], "SUB_CONST": [], "CALL": []}
    operand1 = operand1.strip(' ')
    operand1 = operand1.replace('ptr ', '')
    operand1 = operand1.replace('offset ', '')
    operand1 = operand1.replace('xmmword ', '')
    operand1 = operand1.replace('dword ', '')
    operand1 = operand1.replace('qword ', '')
    operand1 = operand1.replace('word ', '')
    operand1 = operand1.replace('byte ', '')
    operand1 = operand1.replace('short ', '')
    operand1 = operand1.replace('-', '+')

    # print("operand1", operand1)
    if operand1[0:3] == 'cs:':
        operand1 = 'cs:xxx'
        return operand1, info
    if operand1[0:3] == 'ss:':
        operand1 = 'ss:xxx'
        return operand1, info
    if operand1[0:3] == 'fs:':
        operand1 = 'fs:xxx'
        return operand1, info
    if operand1[0:3] == 'ds:':
        operand1 = 'ds:xxx'
        return operand1, info
    if operand1[0:3] == 'es:':
        operand1 = 'es:xxx'
        return operand1, info
    if operand1[0:3] == 'gs:':
        operand1 = 'gs:xxx'
        return operand1, info
    if operator[0] == 'j' and not isregister(operand1):
        if operand1[0:4] == 'loc_' or operand1[0:7] == 'locret_' or operand1[0:4] == 'sub_':
            operand1 = 'hex_' + operand1[operand1.find('_') + 1:]
            return operand1, info
        else:
            # print("JUMP ",operand1)
            operand1 = 'UNK_ADDR'
            return operand1, info

    if operand1[0:4] == 'loc_':
        operand1 = 'loc_xxx'
        return operand1, info
    if operand1[0:4] == 'off_':
        operand1 = 'off_xxx'
        return operand1, info
    if operand1[0:4] == 'unk_':
        operand1 = 'unk_xxx'
        return operand1, info
    if operand1[0:6] == 'locret':
        operand1 = 'locretxxx'
        return operand1, info
    if operand1[0:4] == 'sub_':
        if not keep_sub:
            operand1 = 'sub_xxx'
        return operand1, info
    if operand1[0:4] == 'arg_':
        operand1 = 'arg_xxx'
        return operand1, info
    if operand1[0:4] == 'def_':
        operand1 = 'def_xxx'
        return operand1, info
    if operand1[0:4] == 'var_':
        operand1 = 'var_xxx'
        return operand1, info
    if operand1[0] == '(' and operand1[-1] == ')':   # what ?
        operand1 = 'CONST'
        return operand1, info
    if operator == 'lea' and location == 2:
        if not ishexnumber(operand1) and not isaddr(operand1):  # handle some address constants
            operand1 = 'GLOBAL_VAR'
            return operand1, info

    if operator == 'call' and location == 1:
        if len(operand1) > 3:
            info["CALL"].append(operand1)
            operand1 = 'callfunc_xxx'
            return operand1, info

    if operator == 'extrn':
        operand1 = 'extrn_xxx'
        return operand1, info
    if ishexnumber(operand1):
        info["CONST"].append(operand1)
        operand1 = 'CONST'
        return operand1, info
    elif ispurenumber(operand1):
        info["CONST"].append(operand1)
        operand1 = 'CONST'
        return operand1, info
    if isaddr(operand1):
        params = operand1[1:-1].split('+')
        params = [param.strip() for param in params]
        for i in range(len(params)):
            if ishexnumber(params[i]):
                info["SUB_CONST"].append(params[i])
                params[i] = 'CONST'
            elif ispurenumber(params[i]):
                info["SUB_CONST"].append(params[i])
                params[i] = 'CONST'
            elif params[i][0:4] == 'var_':
                params[i] = 'var_xxx'
            elif params[i][0:4] == 'arg_':
                params[i] = 'arg_xxx'
            elif not isregister(params[i]):
                if params[i].find('*') == -1:
                    params[i] = 'CONST_VAR'
        s1 = '+'
        operand1 = '[' + s1.join(params) + ']'
        return operand1, info

    if not isregister(operand1) and len(operand1) > 4:
        info["CONST"].append(operand1)
        operand1 = 'CONST'
        return operand1, info
    return operand1, info


def parse_asm(code, parse_kwargs={}, with_info=False):  # handle ida code to better quality code for NLP model
    annotation = None
    operator, operand = None, None
    operand1, operand2, operand3 = None, None, None
    if code.find(';') != -1:
        id = code.find(';')
        annotation = code[id + 1:]
        code = code[0:id]
    if code.find(' ') != -1:
        id = code.find(' ')
        operand = code[id + 1:]
        operator = code[0:id]
    else:
        operator = code
    # print("operator operand", operator, operand)
    if operand != None:
        if operand.find(',') != -1:
            strs = operand.split(',')
            if len(strs) == 2:
                operand1, operand2 = strs[0], strs[1]
            else:
                operand1, operand2, operand3 = strs[0], strs[1], strs[2]
        else:
            operand1 = operand
            operand2 = None
    origin_inst = [operator, operand1, operand2, operand3]  # add by opt_rm

    # parse origin_inst
    info1, info2, info3 = None, None, None
    if operand1 != None:
        operand1, info1 = parse_operand(operator, 1, operand1, **parse_kwargs)
    if operand2 != None:
        operand2, info2 = parse_operand(operator, 2, operand2, **parse_kwargs)
    if operand3 != None:
        operand3, info3 = parse_operand(operator, 3, operand2, **parse_kwargs)
    if with_info:
        return operator, operand1, operand2, operand3, annotation, \
               {"origin_inst": origin_inst, "op_info": [info1, info2, info3]}
    else:
        return operator, operand1, operand2, operand3, annotation


def isregister(x):
    registers = ['rax', 'rbx', 'rcx', 'rdx', 'esi', 'edi', 'rbp', 'rsp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14',
                 'r15']
    return x in registers


def ispurenumber(number):
    if len(number) == 1 and str.isdigit(number):
        return True
    return False


def isaddr(number):
    return number[0] == '[' and number[-1] == ']'


def ishexnumber(number):
    if number[-1] == 'h':
        for i in range(len(number) - 1):
            if str.isdigit(number[i]) or (number[i] >= 'A' and number[i] <= 'F'):
                continue
            else:
                return False
    elif number.startswith('0x'):
        try:
            number = int(number, 16)
        except ValueError:
            return False
        return True
    else:
        return False
    return True
