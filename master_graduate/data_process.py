from pycparser import c_lexer

type_keyword=['char','double','enum','float','int',
              'long','short','signed','struct','union',
              'unsigned']# delete 'void'

def error_print(msg,row,col):
    # pass
    print("ERROR"+msg+" "+row+" "+col)

def l_print():
    pass # print("on_lbrace_func")

def r_print():
    pass# print("on_rbrace_func")

def type_print(s):
    pass # print("unkonw type:" + s)

def outOfComment(source):

    _content = ''
    for line in source:
        if(line.find('//')>=0):
                 span = line.find('//')
                 line = line[:span]+"\n"

        _content = _content + line


    state = 0
    index = -1
    for c in _content:
        index = index + 1

        if state == 0:
            if c == '/':
                state = 1
                startIndex = index

        elif state == 1:
            if c == '*':
                state = 2
            else:
                state = 0

        elif state == 2:
            if c == '*':
                state = 3
            else:
                pass

        elif state == 3:
            if c == '/':
                endIndex = index + 1
                comment = _content[startIndex:endIndex]
                _content = _content.replace(comment, '')
                index = startIndex - 1
                state = 0
            elif c == '*':
                pass
            else:
                state = 2
    return _content

def code_to_token(code):

    lexer = c_lexer.CLexer(error_print, l_print, r_print, type_print)
    lexer.build()
    lexer.input(code)  # 'LexToken(%s,%s,%d,%d)' % (self.type, self.value, self.lineno, self.lexpos)
    token_type = list()
    token_value = list()
    value_type = dict()
    while (lexer.has_next()):
        tmp = lexer.token()

        tmp = tmp.__repr__().split(",")
        # print(tmp)
        if len(tmp) < 2:
            continue
        left = tmp[0].split("(")[-1]
        if left == 'COMMA':
            right = ','
        else:
            right = tmp[1]
        token_type.append(left)
        token_value.append(right)
        if right not in value_type:
            value_type[right] = left


    return token_type, token_value, value_type

def get_token_type(source):
    type,_ ,_  = ID_to_VAR(source)
    return type

def get_token_value(source):
    _,value,_ = ID_to_VAR(source)
    return value

def get_value_type(source):
    _,_,value_type = ID_to_VAR(source)
    return value_type

# def get_type_value(source):
#     _, _, value_type = ID_to_VAR(source)




def ID_to_VAR(code):

    token_type,token_value,value_type = code_to_token(code)

    length = len(token_value)
    for i in range(length):
        token = token_value[i]
        if (token_type[i] != 'ID'):
            continue

        if (token_type[i] == 'ID' and value_type[token] != 'ID'):
            token_type[i] = value_type[token]
            continue

        if ((i + 1) < length and token_value[i + 1] == '('):
            token_type[i] = 'FUNCTION_VAR'
            value_type[token] = 'FUNCTION_VAR'
            continue

        for j in range(i - 1, -1, -1):
            tmp = token_value[j]
            if (tmp in type_keyword):
                token_type[i] = tmp.upper() + '_VAR'
                value_type[token] = tmp.upper() + '_VAR'
                break
    return token_type, token_value, value_type

def get_map(filepath):
    # filepath = './type/type_to_value.txt'
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    type_to_value={}
    for line in lines:
        line = line.split()
        type_to_value[line[0]] = line[1]
    return type_to_value


