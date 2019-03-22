# -*- coding: GBK -*-
import os

# n=0
# for i in f:
#     oldname = path+i
#     newname = path+str(n+1)+'.cpp'
#     os.rename(oldname,newname)

# dir = 'Data/1.cpp'
# with open(dir,'r') as file:
#     data = file.read()
    # print(data)
    # print(set(data))



#####################去除C语言源文件中的注释
def outOfComment(filepath):
    file = open(filepath,'r')
    _content = ''
    for line in file.readlines():
        if(line.find('//')>=0):
                 span = line.find('//')
                 line = line[:span]

        _content = "%s%s" % (_content, line)

    file.close()
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


###############.txt文件转为.cpp文件

import os
def txt_to_cpp():
    num = [(i + 1) for i in range(20)]
    for n in num:
        path = 'G:/论文源码/ProgramData/' + str(n) + '/'
        f = os.listdir(path)
        newDir = 'G:/The_Secode_Version_Data/ProgramData/'
        cou = 1
        for filename in f:
            filename = path + filename
            newFile = newDir + 'Q' + str(n) + '_' + str(cou) + '.cpp'
            code = outOfComment(filename)
            fileWrite = open(newFile, 'w')
            fileWrite.write(code)
            fileWrite.close()
            cou += 1
            print(filename, ' write to the ', newFile)
    print('Done')




###########################将源代码token化
# import clang.cindex


def code_to_token():
    path = 'G:/The_Secode_Version_Data/ProgramData/'
    f = os.listdir(path)
    newDir = 'G:/The_Secode_Version_Data/CodeToken/'
    index = clang.cindex.Index.create()
    for filename in f:
        filename = path + filename
        # print(os.path.basename(filename))
        tu = index.parse(filename)
        cur = tu.cursor
        newFile = newDir + os.path.basename(filename).split('.')[0] + '.txt'
        fileWrite = open(newFile, 'w')
        for token in cur.get_tokens():
            t = token.spelling
            fileWrite.write(t + ' ')
        # fileWrite.write('@')
        fileWrite.close()
        print(filename, ' write to the ', newFile)
    print('Done')


############### 制造词典

def generate_dict():
    codePath = 'G:/The_Secode_Version_Data/CodeToken/'
    fileList = os.listdir(codePath)
    tokens = {}
    i = 0
    for filename in fileList:
        filename = codePath + filename
        fileRead = open(filename, 'r')
        code = fileRead.readline().split()
        fileRead.close()
        for c in code:
            if (tokens.has_key(c)):
                tokens[c] += 1
            else:
                tokens[c] = 1
        i += 1
        if i % 500 == 0:
            print(filename)
    tokenList = list()
    for token in tokens:
        tokenList.append((token, tokens[token]))
    tokenList.sort(key=lambda x: x[1], reverse=True)

    file = open('G:/The_Secode_Version_Data/dictionary.txt', 'w')
    for token in tokenList:
        if (token[1] > 2):
            file.write(token[0] + ':' + str(token[1]) + '\n')
    file.close()
    print('Done')



##############读词汇表
# import os
# file = open('dictionary','r')
# vocab =list()
# for line in file.readlines():
#     line = line.strip().split(':')
#     vocab.append(line[0])
# file.close()
# token_to_int_table = {c: i for i, c in enumerate(vocab)}
# int_to_token_table = dict(enumerate(vocab))

#######################把所有源代码合并到一个文件里

# import os
# path ='G:/The_Secode_Version_Data/CodeToken/'
# fileList = os.listdir(path)
# wriPath = 'G:/The_Secode_Version_Data/all_source_code.txt'
# writeFile = open(wriPath,'w')
#
# for filename in fileList:
#     filename = path + filename
#     file = open(filename,'r')
#     code = file.readline()
#     writeFile.write(code.strip()+'\n')
#     file.close()
# writeFile.close()
# print('done')

########################## 分离出训练数据和测试数据 (训练 8k,测试 2k)

# file = open('./data/all_source_code.txt','r')
# allCode = file.readlines()
# l = len(allCode)
# train_data = open('./data/train_data.txt','w')
# test_data = open('./data/test_data.txt','w')
# for i in range(l):
#     t = i%500
#     if(t < 100):
#         test_data.write(allCode[i])
#     else:
#         train_data.write(allCode[i])
# train_data.close()
# test_data.close()




