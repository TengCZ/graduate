import data_process as dp

def generate_data():
    filepath = './data/test_data.txt'
    file = open(filepath,'r')
    lines = file.readlines()
    file.close()

    new_path = './type/test_type.txt'
    new_file = open(new_path, 'w')
    i=1
    for source in lines:
        token_value = dp.get_token_value(source)
        token_type = dp.get_token_type(source)
        value_type = dp.get_value_type(source)

        line = ' '.join(token_type)
        new_file.write(line+'\n')
        new_file.write(line)
        print(i)
        i +=1
    new_file.close()
    print('done')



def generate_dict():
    train_file = './type/train_type.txt'
    test_file ='./type/test_type.txt'
    dict_file = './type/dictionary.txt'
    dic = {}

    file = open(train_file,'r')
    lines = file.readlines()

    file.close()
    i=1
    for line in lines:
        for token in line.split():
            if (token in dic):
                dic[token] += 1
            else:
                dic[token] = 1
        print(i)
        i+=1

    j=1
    file = open(test_file,'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        for token in line.split():
            if (token in dic):
                dic[token] += 1
            else:
                dic[token] = 1
        print(j)
        j += 1



    tokenList = list()
    for token in dic.keys():
        tokenList.append((token, dic[token]))
    tokenList.sort(key=lambda x: x[1], reverse=True)

    file = open(dict_file,'w')
    for token in tokenList:
            file.write(token[0] + ':' + str(token[1]) + '\n')
    file.close()
    print('done')


def fun():
    filepath = './type/type_to_value.txt'
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    type_to_value={}
    for line in lines:
        line = line.split()
        type_to_value[line[0]] = line[1]
    return type_to_value
    print(type_to_value)

if __name__ =='__main__':
    fun()





