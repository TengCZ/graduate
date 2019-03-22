import data_process as dp

key_word = ['char','double','enum','float','int','long',
            'short','signed','struct','union','unsigned',
            'void','for','do','while','break','do','while',
           'continue','if','else','goto','switch','case',
            'default','return','auto','extern','register',
            'static','const','sizeof','typedef','volatile']

def compare(cur,last):
    length = len(last)
    num = 0
    for i in range(length):

        if( i<len(cur) and cur[i]==last[i]):
            num+=1
        else:
            break
    return num


def match(source):

    token_value = dp.get_token_value(source)
    last_token = token_value[-1]
    token_value = token_value[:-1]

    token = set(key_word)
    for c in token_value:
        token.add(c)

    match = dict()
    for word in token:
        num = compare(word,last_token)
        if(num != 0):
           match[word]=num

    ret = sorted(match.items(),key=lambda item:item[1],reverse=True)

    if(len(ret)!=0):
        for tmp in ret:
            print(tmp[0])
    else:
      print("string match is none")


def read_file(filepath):
    file = open(filepath, 'r')
    code = file.readlines()
    source = dp.outOfComment(code)
    return source

def apriori(source):

    token_type = dp.get_token_type(source)
    token_value = dp.get_token_value(source)
    value_type = dp.get_value_type(source)
    print(token_type)
    print(token_value)
    print(value_type)



def main():
    filepath = './data/sample.txt'
    source = read_file(filepath)
    print(source)
    print('-----------------')
    match(source)

if __name__ =='__main__':
    main()






