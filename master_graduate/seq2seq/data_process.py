
filepath = './data/test_data.txt'
with open(filepath,'r') as file:
    line = file.readline()
tmp = line.split(';')

new_sent =list()
for t in tmp:
   t = t.split()
   if(len(t)!=1):
       t.append(';')
       new_sent.append(t)
for sent in new_sent:
    print(sent)

