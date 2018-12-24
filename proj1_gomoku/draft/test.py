import re

five = '22222'
live4 = '022220'
live3 = '022200|002220|020220|022020'
live2 = '002200|002020|020200|020020'
form4 = '22220|02222|20222|22022|22202'
sleep3 = '22200|00222|002022|220200|20202|20022|22002|02220'
sleep2 = '22000|00022|000202|202000|20020|02002'
live1 = '00200'
sleep1 = '00002|20000'

five1 = '11111'
live41 = '011110'
live31 = '001110|011100|010110|011010'
live21 = '001100|001010|010100|010010'
form41 = '11110|01111|10111|11011|11101'
sleep31 = '11100|00111|001011|110100|10101|10011|11001|01110'
sleep21 = '11000|00011|000101|101000|10010|01001'
live11 = '00100'
sleep11 = '00001|10000'

idx_5 = 0
idx_live4 = 1
idx_form4 = 2
idx_live3 = 3
idx_sleep3 = 4
idx_live2 = 5
idx_sleep2 = 6
idx_live1 = 7
idx_sleep1 = 8

l = {}

def get_dir(p, value):
    for i in range(0, 222223):
        i = str(i)
        s = i.zfill(6)
        flag = 0
        for j in s:
            if int(j) > 2:
                flag = 1
                break
        if flag:
            continue
        if re.search(p, s):
            if not l.__contains__(s):
                l[s] = value

get_dir(five, idx_5)
get_dir(live4,idx_live4)
get_dir(form4,idx_form4)
get_dir(live3,idx_live3)
get_dir(sleep3,idx_sleep3)
get_dir(live2,idx_live2)
get_dir(sleep2,idx_sleep2)
get_dir(live1,idx_live1)
get_dir(sleep1,idx_sleep1)
print(l)
l.clear()
get_dir(five1, idx_5)
get_dir(live41,idx_live4)
get_dir(form41,idx_form4)
get_dir(live31,idx_live3)
get_dir(sleep31,idx_sleep3)
get_dir(live21,idx_live2)
get_dir(sleep21,idx_sleep2)
get_dir(live11,idx_live1)
get_dir(sleep11,idx_sleep1)
print(l)