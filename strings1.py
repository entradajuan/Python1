
case1 = "wetgr123rtytg"

center = len(case1)//2
print("Center", center)
print(case1[center-1:center+2])


s1 = "XXXX"
s2 = "Lopez"

s1_ini = s1[0:len(s1)//2]
s1_end = s1[len(s1)//2:]
print(s1_ini+s2+s1_end)

s3 = "a4AAAA&!"
char = 0
char_capitalized = 0
number = 0

for i in s3:
    if (i.isdigit()):
        number += 1
    elif (i.islower()) :
        char += 1
    elif (i.isupper()):
        char_capitalized += 1

print('number: ', number)
print('char: ', char)
print('char_capitalized: ', char_capitalized)

s4 = "word1 word2-word3 word4 word5"
print(s4.split(" "))
print(s4.split("-"))

