import os
import random

s = ''

for i in range(1000):
    rand = random.randrange(3,10)
    rand2 = random.randrange(2, rand)
    # s += "<s>"
    for n in range(rand2):
        s += str(rand - n) + ','

    s += "\n"

text_file = open("train.txt", "w")
text_file.write(s)
text_file.close()