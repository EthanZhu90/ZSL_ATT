import numpy as np
import glob
import json
input_dir = 'all_fragments/'

filelist_phrase = []
for i in range(1,201):
    filelist = glob.glob(input_dir + str(i) + '.*')
    if(len(filelist) == 1): filelist_phrase.append(filelist[0])
    elif(len(filelist) == 2):
        if(filelist[0] > filelist[1]) : filelist_phrase.append(filelist[0])
        else: filelist_phrase.append(filelist[1])
    else:
        print('Error:  len > 3')

# read file, format and store
class_phrases = []
for i in range(len(filelist_phrase)):
    content = [line.rstrip('\n') for line in open(filelist_phrase[i], 'r')]
    content_cleaned = list(filter(
        lambda x: len(x), content
    ))
    class_phrases.append(content_cleaned)

# Writing JSON data
with open('class_phrases.json', 'w') as f:
    json.dump(class_phrases, f)

with open('class_phrases.txt', 'w') as f:
    for i in range(len(filelist_phrase)):
        for phrase in class_phrases[i]:
            f.write(phrase + '\n')