import sys
import os


directory = os.listdir('C:/Users/default.DESKTOP-6FG4SCS/anaconda3/envs/tensorflow/exa/Wiki/text')
print(directory)

for files in directory:
    qw = os.path.join('C:/Users/default.DESKTOP-6FG4SCS/anaconda3/envs/tensorflow/exa/Wiki/text', files)
    directory1 = os.listdir(qw)
    print(directory1)
    for file in directory1:
        print(file)
        with open(os.path.join(qw, file), 'r', encoding='utf-8') as f:
                    for line in f:
                           if line.startswith('<doc'):
                               print(line)
                               newline = next(f)
                               newline = next(f)
                               newline = next(f)
                               print(newline.strip('\n'))
                               print("\n")