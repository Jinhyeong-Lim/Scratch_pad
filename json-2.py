import json

lines=[]
with open('C:/Users/default.DESKTOP-6FG4SCS/Downloads/sample2.json', 'r', encoding='utf-8') as f:
    for line in f:
        lines=(json.loads(line))
        for x in lines['id']:
            if(int(x)%5==0):
                 print(lines)


