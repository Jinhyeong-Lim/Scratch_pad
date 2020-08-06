import json

lines=[]
with open('C:/Users/default.DESKTOP-6FG4SCS/Downloads/sample2.json', 'r', encoding='utf-8') as f:
    for line in f:
        lines=(json.loads(line))

        if(int(lines['id'])%5==0):
                print(lines)
                with open('sample3.json', 'a', encoding='utf-8') as wfile:
                    json.dump(lines, wfile, ensure_ascii=False)
                    wfile.write('\n')



