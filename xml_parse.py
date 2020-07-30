import xml.etree.ElementTree as elemTree
import wikiextractor.WikiExtractor

  # tag name -> user

def strip_tag_name(t):
    t = elem.tag
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t


events = ("start", "end")

title = None
for event, elem in elemTree.iterparse('kowiki-20200720-pages-articles.xml', events=events):
    tname = strip_tag_name(elem.tag)
    #print(tname)
    if event == 'end':
        if tname == 'title':
            title = elem.text
            #print("title!!!!!!!!!!!!!!!!!!!!!")


            print(title)

        elif tname == 'text':
            tem = []
            id = elem.text
            lw = str(id)
            resul = lw.split()
            for line in resul:
                    tem.append(line)
                    if(line.startswith('{{') and line.endswith('}}')):
                        print(line)
                        print(tem)
                        del tem
                        break


#print(tname)