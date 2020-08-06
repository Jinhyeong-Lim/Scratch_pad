def ngram(text, n, pad_left=True, pad_right=True):
    list =[]
    list = text.split()
    if(pad_left==True): list.insert(0, '<s>')
    if (pad_right == True): list.insert(len(list), '</s>')
    word=[]

    for i in range(len(list)-n+1):
        word.append(list[i:i+n])

    return word



text = "i am a boy"
n_list=[2,3,4]

for n in n_list:
    print("{}-gram".format(n))
    for gram in ngram(text, n):
        print(gram)
