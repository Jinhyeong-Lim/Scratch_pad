from nltk import ngrams
def ngram(text, n, pad_left=False, pad_right=False):
    list =[]
    list = text.split()
    word=[]
    for i in range(len(list)-n+1):
        #word.append(list[i:i+n])
        yield tuple(word[i:i+n])
    #return word


text = "json is an open standardfile format"
print(ngram(text, 2))
print(ngrams(text.split(), 2))

n_list = [2,3,4]
for n in n_list:
    print("{}-gram".format(n))
    for gram in ngrams(text.split(), n):
        print(gram)

 #기존 함수는 list를 사용하여 가변적이지만 ngrams 함수는 튜플이라 변경x
 #tuple은 해싱이 되지만, list는 해싱이 되지 않는다.

#iterator는 객체를 생성할수 있다.
#generator는 반복자(iterator)과 같은 루프의 작용을 컨트롤하기 위해 쓰여짐, yeild를 사용하여
#함수가 return되면 지역변수는 사라지지만 yeild는 내부 값들을 보존해놈
#리스트나 배열을 리턴하는 함수와 피슷, 호출할수 있는 파라메터를 가지고 있다.
#어떠한 배열이나 리스트 같은 반복가능한 연속적인 값들에 대하여 객체를 만들어 놓고
#generator 1억 단위



