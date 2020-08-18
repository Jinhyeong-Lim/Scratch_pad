CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'N']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                 'ㅣ','N']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'N']
COMPLEX_LIST = {'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ'}
UNCOMPLEX_LIST={'ㄱ', 'ㄷ', 'ㅂ', 'ㅅ', 'ㅈ'}

#글자 유니코드 값 = ((초성 * 21) + 중성) * 28 + 종성 + 0xAC00
r_lst = []
def korean_to_be_englished(korean_word):

    l=[]
    for w in list(korean_word.strip()):
        ## 영어인 경우 구분해서 작성함.
        if '가' <= w <= '힣':
            ## 588개 마다 초성이 바뀜.
            ch1 = (ord(w) - ord('가')) // 588
            ## 중성은 총 28가지 종류
            ch2 = ((ord(w) - ord('가')) - (588 * ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588 * ch1) - 28 * ch2

            if(ch1 < 0 and ch1>18): ch1 = len(CHOSUNG_LIST)-1
            if (ch2 < 0 and ch2 > 21): ch2 = len(JUNGSUNG_LIST)-1
            if (ch3 < 0 and ch3 > 26): ch3 = len(JONGSUNG_LIST)-1
            #print(ch1, ch2, ch3)
            r_lst.append([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]])
        elif w == ' ': continue
        else:
            r_lst.append([w])
    cnt=0
    for i in range(len(r_lst)):
        if(len(r_lst[i])>=3):
            if(r_lst[i][2]!=' '): cnt+=1

    if(cnt == len(r_lst)): return True
    cnt=0
    for i in range(len(r_lst)):
        if(len(r_lst[i])==1): cnt+=1

    if(cnt == len(r_lst)): return True
    CHO = []
    for i in range(len(r_lst)):
        if(len(r_lst[i])>1):
            CHO.append(r_lst[i][0])
    sum1=0
    sum2=0
    for i in CHO:
        if i in COMPLEX_LIST:
            sum1+=1
        if i in UNCOMPLEX_LIST:
            sum2+=1
    if(sum2==0 and sum1>0): return True

    return False



print(korean_to_be_englished("애플워치"))
print(r_lst)