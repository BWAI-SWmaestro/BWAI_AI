"""
욕설 리스트 만들기

인스티즈 욕설 필터링 리스트와 티스토리에 누군가가 올려놓은 리스트를 합쳐 제작
"""

import numpy as np


def make_bad_word_list():
    instiz = np.loadtxt("bad_word_list/bad_words_original/instiz.csv", delimiter=',', encoding='utf-8', dtype=np.unicode)
    bad_word_from_tistory = np.loadtxt("bad_word_list/bad_words_original/bad_word_from_tistory.csv", delimiter=',', encoding='utf-8', dtype=np.unicode)

    bad_word = list(sorted(set(list(instiz) + list(bad_word_from_tistory))))

    np.savetxt("bad_word_list/bad_word.csv", bad_word, fmt='%s', delimiter=',', encoding='utf-8')
