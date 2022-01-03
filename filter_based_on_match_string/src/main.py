import os.path
from bad_word_list.make_list import make_bad_word_list
from filter.filter import Filter

if __name__ == '__main__':
    # 욕설 리스트 만들기
    if not os.path.isfile("bad_word_list/bad_word.csv"):
        make_bad_word_list()

    # 만든 리스트로 욕설 필터링
    Filter().run_filtering_only()
