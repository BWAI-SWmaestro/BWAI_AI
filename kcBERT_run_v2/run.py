# -*- coding: utf-8 -*-
import time
import pandas as pd

from modules.kcBERT_run_v2.arg_badword_pretrained_puri_000 import Arg
from modules.kcBERT_run_v2.model_for_inference import Model

args = Arg()
# 모델 통과
print("Model importing...")
model = Model(args)
model.eval()


def inference(sentence):
    start = time.time()

    logits, pred, sentence_filtered, tokens, prob = model.inference(sentence, 3)

    print("processing time {:.4f}s".format(time.time() - start))

    return logits, pred, sentence_filtered, tokens, prob


if __name__ == '__main__':
    test = pd.read_csv("badword/ratings_labeled_test.csv")
    for i in range(100):
        text, label = test.iloc[i]['document'], test.iloc[i]['label']
        probability, prediction, text_filtered, tokens, prob_text = inference(text)
        print(text)
        print(text_filtered)
        print(tokens)
        print(prob_text)
        print("label: {},\tpred: {},\tprobability: {:.2f}\t{:.2f}\n".format(label, prediction, *probability))
