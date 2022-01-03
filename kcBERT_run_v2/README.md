# KcBERT

## Install

```
sudo apt update && sudo apt install -y python3-pip

git clone https://git.swmgit.org/swmaestro/bwai.git
cd bwai/src/module/kcBERT_run

pip3 install -r requirements.txt
```

<br></br>
## run.inference(sentence)

문장에 욕설이 있는지 없는지 판단하고 토큰별로 확률을 반환하는 함수

### Parameters: 
#### sentence : str  
입력할 문장  
길이는 128자 이하  
#### iteration : int
욕설을 탐지하기 위해 모델을 통과하는 횟수

<br></br>
### Returns:
#### logits : list
욕설이 아닐 확률과 욕설일 확률의 log
#### pred : int
욕설이면 1, 욕설이 아니면 0
#### sentence_filtered : str
욕설로 판단된 토큰을 '*'로 치환한 문장
#### tokens : list
입력 문장을 토큰으로 나눈 상태
#### prob : list
토큰별 욕설일 확률

<br></br>
### Examples:
```python
>>> import run
>>> run.inference(“돈이 있다는 조건으로. 돈없는 대학 생활은 좃이제.”)
>>> [-0.473858, 5.485849], 1, "돈이 있다는 조건으로. 돈없는 대학 생활은 *이제.", ["돈", "이", ..., "이제"], [0.12, 0.32, ..., 0.02]
```

<br></br>
## Run

`python3 run.py`
