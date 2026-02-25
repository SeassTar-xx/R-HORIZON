import argparse 
import json
import pandas as pd
import re
from sympy import simplify
from training.verl.utils.reward_score.math import is_equiv
import os
import json
import threading 
from typing import Dict, Any,Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm 
import sys

def normalize_answer(answer: str):
    return answer.strip().replace("\\dfrac", "\\frac")

def extract_problem_answers(text: str):
    # 使用正则表达式匹配 "Problem X:" 和对应的答案
    pattern = r'Problem (\d+)\n?.*?\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}'
    matches = re.findall(pattern, text, re.DOTALL|re.IGNORECASE)
    matches = list(matches)
    answers = []
    for problem_num, answer in matches:
        # print(problem_num, answer)
        answers.append((int(problem_num), normalize_answer(answer.strip())))
    return answers

def judge(resp, gold):
    if 'variable' in resp.lower() or 'Undefined' in resp.lower():
        return False
    return is_equiv(resp, gold, fast = True)

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
    
def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

### add labels
def judge_multiquery_answer(key, extract_data, gold):
    gold_target = gold
    # print(extract_data, gold_target)
    cnt_valid, cnt_acc_all, cnt_acc_last, cnt_error = 0, 0, 0, 0

    try:
        extract_data = json.loads(extract_data)
    except Exception as e:
        extract_data = {}

    extract_data = {str(k) : str(extract_data[k]) for k in extract_data} 
    if len(extract_data) == len(gold_target):
        cnt_valid = 1
    labels = []
    for j, item in enumerate(gold_target):
        label = 0
        num = str(j + 1)
        if num in extract_data:
            try:
                label = int(judge(extract_data[num], item))
            except Exception as e:
                cnt_error = 1
                label = -1
        else:
            label = -1
        labels.append(label)
    
    # 全对
    if cnt_valid:
        if sum(labels) == len(gold_target):
            cnt_acc_all = 1
        # 最后一题对
        if labels[-1] == True:
            cnt_acc_last = 1
            
    return key, {
        'cnt_valid' : cnt_valid,
        'cnt_acc_all' : cnt_acc_all,
        'cnt_acc_last' : cnt_acc_last,
        'cnt_error' : cnt_error,
        'data' : extract_data,
        'gold' : gold, 
        'labels' : labels
    }


def equal_judgement(queries, fo):
    lock = threading.Lock()
    max_workers = 15
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
        futures = []
        for key, value, target in queries:
            futures.append(executor.submit(judge_multiquery_answer, key, value, target))

        # 等待所有任务完成
        for future in tqdm(concurrent.futures.as_completed(futures)):
            try:
                key, info = future.result()
                info['key'] = key
                with lock:
                    fo.write(json.dumps(info, ensure_ascii = False) + '\n')
                    fo.flush()
                
                # print(f"完成处理: {result}")
            except Exception as e:
                print(f"任务执行失败: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_input', type=str, default=None)
    parser.add_argument('--prediction', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    print(args)


    query_lst = []
    cnt = 0
    for line in open(args.raw_input, 'r'):
        item = json.loads(line)
        key = item['instanceId']
        labels = item['target'][0].split(",")
        query_lst.append((key, labels))
        cnt += 1
    raw_df = pd.DataFrame(query_lst, columns=['key', 'labels'])

    pred_df = pd.read_json(args.prediction, lines=True)
    df = pd.merge(raw_df, pred_df, on='key')
    # print(df)
    wait_to_cal = []
    for i, row in df.iterrows():
        key = row['key']
        labels = row['labels']
        pred = row['response']
        wait_to_cal.append((key, pred, labels))

    with open(args.output, 'w') as fo:
        equal_judgement(wait_to_cal, fo)

    res_df = pd.read_json(args.output, lines = True)
    # print(res_df)
    print("all correct precison : ", res_df['cnt_acc_all'].mean())
    print("last correct precision : ", res_df['cnt_acc_last'].mean())
