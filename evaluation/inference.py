import argparse 
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import os
import requests
import time

# (base_url, model_name) -> int | None — only filled when /v1/models exposes max_model_len (e.g. vLLM).
_OPENAI_MAX_MODEL_LEN_CACHE = {}

# 本地 vLLM 长生成时默认 600s 读超时易中断整条评测链；可通过环境变量调大（不改 config 里的 max_tokens）。
_INFERENCE_HTTP_CONNECT_TIMEOUT = int(os.environ.get("INFERENCE_HTTP_CONNECT_TIMEOUT", "120"))
_INFERENCE_HTTP_READ_TIMEOUT = int(os.environ.get("INFERENCE_HTTP_READ_TIMEOUT", "7200"))


def _chat_completions_url_to_models_url(chat_url: str):
    cu = chat_url.rstrip("/")
    suf = "/v1/chat/completions"
    if cu.endswith(suf):
        return cu[: -len(suf)] + "/v1/models"
    return None


def _probe_openai_max_model_len(base_url: str, model_name: str):
    key = (base_url, model_name)
    if key in _OPENAI_MAX_MODEL_LEN_CACHE:
        return _OPENAI_MAX_MODEL_LEN_CACHE[key]
    models_url = _chat_completions_url_to_models_url(base_url)
    if not models_url:
        _OPENAI_MAX_MODEL_LEN_CACHE[key] = None
        return None
    try:
        r = requests.get(models_url, timeout=5)
        if r.status_code != 200:
            _OPENAI_MAX_MODEL_LEN_CACHE[key] = None
            return None
        payload = r.json()
        for row in payload.get("data") or []:
            if row.get("id") == model_name and row.get("max_model_len") is not None:
                mlen = int(row["max_model_len"])
                _OPENAI_MAX_MODEL_LEN_CACHE[key] = mlen
                return mlen
    except Exception:
        pass
    _OPENAI_MAX_MODEL_LEN_CACHE[key] = None
    return None


def do_post(url, data, headers, model_name):
    print(url)
    retry_times = 0
    max_tokens_floor = 8192
    context_safe_floor = 1024
    while retry_times <= 3:
        response = None
        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=(_INFERENCE_HTTP_CONNECT_TIMEOUT, _INFERENCE_HTTP_READ_TIMEOUT),
            )
        except requests.exceptions.Timeout as e:
            cur_max = int(data.get("max_tokens", 0) or 0)
            if cur_max > max_tokens_floor:
                next_max = max(max_tokens_floor, cur_max // 2)
                if next_max < cur_max:
                    data["max_tokens"] = next_max
                    print(f"warn!! {model_name} timeout, downgrade max_tokens {cur_max}->{next_max}")
            print(f"warn!! {model_name} timeout: {e}")
            retry_times += 1
            if retry_times > 3:
                break
            time.sleep(10)
            continue
        except requests.exceptions.RequestException as e:
            cur_max = int(data.get("max_tokens", 0) or 0)
            if cur_max > max_tokens_floor:
                next_max = max(max_tokens_floor, cur_max // 2)
                if next_max < cur_max:
                    data["max_tokens"] = next_max
                    print(f"warn!! {model_name} request-exception, downgrade max_tokens {cur_max}->{next_max}")
            print(f"warn!! {model_name} request exception: {e}")
            retry_times += 1
            if retry_times > 3:
                break
            time.sleep(10)
            continue

        text = None
        try:
            if response is not None and response.text is not None:
                text = json.loads(response.text)
        except Exception as e:
            print(f"error!! {model_name} result parse error: {e}, status={response.status_code}")

        if response.status_code == 200 and isinstance(text, dict) and len(text) > 0:
            return text

        # vLLM/OpenAI-compatible APIs may reject when max_tokens + prompt exceeds context window.
        # Keep config max_tokens unchanged; only downgrade this request and retry.
        if response is not None and response.status_code == 400:
            err_text = ""
            try:
                err_text = response.text or ""
            except Exception:
                err_text = ""
            m = re.search(r"max_total_tokens=(\d+)", err_text)
            if m:
                lim = int(m.group(1))
                slack = 4096
                target = max(context_safe_floor, lim - slack)
                cur_max = int(data.get("max_tokens", 0) or 0)
                if cur_max > target:
                    data["max_tokens"] = target
                    print(f"warn!! {model_name} vLLM max_model_len={lim}, set max_tokens {cur_max}->{target}")
            elif "maximum context length" in err_text or "max_tokens" in err_text:
                cur_max = int(data.get("max_tokens", 0) or 0)
                if cur_max > context_safe_floor:
                    next_max = max(context_safe_floor, cur_max - 2048)
                    if next_max < cur_max:
                        data["max_tokens"] = next_max
                        print(f"warn!! {model_name} context-limit, downgrade max_tokens {cur_max}->{next_max}")

        retry_times += 1
        if retry_times > 3:
            break
        time.sleep(10)

    raise Exception(f"{model_name} no result")


def request_response(key, messages, config):
    if 'params' in config:
        request_params = config['params'].copy()
    else:
        request_params = {}
    # Keep config max_tokens as the paper baseline (65536); only shrink per-request when
    # the OpenAI-compatible server exposes max_model_len (vLLM /v1/models). HF local server
    # typically has no such field and keeps full requested budget.
    mt = int(request_params.get("max_tokens", 0) or 0)
    if mt >= 65536:
        mt = 63488
    mlen = _probe_openai_max_model_len(config["base_url"], config["model_name"])
    if mlen is not None:
        slack = 4096
        capped = max(1024, min(mt, mlen - slack))
        if capped < mt:
            print(
                f"info: server max_model_len={mlen}, cap max_tokens {mt}->{capped} for this process only"
            )
        request_params["max_tokens"] = capped
    else:
        request_params["max_tokens"] = mt
    # OpenAI-compatible chat request format
    request_params['model'] = config['model_name']
    request_params['messages'] = [
        {"role": "user", "content": messages}
    ]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    url = config['base_url']
    result = do_post(url, request_params, headers, config['model_name'])
    # 正确读取 chat 返回格式
    text = result['choices'][0]['message']['content']
    return {
        "key": key,
        "response": text
    }


def inference(queries, fo, config, max_workers = 1):
    lock = threading.Lock()
    success_count = 0
    fail_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for key, value in queries:
            futures.append((executor.submit(request_response, key, value, config)))

        for future in tqdm(concurrent.futures.as_completed(futures)):
            result = None
            try:
                result = future.result()
                item = {
                    'key' : result['key'],
                    'response' : result['response']
                }
                with lock:
                    fo.write(json.dumps(item, ensure_ascii = False) + '\n')
                    fo.flush()
                success_count += 1
                # print(f"完成处理: {result}")
            except Exception as e:
                print(f"task fail: {e}")
                fail_count += 1
                # Fail fast when backend is unhealthy; supervisor will restart service and resume.
                if fail_count >= 8 and success_count == 0:
                    raise RuntimeError("backend unhealthy, abort current round")
                if fail_count >= 20 and fail_count > success_count * 2:
                    raise RuntimeError("too many failures, abort current round")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='output.json')
    parser.add_argument('--config', type=str, default='evaluation/config.json')
    parser.add_argument('--model_name', type=str, default='THUDM/chatglm-6b')
    parser.add_argument('--max_workers', type=int, default=1)
    args = parser.parse_args()
    print(args)
    model_configs = json.load(open(args.config, 'r'))
    assert "inference" in model_configs 
    assert args.model_name in model_configs['inference']
    config = model_configs['inference'][args.model_name]

    exists = set()
    if os.path.exists(args.output):
        for line in tqdm(open(args.output)):
            item = json.loads(line)
            key = item['key']
            exists.add(key)
    print(f"load exists {len(exists)} from {args.output}")

    query_lst = []
    cnt = 0
    for line in open(args.input, 'r'):
        item = json.loads(line)
        key = item['instanceId']
        if key in exists:
            continue
        prompt = item['input']
        if "prompt_prefix" in config:
            prompt = config['prompt_prefix'] + prompt
        if "prompt_suffix" in config:
            prompt = prompt + config['prompt_suffix']
        query_lst.append((key, prompt))
        cnt += 1
    print(f"{cnt} in total, load {len(query_lst)} new queries")

    out_abs = os.path.abspath(args.output)
    print(f"[inference] append results to: {out_abs}")
    with open(args.output, "a") as fo:
        inference(query_lst, fo, config, max_workers=max(1, args.max_workers))




