import requests
import time
import json

def do_post(url, data, headers, model_name):
    response = requests.post(url, json=data, headers=headers, timeout=(60, 600))
    retry_times = 0
    while True:
        try:
            if response is None or response.text is None:
                text = response
            else:
                text = json.loads(response.text)
        except Exception as e:
            print(f'error!! {model_name} result error: {e}')
        answer = text
        if response.status_code == 200 and answer is not None and len(answer) > 0:
            return answer
        retry_times += 1
        if retry_times > 3:
            break
        time.sleep(60)
        response = requests.post(url, json=data, headers=headers, timeout=(60, 600))
    raise Exception(f"{model_name} no result")


def request_response(key, messages, config):
    # 复制默认参数
    request_params = config.get('params', {}).copy()
    #  统一转成 chat 格式
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    request_params["model"] = config["model_name"]
    request_params["messages"] = messages
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    url = config["base_url"]
    # 发送请求
    resp = do_post(url, request_params, headers, config["model_name"])
    # 统一解析 chat 返回格式
    try:
        text = resp["choices"][0]["message"].get("content", "")
    except Exception as e:
        raise Exception(f"Invalid response format: {resp}")
    return {
        "key": key,
        "response": text
    }


def extract_content(response):
    assert 'choices' in response
    assert len(response['choices']) >= 1
    if 'text' in response['choices'][0]:
        return response['choices'][0]['text']
    elif  'message' in response['choices'][0]:
        return response['choices'][0]['message']['content']
    else:
        raise ValueError(f"Unexpected API format: {response}")