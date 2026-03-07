import requests
import time
import json


def do_post(url, data, headers, model_name):
    """
    更健壮的 POST 封装：
    - 总是打印 status_code 和部分返回内容，方便排查 DeepSeek 接口错误
    - 正确处理 JSON 解析失败的情况
    - 仅在 HTTP 200 且返回中包含 choices 字段时认为成功
    """
    retry_times = 0
    last_error_text = None
    while retry_times <= 3:
        try:
            response = requests.post(url, json=data, headers=headers, timeout=(600, 600))
            status = response.status_code
            raw_text = response.text
            try:
                resp_json = response.json()
            except Exception as e:
                print(f"[{model_name}] JSON 解析失败: {e}, status={status}, raw={raw_text[:300]}")
                resp_json = None

            # 只在 200 且返回中有 choices 时认为成功（DeepSeek/OpenAI chat 标准格式）
            if status == 200 and isinstance(resp_json, dict) and resp_json.get("choices"):
                return resp_json

            # 打印错误信息，继续重试
            print(f"[{model_name}] 请求失败或无有效 choices, status={status}, body={str(resp_json)[:300]}")
            last_error_text = raw_text
        except Exception as e:
            print(f"[{model_name}] 请求异常: {e}")

        retry_times += 1
        if retry_times > 3:
            break
        time.sleep(60)

    # 所有重试失败，抛出更详细的异常信息
    raise Exception(f"{model_name} no result, last_response={str(last_error_text)[:300]}")


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