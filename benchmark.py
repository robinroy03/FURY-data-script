import requests
import json
import subprocess
import os
from tqdm import tqdm

from data.dataset import BENCHMARK_QUESTIONS, LLM_BENCHMARK_PROMPT, MOONDREAM_PROMPT
from image_validator import image_description

def output_parser(prompt: str, llm: str = "llama3-70b-8192", knn: int = 3, stream: bool = False) -> tuple[str, list, str]:
    """
    returns:
    --------
    LLM Response: Raw response from the LLM, str
    References: References, List[str]
    Python code: Python code, str
    """

    LLM_ENDPOINT = 'https://robinroy03-fury-engine.hf.space'
    obj = {
        "query": prompt,
        "llm": llm,
        "knn": knn,
        "stream": stream
    }
    response = requests.post(LLM_ENDPOINT+"/api/groq/generate", json=obj)
    response_json = json.loads(response.text)

    try:
        python_code: str = response_json['response'].split("```")[1]
    except Exception as e:
        print(e)
        python_code = ""

    return (response_json['response'], response_json['references'], python_code)


def llm_output(prompt: str, llm: str = "llama3-70b-8192"):
    LLM_ENDPOINT = "https://robinroy03-fury-bot.hf.space"
    obj = {
        "model": llm,
        "prompt": prompt
    }
    response = requests.post(LLM_ENDPOINT+"/api/groq/generate", json=obj)
    response_json = json.loads(response.text)

    try:
        print(response_json['choices'][0]['message']['content'])
        verdict: str = response_json['choices'][0]['message']['content'].split("Verdict:")[1]
    except Exception as e:
        print(e)
        verdict = ""

    return verdict


def benchmark():
    for problem in tqdm(BENCHMARK_QUESTIONS):
        coding_question = output_parser(problem[0])
        coding_question_python_code = coding_question[2]

        with open("test_code.py", "w") as f:
            f.write(coding_question_python_code)
        try:
            subprocess.call(['python', 'test_bench.py'], timeout=5)
        except Exception as e:
            print(e)
        
        description = image_description("output.png", MOONDREAM_PROMPT)
        BENCHMARK_PROMPT = LLM_BENCHMARK_PROMPT.format(problem[1], description)
        verdict = llm_output(BENCHMARK_PROMPT).lower()


benchmark()