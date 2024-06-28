import requests
import json
import subprocess
import os
from tqdm import tqdm

from data.dataset import BENCHMARK_QUESTIONS

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
    response = requests.post(LLM_ENDPOINT+"/api/generate", json=obj)
    response_json = json.loads(response.text)

    try:
        python_code: str = response_json['response'].split("```")[1]
    except:
        python_code = ""

    return (response_json['response'], response_json['references'], python_code)


def benchmark():
    for problem in tqdm(BENCHMARK_QUESTIONS):
        with open("test_bench.py", "w") as f:
            f.write(problem[2])
        with open("code.py", "w") as f:
            f.write(output_parser(problem[1])[2])

        try:
            subprocess.call(['python', 'test_bench.py'], timeout=5)
            # os.remove("test_bench.py")
            # os.remove("code.py")
        except Exception as e:
            print(e)

benchmark()