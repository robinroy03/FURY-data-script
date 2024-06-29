import requests
import json
import subprocess
import os
from tqdm import tqdm
from colorama import Fore, Style

from data.dataset import BENCHMARK_QUESTIONS, LLM_BENCHMARK_PROMPT, MOONDREAM_PROMPT
from image_validator import image_description

def output_parser(prompt: str, llm: str = "llama3-70b-8192", knn: int = 3, stream: bool = False, company: str = "groq") -> tuple[str, list, str]:
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
    response = requests.post(LLM_ENDPOINT+f"/api/{company}/generate", json=obj)
    response_json = json.loads(response.text)

    try:
        python_code: str = response_json['response'].split("```")[1]
        if python_code.startswith("python") or python_code.startswith("Python"):
            python_code = python_code[6:]
    except Exception as e:
        print(e)
        python_code = ""

    return (response_json['response'], response_json['references'], python_code)


def llm_output(prompt: str, llm: str = "llama3-70b-8192", company: str = "groq"):
    LLM_ENDPOINT = "https://robinroy03-fury-bot.hf.space"
    obj = {
        "model": llm,
        "prompt": prompt
    }
    response = requests.post(LLM_ENDPOINT+f"/api/{company}/generate", json=obj)
    response_json = json.loads(response.text)

    try:
        print(response_json['choices'][0]['message']['content'])
        verdict: str = response_json['choices'][0]['message']['content'].split("Verdict:")[1]
    except Exception as e:
        print(e)
        verdict = ""

    return verdict


def benchmark(benchmark_questions: list = BENCHMARK_QUESTIONS):

    INSTRUCTIONS = "\nCode should be inside one python block. Comment `window.show()` (Do not show on screen). Do not write code inside a function. Remember to create `scene = window.Scene()`"

    for problem in tqdm(benchmark_questions):

        try:
            coding_question = output_parser(problem[1] + INSTRUCTIONS)
            # coding_question = output_parser(problem[1] + INSTRUCTIONS, llm="gemini-1.5-pro", company="google")
        except:
            print(Fore.RED + f"{problem[0]} FAILED" + Fore.BLUE + " JSONDecodeError" + Style.RESET_ALL)
            continue

        coding_question_python_code = coding_question[2]

        with open("test_code.py", "w") as f:
            f.write(coding_question_python_code)
        
        try:
            subprocess.call(['python', 'test_bench.py'], timeout=2)
        except:
            print(Fore.RED + f"{problem[0]} FAILED" + Fore.BLUE + " TimeoutError" + Style.RESET_ALL)
            continue

        try:
            description = image_description("output.png", MOONDREAM_PROMPT)
        except FileNotFoundError:
            print(Fore.RED + f"{problem[0]} FAILED" + Fore.BLUE + " FileNotFoundError" + Style.RESET_ALL)
            continue

        os.remove("output.png")
        BENCHMARK_PROMPT = LLM_BENCHMARK_PROMPT.format(problem[2], description)
        verdict = llm_output(BENCHMARK_PROMPT).lower().strip()

        if (verdict == "yes"):
            print(Fore.GREEN + f"{problem[0]} PASSED" + Fore.BLUE + " Verdict: YES" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"{problem[0]} FAILED" + Fore.BLUE + " Vedict: NO" + Style.RESET_ALL)


def run_specific_benchmark(i: list):
    test_cases = [BENCHMARK_QUESTIONS[x] for x in i]
    benchmark(test_cases)


benchmark()
# run_specific_benchmark([1])