import requests
import json
import subprocess
import os
from tqdm import tqdm
from colorama import Fore, Style

from data.dataset import BENCHMARK_QUESTIONS, LLM_BENCHMARK_PROMPT, MOONDREAM_PROMPT
from data.rag_dataset import RAG_QUESTIONS_V10
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


def display_output(i, result: bool, reason: str):
    if result:
        print(Fore.GREEN + f"{i} PASSED" + Fore.BLUE + " " +  reason + Style.RESET_ALL)
    else:
        print(Fore.RED + f"{i} FAILED" + Fore.BLUE + " " + reason + Style.RESET_ALL)


def benchmark(benchmark_questions: list = BENCHMARK_QUESTIONS, fast_eval: bool = False):
    """
    args:
        fast_eval: disables moondream2 image verification. If the code compiles, it's said to be passed.
    """

    INSTRUCTIONS = "\nCode should be inside one python block. Comment `window.show()` (Do not show on screen). Do not write code inside a function. Remember to create `scene =     window.Scene()`"
    success = 0
    fail = 0

    for problem in tqdm(benchmark_questions):
        try:
            coding_question = output_parser(problem[1] + INSTRUCTIONS)
            # coding_question = output_parser(problem[1] + INSTRUCTIONS, llm="gemini-1.5-pro", company="google")
        except:
            print(Fore.RED + f"{problem[0]} FAILED" + Fore.BLUE + " JSONDecodeError" + Style.RESET_ALL)
            fail += 1
            continue

        coding_question_python_code = coding_question[2]
        with open("test_code.py", "w") as f:
            f.write(coding_question_python_code)
        
        try:
            subprocess.run(['python', 'test_bench.py'], timeout=2, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            fail += 1
            display_output(problem[0], False, str(exc.stderr))
            continue
        except subprocess.TimeoutExpired as exc:
            if fast_eval:
                success += 1
                display_output(problem[0], True, str(exc.stderr))
            else:
                fail += 1
                display_output(problem[0], False, str(exc.stderr))
            continue
        except Exception as e:
            print(Fore.RED + f"{problem[0]} FAILED" + Fore.BLUE + f" {e}" + Style.RESET_ALL)
            fail += 1
            continue

        if fast_eval:
            success += 1
            display_output(problem[0], True, "fast eval")
            continue

        description = image_description("output.png", MOONDREAM_PROMPT)

        os.remove("output.png")
        BENCHMARK_PROMPT = LLM_BENCHMARK_PROMPT.format(problem[2], description)
        verdict = llm_output(BENCHMARK_PROMPT).lower().strip()

        if (verdict == "yes"):
            print(Fore.GREEN + f"{problem[0]} PASSED" + Fore.BLUE + " Verdict: YES" + Style.RESET_ALL)
            success += 1
        else:
            print(Fore.RED + f"{problem[0]} FAILED" + Fore.BLUE + " Vedict: NO" + Style.RESET_ALL)
            fail += 1

    eval_type = "fast eval" if fast_eval else "normal eval"
    print("\n\n\n")
    display_output(success, True, eval_type)
    display_output(fail, False, eval_type)
    display_output((success/(success+fail)*100), True, "SUCCESS %")


def run_specific_benchmark(i: list):
    test_cases = [BENCHMARK_QUESTIONS[x] for x in i]
    benchmark(test_cases)


def rag_benchmark(benchmark_questions: list = RAG_QUESTIONS_V10):
    score = 0
    total_score = len(benchmark_questions)

    for problem in tqdm(benchmark_questions):
        question = problem[1]
        references = problem[2]
        point_per_reference = 1/len(references)
        response, output_references, code = output_parser(question)

        for reference in references:
            if reference in output_references:
                score += point_per_reference
                display_output(problem[0], True, "SUCCESS")
            else:
                display_output(problem[0], False, f"Reference \n{reference} not found in \n{output_references}")

    print("\n\n")
    display_output(score/total_score, True, f"% RAG Success\n{score} correct out of {total_score}")


if __name__ == "__main__":
    # benchmark(fast_eval=True)
    # run_specific_benchmark([20])
    rag_benchmark()
