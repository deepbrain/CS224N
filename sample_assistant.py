import json
from tokenized_dataset import TokenizedQADataset
from math_llm import MathLLM
import random
from tqdm import tqdm
from prompt import get_old_prompts

def parse_input_file(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            sample = json.loads(line.strip())
            res.append(sample)
    return res

def prepare_problems(problems):
    prepared_problems = []
    for problem in problems:
        problem = problem.rstrip() + "\n"
        problem = f"Rephrase the following problem: {problem}"
        prepared_problems.append(problem)
    return prepared_problems

def format_output(orig_problem, input, outputs):
    rephrases = []
    for output in outputs:
        rephrase = output.replace(input, "").lstrip().rstrip() + "\n"
        if len(orig_problem) * 0.5 <= len(rephrase) <= len(orig_problem) * 2.0:
            rephrases.append(rephrase)
        else:
            rephrases.append(None)
    not_none_rephrases = [r for r in rephrases if r is not None] + [orig_problem]
    
    for i in range(len(rephrases)):
        if rephrases[i] is None:
            rephrases[i] = random.sample(not_none_rephrases, 1)[0]
    return rephrases

def get_max_tokens(problem):
    # multiply by 2 to account for 2 tokens per word
    # multiply by 2.5 to account for longer responses
    return len(problem.split(" ")) * 2 * 2.5

def _rephrase_problems(path, num_rephrases, assistant_checkpoint="/home/shubhra/Stanford/gsm/grade_school_math/rephrase-phi-v1/rephrase-phi-20240226-013026"):
    rephrase_llm = MathLLM(
        model_id=assistant_checkpoint,
        use_vllm=True,
        load=True,
        dataset_class=TokenizedQADataset,
    )
    
    parsed_lines = parse_input_file(path)
    problems = [line["problem"] for line in parsed_lines]
    inputs = prepare_problems(problems)
    
    all_rephrases = []
    for problem, input in tqdm(zip(problems, inputs)):
        max_tokens = get_max_tokens(problem)
        out = rephrase_llm.process_batch(batch=[input]*num_rephrases, max_tokens=max_tokens, temperature=0.5, top_p=1.0, presence_penalty=0, frequency_penalty=0)
        rephrases = format_output(problem, input, out)
        all_rephrases.append(rephrases)

    return list(zip(problems, all_rephrases))

def rephrase_and_write_to_json(input_path, output_path):
    prompts = get_old_prompts()
    problems_rephrases = _rephrase_problems(input_path, len(prompts))
    with open(output_path, 'a', encoding='utf-8') as file:
        for problem, rephrases in problems_rephrases:
            output_obj = {}
            output_obj["problem"] = problem
            for i, (prompt, rephrase) in enumerate(zip(prompts, rephrases)):
                output_obj[f"prompt{i}"] = prompt.get_train_prompt()
                output_obj[f"rephrase{i}"] = rephrase
            serialized_sample = json.dumps(output_obj, ensure_ascii=False)
            file.write(serialized_sample + "\n")
