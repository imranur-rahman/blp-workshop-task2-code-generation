import mlx_lm
import mlx_lm.sample_utils
import pandas as pd
import subprocess
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
import json
import ast

import mlx.core as mx

from ai_code_sandbox import AICodeSandbox
from langchain_sandbox import SyncPyodideSandbox

from make_submission import make_submission_zip

def load_models():
    """Load code generation and test case generation models using MLX"""
    # Load code generation model
    code_model, code_tokenizer = mlx_lm.load("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit")
    
    # Load test case generation model
    test_model, test_tokenizer = mlx_lm.load("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit")
    
    return (code_model, code_tokenizer), (test_model, test_tokenizer)

def load_examples():
    """Load examples from trial_updated.csv"""
    examples_df = pd.read_csv('data/trial_updated.csv')
    # examples_df = pd.read_csv('data/dev_v2.csv')
    return examples_df

def generate_code_solutions(model_tokenizer, instruction, examples_df, num_solutions=3):
    """Generate multiple code solutions for given instruction"""
    model, tokenizer = model_tokenizer
    solutions = []
    
    # Create examples from the examples_df
    examples_text = ""
    sampled_examples = examples_df.sample(n=40, random_state=42)  # Use random seed 42
    for i, row in sampled_examples.iterrows():
        examples_text += f"""<<instruction>> 
{row['instruction']}

<<output>> {row['response']}

"""
    
    for i in range(num_solutions):
        prompt = f'''
Write the function defined in <<instruction>>.
Create a new class if that helps in writing the solution.
Do not include any explanations or repeated elements in the output.
Do not write any test cases.

{examples_text}

Now do this for
<<instruction>>
{instruction}

The code needs to pass the following test case:
{row['test_list']}

<<output>>

        '''
        sampler = mlx_lm.sample_utils.make_sampler(
            temp=0.8,
            top_p=0.9,
        )
        response = mlx_lm.generate(
            model, tokenizer, 
            prompt=prompt,
            max_tokens=512,
            sampler=sampler,
            verbose=False
        )
        # Discard solutions if they are the same
        if response.strip() in [sol.strip() for sol in solutions]:
            continue
        if response.strip() != "": # So that empty responses are not included
            solutions.append(response)
        
    return solutions

def generate_test_cases(model_tokenizer, instruction, examples_df, num_tests=3):
    """Generate multiple test cases for given instruction"""
    model, tokenizer = model_tokenizer
    test_cases = []
    
    # Create examples from the examples_df
    examples_text = ""
    sampled_examples = examples_df.sample(n=40, random_state=42)  # Use random seed 42
    for i, row in sampled_examples.iterrows():
        # Parse the test_list string to get the actual list
        try:
            test_list = ast.literal_eval(row['test_list'])
            # test_list = row['test_list']
            examples_text += f"""<<instruction>> 
{row['instruction']}

<<output>> {test_list}

"""
        except:
            # If parsing fails, use the raw string
            examples_text += f"""<<instruction>> 
{row['instruction']}

<<output>> {row['test_list']}

"""
    
    prompt = f'''
Create {num_tests} Python test cases for <<instruction>> and return them as a list.
Each test case should use only the assert statement with the expected output.
Use the <<given_test_list>> as a guide for the format of the test cases.
Focus on covering edge cases, empty inputs, and maximizing test coverage.
Do not include any explanations, natural language, or code beyond the test cases.
Do not provide the solution.
The output should be a list of assert statements and not a list of lists.

{examples_text}

Now do this for
<<instruction>>
{instruction}

<<given_test_list>>
{row['test_list']}

<<output>>

    '''
    sampler = mlx_lm.sample_utils.make_sampler(
        temp=0.8,
        top_p=0.9,
    )
    response = mlx_lm.generate(
        model, tokenizer, 
        prompt=prompt,
        max_tokens=512,
        sampler=sampler,
        verbose=False
    )
    test_cases.append(response)
    
    return test_cases

def run_code_with_tests(code, test_cases, given_test_list, sandbox):
    """Run code with test cases in secure environment and return pass count"""
    passed_tests = 0

    print(f"Given test cases:\n{given_test_list}\n")
    # test_cases = test_cases[0] if isinstance(test_cases, list) else test_cases  # Ensure test_cases is a list
    try:
        given_test_cases = ast.literal_eval(given_test_list)
    except Exception as e:
        print(f"Error evaluating given test cases: {e}")
        return passed_tests

    for test_case in given_test_cases:
        try:
            # Combine code and test case
            full_code = f"{code}\n\n{test_case}"
            print(f"Running code:\n{full_code}\n")
            # Run the code in the reused sandbox
            # result = sandbox.run_code(full_code)
            result = sandbox.execute(full_code, timeout_seconds=600)
            print(f"Result:\n{result}\n")
            # if result == 'No output': # The code passed the test
            if result.status == 'success':
                passed_tests += 1
            else:
                print (f"Given test case failed: {test_case}\n")
                return passed_tests
        except Exception:
            print (f"Exception while running given test case: {test_case}\n")
            return passed_tests
    
    print(f"Test cases:\n{test_cases}\n")
    # test_cases = test_cases[0] if isinstance(test_cases, list) else test_cases  # Ensure test_cases is a list
    try:
        test_cases = ast.literal_eval(test_cases[0])
    except Exception as e:
        print(f"Error evaluating test cases: {e}")
        return passed_tests
    
    for test_case in test_cases:
        try:
            # Combine code and test case
            full_code = f"{code}\n\n{test_case}"
            print(f"Running code:\n{full_code}\n")
            # Run the code in the reused sandbox
            # result = sandbox.run_code(full_code)
            result = sandbox.execute(full_code, timeout_seconds=600)
            print(f"Result:\n{result}\n")
            # if result == 'No output': # The code passed the test
            if result.status == 'success':
                passed_tests += 1
                
        except Exception:
            continue
    
    return passed_tests

def process_row(id, row, code_model_tokenizer, test_model_tokenizer, examples_df, sandbox, code_solution_count=3, test_case_count=3):
    """Process a single row from the dataset"""
    instruction = row['instruction']
    
    # Create a single sandbox for this row
    # sandbox = AICodeSandbox(packages=["numpy", "pandas"])
    

    
    # Generate code solutions
    code_solutions = generate_code_solutions(code_model_tokenizer, instruction, examples_df, code_solution_count)
    print(f"Generated {len(code_solutions)} code solutions:")
    for i, solution in enumerate(code_solutions):
        print(f"Solution {i+1}:\n{solution}\n")
    
    # Generate test cases
    test_cases = generate_test_cases(test_model_tokenizer, instruction, examples_df, test_case_count)
    print(f"Generated test cases:\n{test_cases}\n")
    
    best_code = ""
    max_passed = -1
    
    # Test each code solution with all test cases using the same sandbox
    for code in code_solutions:
        passed_count = run_code_with_tests(code, test_cases, row['test_list'], sandbox)
        if passed_count > max_passed or (passed_count == max_passed and len(code) < len(best_code)):
            max_passed = passed_count
            best_code = code
    
    # If no code passed any tests, use the first code solution as fallback
    if best_code == "" and code_solutions:
        best_code = code_solutions[0]
    
    print(f"Best code selected (passed {max_passed} tests):\n{best_code}\n")
    
    # return {
    #     'id': id,
    #     # 'instruction': instruction,
    #     'response': best_code,
    #     # 'tests_passed': max_passed,
    #     # 'test_cases': test_cases
    # }
    return best_code.strip()
    
    # finally:
    #     # Close the sandbox after processing all code solutions for this row
    #     sandbox.close()

def main():
    # Load models
    print("Loading models...")
    code_model_tokenizer, test_model_tokenizer = load_models()
    
    # Load examples
    print("Loading examples...")
    examples_df = load_examples()
    
    # Load dataset
    print("Loading dataset...")
    # df = pd.read_csv('data/dev_v2.csv')
    df = pd.read_csv('data/test_v1.csv')
    print(f"Dataset contains {len(df)} rows.")

    # Create the sandbox
    sandbox = SyncPyodideSandbox(allow_net=True) # No read, write, subprocess permissions
    # deno needs to be installed for this sandbox
    
    # Process each row
    results = []
    temp_row_for_testing = df.shape[0]  # Use all rows for final submission, but can limit for testing
    # temp_row_for_testing = 5  # For quick testing
    for idx, row in df.head(temp_row_for_testing).iterrows():
    # for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)}")
        result = process_row(idx, row, code_model_tokenizer, test_model_tokenizer, examples_df, sandbox, code_solution_count=4, test_case_count=6)
        results.append(result)

    print (f"Processed result:\n {results}")
    
    # Create submission file
    # submission_df = pd.DataFrame(results)
    # submission_df.to_csv('submission.csv', index=False)
    # print("Submission file created successfully!")
    # json.dump(results, open("submission.json", "w"), indent=2)
    out_df = pd.DataFrame({"id": df["id"][:temp_row_for_testing], "response": [f"```python\n{result}\n```" for result in results]})
    out_df.to_json("submission.json", orient="records", force_ascii=False, indent=2)
    print(f"✅ Wrote submission.json with {len(results)} rows (id, response).")

    # Create submission zip
    make_submission_zip()

if __name__ == "__main__":
    main()