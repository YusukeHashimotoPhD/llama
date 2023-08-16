import numpy as np
import pandas as pd

from llama_cpp import Llama


def call_llama(llama_model, prompt, temperature, top_p):
    prompt_format = f"""
    SYSTEM: You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe.  
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information.
    Shorter answer is better.
    Do not include the prompt in the answer.
    USER: {prompt}
    ASSISTANT:
    """

    output = llama_model(
        prompt_format,
        temperature=temperature,
        top_p=top_p,
        stop=["SYSTEM:", "USER:", "ASSISTANT:", "\n"],
        echo=False,
    )

    return output["choices"][0]["text"]


def servey_model(model, prompt):
    results = []
    for temperature in np.arange(0.1, 1.0, 0.1):
        for top_p in np.arange(0.1, 1.0, 0.1):
            try:
                result_text = call_llama(model, prompt, temperature, top_p)
                print(model_name, temperature, top_p, prompt)
                print(result_text)

                # Instead of modifying DataFrame, append to results list
                results.append({
                    'prompt': prompt,
                    'model_name': model_name,
                    'temperature': temperature,
                    'top_p': top_p,
                    'length': len(result_text),
                    'result': result_text
                })

            except Exception as e:
                print(f"Error encountered: {e}")  # More descriptive error message

    return results


if __name__ == '__main__':
    # list_model_name = ['llama-2-7b.ggmlv3.q6_K.bin', 'llama-2-13b.ggmlv3.q6_K.bin', 'llama-2-7b-chat.ggmlv3.q6_K.bin', 'llama-2-13b-chat.ggmlv3.q6_K.bin']
    # list_prompt = ['1 + 2 =', '1 plus 2 equal ', '１たす２は？', '1 * 2 =', '1 times 2 equal ', '１かける２は？']
    #
    # # Store results in a list first
    # results = []
    #
    # for model_name in list_model_name:
    #     model = Llama(model_path=f'./model/{model_name}')
    #     for prompt in list_prompt:
    #         results += servey_model(model, prompt)
    #         # for temperature in np.arange(0.1, 1.0, 0.1):
    #         #     for top_p in np.arange(0.1, 1.0, 0.1):
    #         #         try:
    #         #             result_text = main(llama_model, prompt, temperature, top_p)
    #         #             print(model_name, temperature, top_p, prompt)
    #         #             print(result_text)
    #         #
    #         #             # Instead of modifying DataFrame, append to results list
    #         #             results.append({
    #         #                 'prompt': prompt,
    #         #                 'model_name': model_name,
    #         #                 'temperature': temperature,
    #         #                 'top_p': top_p,
    #         #                 'length': len(result_text),
    #         #                 'result': result_text
    #         #             })
    #         #
    #         #         except Exception as e:
    #         #             print(f"Error encountered: {e}")  # More descriptive error message
    #
    #         # Convert results list to DataFrame and write to CSV once
    #         df = pd.DataFrame(results)
    #         file_path = '/home/yusukehashimoto/gdrive/Data_analysis/LLM/llama_test_A.csv'
    #         df.to_csv(file_path)

    model_name = 'llama2-22b-daydreamer-v2.ggmlv3.q6_K.bin'
    model = Llama(model_path=f'./model/{model_name}')
    prompt = '1+2='
    result = servey_model(model, prompt)
    print(result)