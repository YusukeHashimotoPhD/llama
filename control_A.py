import numpy as np
import pandas as pd

from llama_cpp import Llama

import rinna
import Stable_AI


def servey_model(model_name, prompt):
    results = []
    for temperature in np.arange(0.1, 1.0, 0.1):
        for top_p in np.arange(0.1, 1.0, 0.1):
            try:
                if model_name == 'Stable_AI':
                    result_text = Stable_AI.main(prompt, temperature, top_p)
                elif model_name == 'rinna':
                    result_text = rinna.main(prompt, temperature, top_p)
                else:
                    break
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
#    list_model_name = ['llama-2-7b.ggmlv3.q6_K.bin', 'llama-2-13b.ggmlv3.q6_K.bin', 'llama-2-7b-chat.ggmlv3.q6_K.bin', 'llama-2-13b-chat.ggmlv3.q6_K.bin']
    list_model_name = ['rinna']
    list_model_name = ['Stable_AI']

    list_prompt = ['1 + 2 =', '1 plus 2 equal ', '１たす２は？', '1 * 2 =', '1 times 2 equal ', '１かける２は？']

    folder_path = '/home/yusukehashimoto/gdrive/Data_analysis/LLM/'

    # Store results in a list first
    results = []

    for model_name in list_model_name:
        for prompt in list_prompt:
            results += servey_model(model_name, prompt)

            # Convert results list to DataFrame and write to CSV once
            df = pd.DataFrame(results)
            file_path = f'{folder_path}{model_name}.csv'
            df.to_csv(file_path)
