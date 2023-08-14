from llama_cpp import Llama


def main(llama_model, prompt, temperature, top_p):

    prompt_format = f"""
    SYSTEM: You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe.  
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information.
    Shorter answer is better.
    USER: {prompt}
    ASSISTANT:
    """

    # 推論の実行
    output = llama_model(
        prompt_format,
        max_tokens=4096,
        temperature=temperature,
        top_p=top_p,
    #    stop=["Instruction:", "Input:", "Response:", "\n"],
        echo=False,
    )

    return output["choices"][0]["text"]


if __name__ == '__main__':
    # LLMの準備
    temperature = 0.9
    top_p = 0.9

    list_model_name = ['llama-2-7b.ggmlv3.q6_K.bin', 'llama-2-13b.ggmlv3.q6_K.bin', 'llama-2-7b-chat.ggmlv3.q6_K.bin', 'llama-2-13b-chat.ggmlv3.q6_K.bin']
    prompt = 'Please tell me the answer of the following calculation, 1*2=.'

    for model_name in list_model_name:
        llama_model = Llama(model_path=f'./model/{model_name}')
        result = main(llama_model, prompt, temperature, top_p)
        print(model_name)
        print(result)