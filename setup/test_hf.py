from transformers import AutoTokenizer
import transformers
import torch

model = "./Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# sequences = pipeline(
#     'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
#

#------------------------------

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# Define prompt template, the format below is from: http://fastml.com/how-to-train-your-own-chatgpt-alpaca-style-part-one/

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


# Define parameters to generate text
def gen_text(prompts, use_template=False, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1

    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512*8

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,
            # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

preprompt = """
以下の研究助成金に該当する研究者の研究キーワードを最大５つ提示してください。
研究キーワードだけをスペース区切りのリストで出力してください。
研究キーワードはできるだけ短いものにしてください。

ーーー
"""

analyze_data = '''
"公募機関名: 八洲環境技術振興財団 
資金種別: 助成金／寄附金 
分野: 基礎研究、環境・地球観測、エネルギー・資源 
件名（事業 / 助成金名）: 【八洲環境技術振興財団】2023年度研究開発・調査助成 
事業概要: 環境技術分野における基礎的な技術に関する下記の研究課題について、研究に従事しているか、又は具体的に研究着手の段階にあり、２～３年以内に研究の成果が期待されるものとします。
《研究課題》
（1）再生可能エネルギー源等に関連する技術開発
（2）クリーン燃料
（3）エネルギーの転換、輸送、貯蔵、利用の高効率化、合理化およびそれらのシステム
（4）エネルギー材料、デバイス
（5）環境保全、地球温暖化防止、エネルギー利用上の技術
（6）環境技術マネジメントの基礎研究 
'''

postprompt = '''
以下から研究キーワードだけを抽出し、カンマ区切りのリストとして出力してください。

以下は、八洲環境技術振興財団の2023年度研究開発・調査助成に該当する研究者の研究キーワードです。

1. 再生可能エネルギー源
2. クリーン燃料
3. エネルギー効率化
4. エネルギー材料
5. 環境保全

'''

results = gen_text([postprompt], use_template=True)
print(results[0])
