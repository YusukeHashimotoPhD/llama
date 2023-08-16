import numpy as np
import pandas as pd
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1")
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/japanese-stablelm-base-alpha-7b",
    trust_remote_code=True,
)
model.half()

if torch.cuda.is_available():
    model = model.to("cuda:0")


def main(prompt, temperature, top_p):
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    )

    # this is for reproducibility.
    # feel free to change to get different result
    seed = 23
    torch.manual_seed(seed)

    tokens = model.generate(
        input_ids.to(device=model.device),
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        # echo=False
    )

    out = tokenizer.decode(tokens[0], skip_special_tokens=False)
    return out


if __name__ == '__main__':

    preprompt = """
【背景】
研究助成金の対象となる研究のキーワードを特定します。
【公募情報】から、関連性の最も高い研究キーワードを5つ、列挙してください。
研究キーワードはできるだけ短い、10文字以下にしてください。
最後に、研究キーワードだけをスペース区切りのリストで出力してください。

【公募情報】
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

    prompt = preprompt + analyze_data

    # temperature = 0.9
    # top_p = 0.9
    #
    # result = main(prompt, temperature, top_p)
    # print(result)

    temperature = 0.9
    top_p = 0.9

    df_r = pd.DataFrame()

    for temperature in np.arange(0.1, 1.0, 0.1):
        for top_p in np.arange(0.1, 1.0, 0.1):
            try:
                result = main(prompt, temperature, top_p)
                print(temperature, top_p, result)

                df_r.loc[temperature, top_p] = result
            except Exception as e:
                print(e)

    df_r.to_csv('result_Stable_AI.csv')