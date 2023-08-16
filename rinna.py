import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデルとトークナイザーをグローバルで読み込む
tokenizer = AutoTokenizer.from_pretrained("rinna/bilingual-gpt-neox-4b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/bilingual-gpt-neox-4b-instruction-sft")

if torch.cuda.is_available():
    model = model.to("cuda:1")


def main(prompt, temperature, top_p):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            echo=False
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    return output


if __name__ == '__main__':
    prompt = """
【背景】
研究助成金の対象となる研究のキーワードを特定します。
【公募情報】から、関連性の最も高い研究キーワードを5つ、列挙してください。
研究キーワードはできるだけ短い、10文字以下にしてください。
最後に、研究キーワードだけをスペース区切りのリストで出力してください。

【公募情報】
公募機関名: 八洲環境技術振興財団
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
"""

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

    df_r.to_csv('result_rinna.csv')