# # Use a pipeline as a high-level helper
# from transformers import pipeline
#
# pipe = pipeline("text-generation", model="./model/line/")
#
# temperature = 0.9
# top_p=0.9
#
# prompt='1+2='
#
# prompt_format = f"""
# SYSTEM: You are a helpful, respectful and honest assistant.
# Always answer as helpfully as possible, while being safe.
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
# If you don't know the answer to a question, please don't share false information.
# Shorter answer is better.
# Do not include the prompt in the answer.
# USER: {prompt}
# ASSISTANT:
# """
#
# output = model(
#     prompt_format,
#     temperature=temperature,
#     top_p=top_p,
#     stop=["SYSTEM:", "USER:", "ASSISTANT:", "\n"],
#     echo=False,
# )
#
# output["choices"][0]["text"]


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-3.6b", torch_dtype=torch.float16)
# float16は指定しなくても問題ありません
tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-3.6b", use_fast=False)
# use_fast=False は必ず付与してください。なくても動きますが、我々の学習状況とは異なるので性能が下がります。
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
set_seed(101)

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

#prompt = '1+２='

text = generator(
    prompt,
    max_length=30,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    num_return_sequences=5,
)

for t in text:
    print(t)

# 下記は生成される出力の例
# [{'generated_text': 'おはようございます、今日の天気は雨模様ですね。梅雨のこの時期の 朝は洗濯物が乾きにくいなど、主婦にとっては悩みどころですね。 では、'},
#  {'generated_text': 'おはようございます、今日の天気は晴れ。 気温は8°C位です。 朝晩は結構冷え込むようになりました。 寒くなってくると、...'},
#  {'generated_text': 'おはようございます、今日の天気は曇りです。 朝起きたら雪が軽く積もっていた。 寒さもそれほどでもありません。 日中は晴れるみたいですね。'},
#  {'generated_text': 'おはようございます、今日の天気は☁のち☀です。 朝の気温5°C、日中も21°Cと 暖かい予報です'},
#  {'generated_text': 'おはようございます、今日の天気は晴天ですが涼しい1日です、気温は午後になり低くなり25°Cくらい、風も強いようですので、'}]