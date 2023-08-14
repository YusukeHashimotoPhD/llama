import pandas as pd

import Stable_AI
import a_control_Llama
import rinna

preprompt = """
【背景】
研究助成金の対象となる研究のキーワードを特定します。
【公募情報】から、関連性の最も高い研究キーワードを5つ、列挙してください。
研究キーワードはできるだけ短い、10文字以下のものにしてください。
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

temperature = 0.9
top_p = 0.9

df_r = pd.DataFrame()

model_name = 'Stable_AI'
try:
    result = Stable_AI.main(prompt, temperature, top_p)
    df_r.loc[prompt, model_name] = result
except Exception as e:
    print('Stable_AI', e)

model_name = 'rinna'
try:
    result = rinna(prompt, temperature, top_p)
    print(result)
    df_r.loc[prompt, 'rinna'] = result
except Exception as e:
    print('rinna', e)

# list_model_name = ['Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Llama-2-70b-chat-hf']
#
# for model_name in list_model_name:
#     try:
#         results = a_control_Llama.main(model_name, prompt)
#         df_r.loc[prompt, model_name] = results[0]
#     except Exception as e:
#         print(model_name, e)

df_r.to_csv('df_r.csv')
print(df_r)