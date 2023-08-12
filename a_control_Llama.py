import b_call_Llama
import b_talk_Llama


def main(prompt):
    # model = "./Llama-2-13b-hf"
    model = "./Llama-2-70b-chat-hf"
    pipeline, tokenizer = b_call_Llama.main(model)
    use_template = False
    return b_talk_Llama.main(pipeline, tokenizer, prompt, use_template)


if __name__ == '__main__':
    # prompt = '''
    # By the following sentence, extract 5 keywords and give them separated by comma in Japanese.
    # Only the results are given.
    # Your comment is not welcome.
    #
    # 以下は、八洲環境技術振興財団の2023年度研究開発・調査助成に該当する研究者の研究キーワードです。
    #
    # 1. 再生可能エネルギー源
    # 2. クリーン燃料
    # 3. エネルギー効率化
    # 4. エネルギー材料
    # 5. 環境保全
    #
    # '''


    preprompt = """
<<SYS>>
あなたは、とても親切で有能なアシスタントです。
英語は禁止で、日本語だけ使えます。
キーワード以外のコメントは出力できません。

以下の研究助成金に該当する研究者の研究キーワードを最大５つ提示してください。
研究キーワードだけをスペース区切りのリストで出力してください。
研究キーワードはできるだけ短いものにしてください。
<</SYS>>

"""

#     preprompt = '''
# <<SYS>>
# You are an assistant for scientific research.
# By the following sentence, extract 5 keywords and give them separated by comma in Japanese.
# The output is only the extracted keywords.
# No comment is allowed.
# <</SYS>>
# '''

    analyze_data = '''
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
'''

    prompt = preprompt + f'[INST] {analyze_data} [/INST]'

    results = main(prompt)
    print(results[0])