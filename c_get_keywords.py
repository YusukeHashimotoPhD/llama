import pandas as pd
import b_call_Llama
import b_talk_Llama


def main():
    model = "./Llama-2-70b-chat-hf"
    pipeline, tokenizer = b_call_Llama.main(model)

    # preprompt = """
    # 以下の研究助成金に該当する研究者の研究キーワードを最大５つ提示してください。
    # 研究キーワードだけをスペース区切りのリストで出力してください。
    # 研究キーワードはできるだけ短いものにしてください。
    #
    # ーーー
    # """

    preprompt = """
From the following sentences, extract 5 research keywords and give them separated by comma in Japanese.
Example of output:
Keyword_1, Keyword_2, Keyword_3, Keyword_4, Keyword_5

"""

#     preprompt = """
# 以下の研究助成金に該当する研究者の研究キーワードを最大５つ提示してください。
# 研究キーワードだけをスペース区切りの文字列で出力してください。
# 研究キーワードはできるだけ短いものにしてください。
#
# ーーー
# """

#     postprocess = """
# 研究キーワードだけを抽出し、スペース区切りの文字列で出力してください。
#
# ーーー
# """
#
    postprocess = """
Extract 5 research keywords and give them separated by comma in Japanese.

ーーー
"""

    input_file_path = './data/koubo_analyzed.csv'
    output_file_path = './data/koubo_analyzed_1.csv'
    df = pd.read_csv(input_file_path, index_col=0)
    use_template = False

    for index in df.index:
#        if str(df.loc[index, 'キーワード']) == 'nan':
        main_prompt = df.loc[index, 'まとめ']
        input_prompt = preprompt + main_prompt
        # for i in range(3):
        results = b_talk_Llama.main(pipeline, tokenizer, input_prompt, use_template=True)
            # input_prompt = preprompt + results[0]
        keywords = results[0].split('### Response:')[1]
        input_prompt = postprocess + keywords
        results = b_talk_Llama.main(pipeline, tokenizer, input_prompt, use_template=False)
        df.loc[index, 'キーワード_hf'] = results[0]
        # df.loc[index, 'キーワード_hf'] = results[0]
        print(index, df.loc[index, 'キーワード_hf'])

        df.to_csv(output_file_path)


if __name__ == '__main__':
    main()