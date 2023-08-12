import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1")

model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/japanese-stablelm-base-alpha-7b",
    trust_remote_code=True,
)
model.half()

if torch.cuda.is_available():
    model = model.to("cuda")

prompt = """
AI で科学研究を加速するには、
""".strip()

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

prompt = preprompt + analyze_data

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
    max_new_tokens=128,
    temperature=1,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=False)
print(out)
"""
 AI で科学研究を加速するには、データ駆動型文化が必要であることも明らかになってきています。研究のあらゆる側面で、データがより重要になっているのです。
20  世紀の科学は、研究者が直接研究を行うことで、研究データを活用してきました。その後、多くの科学分野ではデータは手動で分析されるようになったものの、これらの方法には多大なコストと労力がかかることが分かりました。 そこで、多くの研究者や研究者グループは、より効率的な手法を開発し、研究の規模を拡大してきました。21 世紀になると、研究者が手動で実施する必要のある研究は、その大部分を研究者が自動化できるようになりました。
"""
