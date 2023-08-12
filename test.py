import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/bilingual-gpt-neox-4b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/bilingual-gpt-neox-4b-instruction-sft")

if torch.cuda.is_available():
    model = model.to("cuda")

prompt = [
    {
        "speaker": "ユーザー",
        "text": "Hello, you are an assistant that helps me learn Japanese."
    },
    {
        "speaker": "システム",
        "text": "Sure, what can I do for you?"
    },
    {
        "speaker": "ユーザー",
        "text": "VRはなんですか。"
    }
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in prompt
]
prompt = "\n".join(prompt)
prompt = (
    prompt
    + "\n"
    + "システム: "
)
print(prompt)
"""
ユーザー: Hello, you are an assistant that helps me learn Japanese.
システム: Sure, what can I do for you?
ユーザー: VRはなんですか。
システム:
"""

token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=512*8,
        do_sample=True,
        temperature=1.0,
        top_p=0.85,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
print(output)