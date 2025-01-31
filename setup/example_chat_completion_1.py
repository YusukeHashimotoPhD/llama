# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

#     dialogs = [
#         [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
#         [
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:
#
# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
#
# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ],
#         [
#             {"role": "system", "content": "Always answer with Haiku"},
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Always answer with emojis",
#             },
#             {"role": "user", "content": "How to go from Beijing to NY?"},
#         ],
#     ]

    dialogs = [
        # [
        #     {"role": "system", "content": "You are a genius scientist. Answer only results."},
        #     {"role": "user", "content": "1 + 3 = "},
        # ],
        # [
        #     {"role": "system", "content": "You are a genius scientist. Answer only results."},
        #     {"role": "user", "content": "5 * 3 = "},
        # ],
        # [
        #     {"role": "system", "content": "You are a genius scientist. Answer only results."},
        #     {"role": "user", "content": "16 / 4 = "},
        # ],
        [
            {"role": "system", "content": "You are a genius scientist. Answer only results."},
            {"role": "user", "content": "1 / 0 = "},
        ],
        [
            {"role": "system", "content": "あなたは、とても有能なアシスタントです。日本語で答えてください。"},
            {"role": "user", "content": "大規模言語モデルとはなんですか？"},
        ],
        # [
        #     {"role": "system", "content": "You are a genius scientist. Always answer by YES or NO"},
        #     {"role": "user", "content": "Is 1 larger than 2?"},
        # ],
    ]

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
