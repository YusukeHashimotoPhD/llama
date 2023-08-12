# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 1024,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # prompts = [
    #     # For these prompts, the expected answer is the natural continuation of the prompt
    #     "I believe the meaning of life is",
    #     "Simply put, the theory of relativity states that ",
    #     """A brief message congratulating the team on the launch:
    #
    #     Hi everyone,
    #
    #     I just """,
    #     # Few shot prompt (providing a few examples before asking model to complete more);
    #     """Translate English to French:
    #
    #     sea otter => loutre de mer
    #     peppermint => menthe poivrée
    #     plush girafe => girafe peluche
    #     cheese =>""",
    # ]
    # prompts = ['Please summarize the following document.',
    #            'In recent years materials informatics, which is the application of data science to problems in '
    #            'materials science and engineering, has emerged as a powerful tool for materials discovery and design. '
    #            'This relatively new field is already having a significant impact on the interpretation of data for a '
    #            'variety of materials systems, including those used in thermoelectrics, ferroelectrics, battery anodes '
    #            'and cathodes, hydrogen storage materials, polymer dielectrics, etc. Its practitioners employ the '
    #            'methods of multivariate statistics and machine learning in conjunction with standard computational '
    #            'tools (e.g., density-functional theory) to, for example, visualize and dimensionally reduce large '
    #            'data sets, identify patterns in hyperspectral data, parse microstructural images of polycrystals, '
    #            'characterize vortex structures in ferroelectrics, design batteries and, in general, establish '
    #            'correlations to extract important physics and infer structure-property-processing relationships. In '
    #            'this Overview, we critically examine the role of informatics in several important materials '
    #            'subfields, highlighting significant contributions to date and identifying known shortcomings. We '
    #            'specifically focus attention on the difference between the correlative approach of classical data '
    #            'science and the causative approach of physical sciences. From this perspective, we also outline some '
    #            'potential opportunities and challenges for informatics in the materials realm in this era of big data.'
    #            ]

    prompts = ['Please answer to the following question by YES or NO. Does 3 is larger than 2?']

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
