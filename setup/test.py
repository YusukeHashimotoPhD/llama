# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

tokenizer = AutoTokenizer.from_pretrained("./Llama-2-13b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("./Llama-2-13b-chat-hf")
revision = "0ede8dd71e923db6258295621d817ca8714516d4"

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
#     revision=revision,
#     return_full_text=False
# )

# Define prompt template, the format below is from: http://fastml.com/how-to-train-your-own-chatgpt-alpaca-style-part-one/

# Prompt templates as follows could guide the model to follow instructions and respond to the input, and empirically it turns out to make Falcon models produce better responses

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


# Define parameters to generate text
def gen_text(prompts, use_template=False, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1

    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,
            # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs
