import concurrent.futures
import re
import time
from string import Template

import pandas as pd
from litellm import completion
from tqdm import tqdm


def get_answer(
    instruction: str,
    prompt: str,
    reference: str,
    metadata: str,
    lang: str,
):

    while True:
        try:
            response = completion(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {
                        "role": "user",
                        "content": instruction,
                    },
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            output = response.choices[0].message.content
            # find either [[YES]] or [[NO]] in the output with a robust regular expression
            decision = re.search(r"\[\[(YES|NO)\]\]", output)
            if decision:
                decision = decision.group(1)
                if decision == "YES":
                    decision = "[[YES]]"
                else:
                    decision = "[[NO]]"
            else:
                raise ValueError("Output does not contain [[YES]] or [[NO]]")
            print(output)
            print(decision)
            break
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(1)
    return {
        "instruction": instruction,
        "prompt": prompt,
        "reference": reference,
        "metadata": metadata,
        "lang": lang,
        "decision": decision,
        "full_output": output,
    }


prompt = Template(
    """Consider this meta prompt we wrote to as a model to generate a data instance:

```meta_prompt
You are tasked with creating a diverse and engaging prompt for a chatbot arena. This prompt will be used to test and compare the capabilities of different language models. Your goal is to generate a question or prompt that will challenge these models and showcase their strengths or weaknesses. Also, generate a reference answer to your prompt that will serve as a benchmark for evaluating the models' responses.

Here are the input variables you will use to craft your prompt:
- Language: ${language}
- Topic: ${topic}
- Subtopic: ${subtopic}
- Difficulty: ${difficulty}
- Style: ${style}
- Writer: ${writer}
- Writing proficiency: ${writing_proficiency}
- Prompt length: ${length}

Guidelines for creating the prompt:

1. Abide strictly by the input variables provided.

2. Ensure that your prompt is open-ended enough to allow for varied and interesting responses from different language models.

3. Avoid prompts that are overly specific to a particular AI model's capabilities or training data.

4. Create a prompt that is engaging and thought-provoking, encouraging creative or analytical thinking.

Generate output in the following format:

<START OF PROMPT>
[Your generated prompt here. IMPORTANT: include only the prompt.]
<END OF PROMPT>
```

This is the data instance it generated:

```data_instance
<START OF PROMPT>
${prompt}
<END OF PROMPT>
```

Classify this reponse with a [[YES]] or [[NO]], based on whether it obeys the input variables that were specified at the beginning of the meta prompt.

Make sure to include either [[YES]] or [[NO]] in your response, but NEVER both."""
)

langs = ["english", "chinese_s", "french", "korean"]

dfs = []
for l in langs:
    df = pd.read_json(
        f"../../data/general_capabilities_{l}.jsonl",
        lines=True,
    )
    df["lang"] = l
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

df_dict = df.to_dict(orient="records")

final_data = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for instance in df_dict:
        out_prompt = instance["prompt"]
        instruction = prompt.substitute(
            language=instance["metadata"]["language"],
            topic=instance["metadata"]["topic"],
            subtopic=instance["metadata"]["subtopic"],
            difficulty=instance["metadata"]["difficulty"],
            style=instance["metadata"]["style"],
            writer=instance["metadata"]["writer"],
            writing_proficiency=instance["metadata"]["writing_proficiency"],
            length=instance["metadata"]["length"],
            prompt=out_prompt,
        )
        reference = instance["reference"]
        metadata = instance["metadata"]
        lang = instance["lang"]
        future = executor.submit(
            get_answer,
            instruction=instruction,
            prompt=out_prompt,
            reference=reference,
            metadata=metadata,
            lang=lang,
        )
        futures.append(future)

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        final_data.append(future.result())

# save the results to a JSON file
out_df = pd.DataFrame(final_data)
out_df.to_json(
    "claude_general_purpose_chat_meta_prompt_attributes_cohesion.json",
    orient="records",
    lines=True,
)
