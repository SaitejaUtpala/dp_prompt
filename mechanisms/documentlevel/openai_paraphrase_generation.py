import traceback

import openai 

from pandarallel import pandarallel
pandarallel.initialize(progress_bar = True, nb_workers=2)
openai.organization = "PLACE_YOUR_ORGANIZATION_KEY_HERE"
openai.api_key = "PLACE_API_KEY_HERE"



def prompt_template_fn(review):
    prompt = f"Review: {review}\nParaphrase of the review:"
    return prompt


def generate(review, temperature):
    prompt = prompt_template_fn(review=review)
    ans = None 
    try : 
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        ans = response
    except Exception as e:
        ans = e
    return ans 

def generate_df(df, df_field, temperature, save_path):
    paraphrased_df = df[df_field].parallel_apply(lambda x: generate(review=x, temperature=temperature))
    paraphrased_df.to_csv(save_path, index=False)
    return paraphrased_df