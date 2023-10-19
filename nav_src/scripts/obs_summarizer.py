'''
Use LLM chain to summarize the observations
'''
import os
import json
import asyncio
import argparse

from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

async def async_generate(chain, viewpointID, ob_list):
    print(f"Summarizing {viewpointID} ...")
    tasks = [chain.arun(description=ob) for ob in ob_list]
    resp_list = await asyncio.gather(*tasks)
    print(f"Summarized {viewpointID}'s observations: {resp_list}\n")
    return resp_list


async def generate_concurrently(chain, obs):
    tasks = [async_generate(chain, viewpointID, ob) for viewpointID, ob in obs.items()]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--obs_dir", type=str, default="../datasets/R2R/observations_list/")
    parser.add_argument("--output_dir", type=str, default="../datasets/R2R/observations_list_summarized/")
    parser.add_argument("--sum_type", type=str, default="list", choices=["list", "single"])
    args = parser.parse_args()

    obs_dir = args.obs_dir
    obs_files = os.listdir(obs_dir)
    output_dir = args.output_dir
    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    llm = OpenAI(
        temperature=0.0,
        model_name="gpt-3.5-turbo",
        )

    if args.sum_type == "single":
        summarize_prompt = PromptTemplate(
            template='Given the description of a viewpoint. Summarize the scene from the viewpoint in one concise sentence.\n\nDescription:\n{description}\n\nSummarization: The scene from the viewpoint is a',
            input_variables=["description"],
    )
    elif args.sum_type == "list":
        summarize_prompt = PromptTemplate(
            template='Here is a single scene view from top, down and middle:\n{description}\nSummarize the scene in one sentence:',
            input_variables=["description"],
        )

    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

    for obs_file in obs_files:
        obs_path = os.path.join(obs_dir, obs_file)
        with open(obs_path) as f:
            obs = json.load(f)
        summary = {}
        viewpointIDs = list(obs.keys())
        # Get the viewpointIDs in batches
        for i in range(0, len(viewpointIDs), args.batch_size):
            batch = viewpointIDs[i:i+args.batch_size]
            print(f"Summarizing scan {obs_file.split('.')[0]} batch [{i//args.batch_size}/{len(viewpointIDs)//args.batch_size}]")
            batch_obs = {viewpointID:obs[viewpointID] for viewpointID in batch}
            summarized_obs = asyncio.run(generate_concurrently(summarize_chain, batch_obs))
            summarized_obs = {viewpointID: summarized_obs[i] for i, viewpointID in enumerate(batch)}
            summary.update(summarized_obs)
        output_path = os.path.join(output_dir, f'{obs_file}.json')
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)