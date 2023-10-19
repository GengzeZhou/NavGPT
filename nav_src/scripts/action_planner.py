import json

from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

from prompt.planner_prompt import (
    PLANNER_PROMPT,
)

from data_utils import construct_instrs

# Using OpenAI davinci-text-003
llm = OpenAI(temperature=0.0)

plan_prompt = PromptTemplate(
    template=PLANNER_PROMPT,
    input_variables=["instruction"],
)

plan_chain = LLMChain(llm=llm, prompt=plan_prompt)


splits = ['val_72']
anno_dir = '../datasets/R2R/annotations'
dataset = 'R2R'
data = construct_instrs(anno_dir, dataset, splits)

for i, sample in enumerate(data):
    print(f"Sample {i}:")
    print(sample['instruction'])
    action_plan = plan_chain.run(sample['instruction'])
    print(action_plan)
    data[i]['action_plan'] = action_plan

with open('../datasets/R2R/annotations/R2R_val_72_action_plan.json', 'w') as f:
    json.dump(data, f, indent=2)