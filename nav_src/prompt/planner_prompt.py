# flake8: noqa

from langchain.prompts.prompt import PromptTemplate

PLANNER_PROMPT = """Given the long instruction: {instruction}

Divide the long instruction into action steps with detailed descriptions in the following format:
Action plan:
1. action_step_1
2. action_step_2
...

Action plan:"""

ACTION_PROMPT = """You are an agent following an action plan to navigation in indoor environment.

Action plan: {action_plan}

You are currently at one of the steps in the plan. You will be given the history of previous steps you have taken, the current observation of the environment, and the navigable viewpoints for the next step.

You should:
1) evaluate the history and observation to decide which step of action plan you are at.
2) choose one viewpoint from the navigable viewpoints.

Each navigable viewpoint has a unique ID, you should only answer the ID in the Final Answer.

----
Starting below, you should strictly follow this format:

History: the history of previous steps you have taken
Observation: the current observation of the environment
Navigable viewpoints: the navigable viewpoints for the next step
Thought: your thought on the next step
Final Answer: 'viepointID'
----

Begin!

History: {history}
Observation: {observation}
Navigable viewpoints: {navigable_viewpoints}
Thought:"""

HISTORY_PROMPT = """You are an agent navigating in indoor environment.

You have reached a new viewpoint after taking previous action. You will be given the navigation history, the current observation of the environment, and the previous action you taken.

You should:
1) evaluate the new observation and history.
2) update the history with the previous action and the new observation.

History: {history}
Previous action: {previous_action}
Observation: {observation}
Update history with the new observation:"""

MAKE_ACTION_TOOL_NAME = "action_maker"
MAKE_ACTION_TOOL_DESCRIPTION = f'Can be used to move to next adjacent viewpoint.\nThe input to this tool should be a viewpoint ID string of the next viewpoint you wish to visit. For example:\nAction: action_maker\nAction Input: "4a153b13a3f6424784cb8e5dabbb3a2c".'

BACK_TRACE_PROMPT = """You are an agent following an action plan to navigation in indoor environment.

You are currently at an intermediate step of the trajectory but seems going off the track. You will be given the action plan describing the whole trajectory, the history of previous steps you have taken, the observations of the viewpoints along the trajectory.

You should evaluate the history, the action plan and the observations along the way to decide the viewpoints to go back to.

Each navigable viewpoint has a unique ID, you should only answer the ID in the Final Answer.
You must choose one from the navigable viewpoints, DO NOT answer None of the above.

----
Starting below, you should follow this format:

Action plan: the action plan describing the whole trajectory
History: the history of previous steps you have taken
Observation: the observations of each viewpoint along the trajectory
Thought: your thought about the next step
Final Answer: 'viewpointID'
----

Begin!

Action plan: {action_plan}
History: {history}
Observation: {observation}
Thought:"""

BACK_TRACE_TOOL_NAME = "back_tracer"
BACK_TRACE_TOOL_DESCRIPTION = f"Can be used to move to any previous viewpoint on the trajectory even if the viewpoint is not adjacent.\nCan be call like {BACK_TRACE_TOOL_NAME}('viewpointID'), where 'viewpointID' is the ID of any previous viewpoint.\nThe input to this tool should be a string of viewpoint ID ONLY."


VLN_ORCHESTRATOR_TOOL_PROMPT = """You are an agent that follows an instruction to navigate in indoor environment. You are required to make sequential decisions according to the observation of the environment to follow the given instruction.
At the beginning of the navigation, you will be given the instruction describing the whole trajectory.
During navigation, you will receive the history of previous steps you have taken, the current observation of the environment at each step.

To navigate in unseen environment is hard, it is possible to go off the track as the description of the instruction.
You should act as a high level controlor, at each step, you should consider whether you are on the right track or not.
If yes, use the action_maker tool to continue.
If not, use the back_tracer tool to move to previous viewpoint on the trajectory.

Here are the descriptions of these tools: {tool_descriptions}

----
Starting below, you should follow this format:

Instruction: the instruction describing the whole trajectory
Initial Observation: the initial observation of the environment
Thought: I should start navigation according to the instruction
Action: action_maker
Action Input: ""
Observation: the result of the action
Thought: you should always think about what to do next
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: ""
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing the instruction.
Final Answer: Finished!

Begin!

Instruction: {action_plan}
Initial Observation: {init_observation}
Thought: I should start navigation according to the instruction
Action: action_maker
Action Input: ""
Observation: {observation}
Thought:{agent_scratchpad}"""

VLN_ORCHESTRATOR_ABS_PROMPT = """You are an agent that follows an instruction to navigate in indoor environment. You are required to make sequential decisions according to the observation of the environment to follow the given instruction.
At the beginning of the navigation, you will be given the instruction describing the whole trajectory.
During navigation, you will receive the history of previous steps you have taken, your current orientation, the current observation of the environment at each step, and the navigable viewpoints' orientations from current viewpoint.
All orientation are normalized in world cooridinate in degrees, you should always consider the relative angle between the observation and navigable viewpoints. i.e. relative angle 0 and 360 are the front, 90 and -270 are the right, 180 and -180 are the back, 270 and -90 are the left.

To navigate in unseen environment is hard, it is possible to go off the track as the description of the instruction. You are allow to back trace but you are encouraged to explore the environment as much as possible. The ultimate goal is to reach the destination in the instruction.
At each step, you should consider:
(1) According to Current Viewpoint observation and History, have you reached the destination?
If yes you should stop, output the 'Final Answer: Finished!' to stop.
If no you should continue:
    (2) Consider whether you are on the right track or not.
    If yes, use the action_maker tool to move to adjacent viewpoint shown in Navigable Viewpoints.
    If not, use the back_tracer tool to move to any previous viewpoint shown in History.
You should always use the action_maker at the begining of navigation. If you are told to wait in the instruction you should output 'Final Answer: Finished!' to stop.

Here are the descriptions of these tools: {tool_descriptions}

The viewpoint ID is a string of 12 characters, for example '4a153b13a3f6424784cb8e5dabbb3a2c'. You are very strict to the viewpoint ID and will never fabricate nonexistent IDs. 

----
Starting below, you should follow this format:

Instruction: the instruction describing the whole trajectory
Initial Observation: the initial observation of the environment
Thought: you should always think about what to do next
Action: the action to take, must be one of the tools [{tool_names}]
Action Input: "Viewpoint ID"
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have reached the destination, I can stop.
Final Answer: Finished!
----

Begin!

Instruction: {action_plan}
Initial Observation: {init_observation}
Thought: I should start navigation according to the instruction, {agent_scratchpad}"""

VLN_ORCHESTRATOR_PROMPT = """You are an agent that follows an instruction to navigate in indoor environment. You are required to make sequential decisions according to the observation of the environment to follow the given instruction.
At the beginning of the navigation, you will be given the instruction describing the whole trajectory.
During navigation, you will receive the history of previous steps you have taken, the current observation of the environment, and the navigable viewpoints' orientations from current viewpoint.
All orientation are in degrees from -180 to 180, i.e. angle 0 is the front, right 90 is 90 degree at the right, right 180 and left 180 are the back, left 90 is 90 degree at the left.

To navigate in unseen environment is hard, it is possible to go off the track as the description of the instruction. You are allow to back trace but you are encouraged to explore the environment as much as possible. The ultimate goal is to reach the destination in the instruction.
At each step, you should consider:
(1) According to Current Viewpoint observation and History, have you reached the destination?
If yes you should stop, output the 'Final Answer: Finished!' to stop.
If no you should continue:
    (2) Consider whether you are on the right track or not.
    If yes, use the action_maker tool to move to adjacent viewpoint shown in Navigable Viewpoints.
    If not, use the back_tracer tool to move to any previous viewpoint shown in History.
You should always use the action_maker at the begining of navigation. Show your reasoning in the Thought section.

Here are the descriptions of these tools: {tool_descriptions}

The viewpoint ID is a string of 12 characters, for example '4a153b13a3f6424784cb8e5dabbb3a2c'. You are very strict to the viewpoint ID and will never fabricate nonexistent IDs. 

----
Starting below, you should follow this format:

Instruction: the instruction describing the whole trajectory
Initial Observation: the initial observation of the environment
Thought: you should always think about what to do next and why
Action: the action to take, must be one of the tools [{tool_names}]
Action Input: "Viewpoint ID"
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have reached the destination, I can stop.
Final Answer: Finished!
----

Begin!

Instruction: {action_plan}
Initial Observation: {init_observation}
Thought: I should start navigation according to the instruction, {agent_scratchpad}"""

VLN_GPT4_PROMPT = """You are an intelligent embodied agent that follows an instruction to navigate in an indoor environment. Your task is to move among the static viewpoints (positions) of a pre-defined graph of the environment, and try to reach the target viewpoint as described by the given instruction with the least steps. 

At the beginning of the navigation, you will be given an instruction of a trajectory which describes all observations and the action you should take at each step.
During navigation, at each step, you will be at a specific viewpoint and receive the history of previous steps you have taken (containing your "Thought", "Action", "Action Input" and "Observation" after the "Begin!" sign) and the observation of current viewpoint (including scene descriptions, objects, and navigable directions/distances within 3 meters).
Orientations range from -180 to 180 degrees: "0" signifies forward, "right 90" rightward, "right (or left) 180" backward, and "left 90" leftward. 

You make actions by selecting navigable viewpoints to reach the destination. You are encouraged to explore the environment while avoiding revisiting viewpoints by comparing current navigable and previously visited IDs in previous "Action Input". The ultimate goal is to stop within 3 meters of the destination in the instruction. If destination visible but the target object is not detected within 3 meters, move closer.
At each step, you should consider:
(1) According to Current Viewpoint observation and History, have you reached the destination?
If yes you should stop, output the 'Final Answer: Finished!' to stop.
If not you should continue:
    (2) Consider where you are on the trajectory and what should be the next viewpoint to navigate according to the instruction.
    use the action_maker tool, input the next navigable viewpoint ID to move to that location.

Show your reasoning in the Thought section.

Here are the descriptions of these tools:
{tool_descriptions}

Every viewpoint has a unique viewpoint ID. You are very strict to the viewpoint ID and will never fabricate nonexistent IDs. 

----
Starting below, you should follow this format:

Instruction: an instruction of a trajectory which describes all observations and the actions should be taken
Initial Observation: the initial observation of the environment
Thought: you should always think about what to do next and why
Action: the action to take, must be one of the tools [{tool_names}]
Action Input: "Viewpoint ID"
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have reached the destination, I can stop.
Final Answer: Finished!
----

Begin!

Instruction: {action_plan}
Initial Observation: {init_observation}
Thought: I should start navigation according to the instruction, {agent_scratchpad}"""

VLN_GPT35_PROMPT = """As an intelligent embodied agent, you will navigate an indoor environment to reach a target viewpoint based on a given instruction, performing the Vision and Language Navigation (VLN) task. You'll move among static positions within a pre-defined graph, aiming for minimal steps.

You will receive a trajectory instruction at the start and will have access to step history (your Thought, Action, Action Input and Obeservation after the Begin! sign) and current viewpoint observation (including scene descriptions, objects, and navigable directions/distances within 3 meters) during navigation. Orientations range from -180 to 180 degrees, with 0 being forward, right 90 rightward, right/left 180 backward, and left 90 leftward.

Explore the environment while avoiding revisiting viewpoints by comparing current and previously visited IDs. Reach within 3 meters of the instructed destination, and if it's visible but no objects are detected, move closer.

At each step, determine if you've reached the destination.
If yes, stop and output 'Final Answer: Finished!'.
If not, continue by considering your location and the next viewpoint based on the instruction, using the action_maker tool.
Show your reasoning in the Thought section.

Follow the given format and use provided tools.
{tool_descriptions}
Do not fabricate nonexistent viewpoint IDs.

----
Starting below, you should follow this format:

Instruction: the instruction describing the whole trajectory
Initial Observation: the initial observation of the environment
Thought: you should always think about what to do next and why
Action: the action to take, must be one of the tools [{tool_names}]
Action Input: "Viewpoint ID"
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have reached the destination, I can stop.
Final Answer: Finished!
----

Begin!

Instruction: {action_plan}
Initial Observation: {init_observation}
Thought: I should start navigation according to the instruction, {agent_scratchpad}"""