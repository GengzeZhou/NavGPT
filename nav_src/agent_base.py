import json
import os

class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path']})
            if detailed_output:
                output[-1]['details'] = v['details']
                output[-1]['action_plan'] = v['action_plan']
                output[-1]['llm_output'] = v['llm_output']
                output[-1]['llm_thought'] = v['llm_thought']
                output[-1]['llm_observation'] = v['llm_observation']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        # self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
                    preds_detail = self.get_results(detailed_output=True)
                    json.dump(
                    preds_detail,
                    open(os.path.join(self.config.log_dir, 'runtime.json'), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                    )
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                        preds_detail = self.get_results(detailed_output=True)
                        json.dump(
                        preds_detail,
                        open(os.path.join(self.config.log_dir, 'runtime.json'), 'w'),
                        sort_keys=True, indent=4, separators=(',', ': ')
                        )
                if looped:
                    break
