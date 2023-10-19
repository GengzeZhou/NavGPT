''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import random
import networkx as nx
from collections import defaultdict

from utils.data import load_nav_graphs
from eval_utils import cal_dtw, cal_cls
from utils.graph_utils import NavGraph

ERROR_MARGIN = 3.0

class Simulator(object):
    ''' A simple simulator in Matterport3D environment '''

    def __init__(
            self,
            navigable_dir: str,):
        self.heading = 0
        self.elevation = 0
        self.scan_ID = ''
        self.viewpoint_ID = ''
        self.navigable_dir = navigable_dir
        self.navigable_dict = {}
        self.candidate = {}
        self.gmap = NavGraph()

    def newEpisode(
            self, 
            scan_ID: str, 
            viewpoint_ID: str,
            heading: int, 
            elevation: int,):
        self.heading = heading
        self.elevation = elevation
        self.scan_ID = scan_ID
        self.viewpoint_ID = viewpoint_ID
        # Load navigable dict
        navigable_path = os.path.join(self.navigable_dir, self.scan_ID + '_navigable.json')
        with open(navigable_path, 'r') as f:
            self.navigable_dict = json.load(f)
        # Get candidate
        self.getCandidate()
    
    def updateGraph(self):
        # build graph
        for candidate in self.candidate.keys():
            self.gmap.update_connection(self.viewpoint_ID, candidate)

    def getState(self) -> dict:
        self.state = {
            'scanID': self.scan_ID,
            'viewpointID': self.viewpoint_ID,
            'heading': self.heading,
            'elevation': self.elevation,
            'candidate': self.candidate,
        }
        return self.state
    
    def getCandidate(self):
        """
        Get the agent's candidate list from pre-stored navigable dict.
        """
        self.candidate = self.navigable_dict[self.viewpoint_ID]
        self.updateGraph()
    
    def makeAction(self, next_viewpoint_ID):
        """
        Make action and update the agent's state.
        """
        if next_viewpoint_ID == self.viewpoint_ID:
            return
        elif next_viewpoint_ID in self.candidate.keys():
            self.heading = self.candidate[next_viewpoint_ID]['heading']
            self.elevation = self.candidate[next_viewpoint_ID]['elevation']
        self.viewpoint_ID = next_viewpoint_ID
        self.getCandidate()


class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, navigable_dir, feat_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        
        self.sims = []
        for i in range(batch_size):
            sim = Simulator(navigable_dir)
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            feature = self.feat_db.get_image_observation(state["scanID"], state["viewpointID"])
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, next_viewpoint_IDs):
        ''' Take an action using the full state dependent action interface (with batched input)'''
        for i, next_viewpoint_ID in enumerate(next_viewpoint_IDs):
            self.sims[i].makeAction(next_viewpoint_ID)


class R2RNavBatch(object):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, view_db, instr_data, connectivity_dir, navigable_dir,
        batch_size=1, seed=0, name=None,
    ):
        self.env = EnvBatch(navigable_dir, feat_db=view_db, batch_size=batch_size)
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.name = name

        self.gt_trajs = self._get_gt_trajs(self.data) # for evaluation

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self._load_nav_graphs()
        
        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def _get_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
                for x in data if len(x['path']) > 1
        }
        return gt_trajs

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            ob = {
                'obs' : feature["detail"],
                'obs_summary' : feature["summary"],
                'objects' : feature["objects"],
                'instr_id' : item['instr_id'],
                # 'action_plan' : item['action_plan'],
                'scan' : state['scanID'],
                'viewpoint' : state['viewpointID'],
                'heading' : state['heading'],
                'elevation' : state['elevation'],
                'candidate': state['candidate'],
                'instruction' : item['instruction'],
                'gt_path' : item['path'],
                'path_id' : item['path_id']
            }
            # RL reward. The negative distance between the state and the final state
            # There are multiple gt end viewpoints on REVERIE. 
            if ob['instr_id'] in self.gt_trajs:
                ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            else:
                ob['distance'] = 0

            obs.append(ob)
        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, next_viewpoint_IDs):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(next_viewpoint_IDs)
        return self._get_obs()

    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }
        return avg_metrics, metrics
        
