import os
import json
import networkx as nx
import math
import numpy as np

# class ImageFeaturesDB(object):
#     def __init__(self, img_ft_file, image_feat_size):
#         self.image_feat_size = image_feat_size
#         self.img_ft_file = img_ft_file
#         self._feature_store = {}

#     def get_image_feature(self, scan, viewpoint):
#         key = '%s_%s' % (scan, viewpoint)
#         if key in self._feature_store:
#             ft = self._feature_store[key]
#         else:
#             with h5py.File(self.img_ft_file, 'r') as f:
#                 ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
#                 self._feature_store[key] = ft
#         return ft

class ImageObservationsDB(object):
    def __init__(self, img_obs_dir, img_obs_sum_dir, img_obj_dir):
        self.img_obs_dir = img_obs_dir
        self.img_obs_sum_dir = img_obs_sum_dir
        self.img_obj_dir = img_obj_dir
        self._obs_store = {}

    def get_image_observation(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._obs_store:
            obs = self._obs_store[key]
        else:
            # Load image observation
            with open(os.path.join(self.img_obs_dir, f'{scan}.json'), 'r') as f:
                obs = json.load(f)[viewpoint]
                self._obs_store[key] = {}
                self._obs_store[key]['detail'] = obs
            # Load image observation summary for history
            with open(os.path.join(self.img_obs_sum_dir, f'{scan}_summarized.json'), 'r') as f:
                obs_sum = json.load(f)[viewpoint]
                self._obs_store[key]['summary'] = obs_sum
            # Load image objects
            with open(os.path.join(self.img_obj_dir, f'{scan}.json'), 'r') as f:
                obj = json.load(f)[viewpoint]
                self._obs_store[key]['objects'] = obj
            obs = self._obs_store[key]
        return obs

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

