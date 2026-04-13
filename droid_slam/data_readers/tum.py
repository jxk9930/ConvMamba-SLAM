import numpy as np
import os
import os.path as osp
from .base import RGBDDataset

class TUMDataset(RGBDDataset):
    @staticmethod
    def is_test_scene(scene):
        return False

    def __init__(self, name='TUM', datapath=None, **kwargs):
        self.datapath = datapath
        self.n_frames = kwargs.get('n_frames', 7)
        super(TUMDataset, self).__init__(name, datapath, **kwargs)

    def _build_dataset(self):
        scene_info = {}
        scenedir = self.datapath
        scene_name = os.path.basename(scenedir.rstrip('/')) 
        
        # Load files
        img_list = sorted([osp.join(scenedir, 'rgb', f) for f in os.listdir(osp.join(scenedir, 'rgb')) if f.endswith('.png')])
        depth_list = sorted([osp.join(scenedir, 'depth', f) for f in os.listdir(osp.join(scenedir, 'depth')) if f.endswith('.png')])
        poses = np.loadtxt(osp.join(scenedir, 'groundtruth.txt'), delimiter=' ')
        
        # Clamp lengths 
        min_len = min(len(img_list), len(depth_list), len(poses))
        img_list = img_list[:min_len]
        depth_list = depth_list[:min_len]
        poses = poses[:min_len, 1:] 
        
        # Intrinsics
        intrinsics = [535.4, 539.2, 320.1, 247.6]

        # Build a basic sequential graph for base.py
        # This links every frame to its immediate neighbors
        graph = {}
        for i in range(min_len):
            graph[i] = [j for j in range(min_len) if i != j and abs(i - j) <= 2]

        # Pack into the dictionary base.py expects!
        scene_info[scene_name] = {
            'images': img_list,
            'depths': depth_list,
            'poses': poses,
            'intrinsics': intrinsics,
            'graph': graph
        }
        
        return scene_info
