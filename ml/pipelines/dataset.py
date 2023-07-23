from torch.utils.data import Dataset
import torchvision.transforms as transform
import torch
import os.path as osp
import json

from sampleframes import SampleFrames

from PIL import Image

class SignDataset(Dataset):
    """Samples frames using MMAction's SampleFrames and handles the amalgam annotaion
    format.

    Example of a annotation file:
    .. code-block:: txt
        {'accident': [{'name': '00625',
            'dataset': 'wlasl',
            'class_number': '51',
            'split': 'test',
            'frames': 35,
            'location': 'test/00625'},
            {'name': '00634',
            'dataset': 'wlasl',
            'class_number': '51',
            'split': 'test',
            'frames': 35,
            'location': 'test/00634'}
        }

    Required keys are "ann_file", "root_dir", "split" and "clip_len".
    Args:
        ann_file (str): Path to annotation file.
        root_dir (str): Root directory of the rawframes.
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        rgb_prefix (str): File format for rgb image files.
        split (str): Split of the dataset e.g. 'test', 'train' etc.
    """

    def __init__(self,                
                ann_file,  
                root_dir,
                split,
                clip_len,
                test_mode,
                resolution=256,
                transforms=None,
                frame_interval=1,
                num_clips=1,
                rgb_prefix =  'img_{:05}.jpg',
                ):

        self.ann_file = ann_file
        self.root_dir = root_dir
        self.rgb_prefix = rgb_prefix
        self.transforms = transforms
        self.split = split

        self.video_infos = self.load_annotations()
        self.sample_frames = SampleFrames(clip_len=clip_len,
                                        frame_interval=frame_interval,
                                        num_clips=num_clips,
                                        test_mode=test_mode)

        self.img2tensorTransforms = transform.Compose(
                                                [
                                                    transform.Resize((resolution, resolution)),
                                                    transform.ToTensor(),
                                                ]
                                            )

    def __len__(self):
        return len(self.video_infos)
    
    def load_annotations(self):
        """Load annotation file to get video information."""
        with open(self.ann_file, 'r') as fin:
            json_dict = json.load(fin)

        for gloss in json_dict:
            video_infos = []
            for video in json_dict[gloss]:
                if video['split'] in self.split:
                    video_info = dict()
                    video_info['video_path'] = osp.join(self.root_dir, self.split,video['name'])
                    video_info['start_index'] = 1
                    video_info['total_frames'] = video['frames']
                    video_info['label'] = int(video['class_number'])
                    video_infos.append(video_info)
        return video_infos


    def load_video(self, idx):
        """Load a video at a particular index and return rgb, flow, depth and 
        pose data in a dictionary.
        
        Args: 
            idx (int): The index position in the annotation file
            corresponding to a video.
        Returns:
            results (dict): The dictionary containing all the video data.
        """
        video_info = self.video_infos[idx]
        results = dict()
        results.update(video_info)
        
        self.sample_frames(results)
        frame_indices = results['frame_inds']
        video_path = results['video_path']

        rgb_frames = []

        cache = dict()

        for frame in frame_indices:
            if frame not in cache:
                rgb_frame = Image.open(osp.join(video_path, self.rgb_prefix.format(frame)))

                # Add frames to cache
                cache[frame] = dict(rgb_frame=rgb_frame)
                
                rgb_frames.append(rgb_frame)
                
            else:
                rgb_frames.append(cache[frame]['rgb_frame'])

        results['rgb'] = rgb_frames

        return results
        

    def to_3dtensor(self, images):
        image_tensors = []
        for img in images:
            image_tensors.append(self.img2tensorTransforms(img).unsqueeze(dim=1))
        tensor = torch.cat(image_tensors, dim = 1)
        return tensor

        
    def __getitem__(self, idx):
        output = dict()
        results = self.load_video(idx=idx)
        if(self.transforms != None):
            results = self.transforms(results)
            
        rgb = self.to_3dtensor(results['rgb'])
    
        label = torch.tensor(results['label'])

        output['rgb'] = rgb
        output['label'] = label

        return rgb, label

        