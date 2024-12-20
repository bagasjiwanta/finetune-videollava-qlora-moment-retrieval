import os 
import shutil
from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from datasets.utils.logging import set_verbosity_error, set_verbosity_info
import numpy as np
from decord import VideoReader, cpu
from bisect import bisect_left, bisect_right
import logging 

class DatasetPreparer():

    def __init__(self,     
        base_dir: str = "dataset",
        processed_dir: str = "processed",
        processor = None,
        num_frames = 14,
        num_worker = 1
    ):
        r'''
        Parameters:
            base_dir (str): root dir for dataset operations
            processed_dir (str): dir for dataset that has been processed
            processor (VideoProcessor): VideoLlavaProcessor or similar type
        '''
        self.repo_id = "jwnt4/a-temporal-upgrade"
        self.base_dir = base_dir
        self.processed_dir = processed_dir
        self.processor = processor
        self.num_frames = num_frames # default value
        self.num_worker = num_worker


    def read_video_decord(self, video_path):
        '''
        Decode the video with Decord decoder.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        vr = VideoReader(uri=video_path, ctx=cpu(0)) # you need to install from source to use gpu ctx
        step = len(vr) / self.num_frames
        indices = np.arange(0, len(vr) - step, len(vr) / (self.num_frames - 1)).astype(int)
        indices = np.append(indices, len(vr)-1)
        frames = vr.get_batch(indices).asnumpy()
        return (frames, np.average(vr.get_frame_timestamp(indices), axis=1))


    def pre_collate_ao_v2(self, e, use_robust=False, train=True):
        video_file = e["video_id"] + ".mp4"
        video_clip, _indices = self.read_video_decord(f'{self.base_dir}/videos/action_ordering_v2/{video_file}')
        question = e['question_robust'] if use_robust else e['question_normal']
        text = f"USER: <video> {question} ASSISTANT:"
        if train:
            text += f" {', '.join(e['answer'])}"

        batch = self.processor(
            text=text,
            videos=video_clip,
            truncation=True,
            max_length=4096,
            return_tensors='pt'
        )
        if not train:
            batch['answer'] = e['answer'] # for labels

        return batch


    def to_str_timestamp(self, seconds):
        m, s = divmod(seconds, 60)
        return f"{round(m):02d}:{round(s):02d}"


    def pre_collate_mr_v2(self, e, use_frame=False, train=True, mr_max_actions=1):
        '''
        Collates and process moment_retrieval_v2 dataset
        Parameters:
            frame (`bool`): indicate whether to use frame-style or timestamp-style prompt
            train (`bool`): is this set belongs to a train dataset
            num_frames (`int`): number of frames sampled
            idx (`int`): which moment index to pick (0 to 3). To pick the largest moment, set to -1 (default)
        '''
        video_file = e['video_id'] + '.mp4'
        video_clip, indices = self.read_video_decord(f'{self.base_dir}/videos/moment_retrieval_v2/{video_file}')
        prompt = e['prompt_frame'] if use_frame else e['prompt_timestamp']
        prompt = str(prompt).replace("<num_frames>", str(self.num_frames))
        action = e['actions'][mr_max_actions - 1] # if max_actions is 1, select first, etc
        answer = e['answers'][mr_max_actions - 1]
        prompt = prompt.replace('<action>', action)
        
        if not use_frame:
            frames = [[i, self.to_str_timestamp(indices[i])] for i in range(len(indices))]
            frame_info = '\n'.join([f" - Frame {f[0] + 1}: {f[1]}" for f in frames])
            prompt = prompt.replace('<frame_info>', frame_info)
            prompt = prompt.replace('<duration>', str(round(e['duration'])))
        
        prompt = f"USER: <video> {prompt} ASSISTANT:"
        
        st_found, ed_found = False, False
        st, ed = answer[0], answer[1]
        for i in range(len(indices)): # (start, end) should be one of the timestamps
            if  not st_found and st > indices[i]:
                st = i + 1 if i == self.num_frames - 1 else i + 2
                if not use_frame:
                    st = self.to_str_timestamp(indices[st])
                st_found = True 
            if not ed_found and ed < indices[i]:
                ed = 1 if i == 0 else i
                if not use_frame:
                    ed = self.to_str_timestamp(indices[ed])
                ed_found = True
        
        if not ed_found:
            if not use_frame:
                ed = self.to_str_timestamp(indices[self.num_frames - 1])
            else:
                ed = self.num_frames
        if not st_found:
            if not use_frame:
                st = self.to_str_timestamp(indices[0])
            else:
                st = 1
        
        if use_frame:
            st = str(st)
            ed = str(ed)

        if train:
            prompt += ', '.join([st, ed])

        batch = self.processor(
            text=prompt,
            videos=video_clip,
            truncation=True,
            max_length=4096,
            return_tensors='pt'
        )

        if not train:
            batch['answer'] = [st, ed]
            if not use_frame:
                batch['ts_info'] = frames

        return batch
    

    def pre_collate_mr_ts(self, e, use_frame=False, train=False):
        # read video
        video_file = e['video_id'] + '.mp4'
        video_clip, v_ts = self.read_video_decord(f'{self.base_dir}/videos/moment_retrieval_v2/{video_file}')

        prompt = e['prompt_frame'] if use_frame else e['prompt_timestamp']
        prompt = prompt.replace("<num_frames>", str(self.num_frames))
        prompt = prompt.replace('<action>', e['action'])

        if not use_frame:
            timestamps = [self.to_str_timestamp(v_ts[i]) for i in range(len(v_ts))]
            ts_infos = '\n'.join([f" - Frame {f[0] + 1}: {f[1]}" for f in enumerate(timestamps)])
            prompt = prompt.replace('<frame_info>', ts_infos)
            prompt = prompt.replace('<duration>', str(round(e['duration'])))

        prompt = f"USER: <video> {prompt} ASSISTANT:"
        st = bisect_left(v_ts, e['answer'][0])
        ed = bisect_right(v_ts, e['answer'][1])
        
        if not use_frame: 
            st = self.to_str_timestamp(v_ts[st])
            ed = self.to_str_timestamp(v_ts[ed])
        else:
            st = str(st)
            ed = str(ed)

        if train:
            prompt += ', '.join([st, ed])

        batch = self.processor(
            text=prompt,
            videos=video_clip,
            truncation=True,
            max_length=4096,
            return_tensors='pt'
        )

        if not train and not use_frame:
            batch['ts_info'] = timestamps

        return batch


    def download_videos(self, dataset, split):
        logger = logging.getLogger(__name__)
        video_ids = set(dataset['train']['video_id'] + dataset['validation']['video_id'] + dataset['test']['video_id'])
        target_dir = f"{self.base_dir}/videos/{split}"
        
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            present_videos = set()
        else:
            present_videos = set([d[:-4] for d in os.listdir(target_dir)])

        if len(video_ids & present_videos) == len(video_ids):
            logger.info("info: all videos are present, skip downloading")
            return
    
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir) # clean up improper downloads
        
        logger.info(f"info: downloading videos for {split}")
        download_path = hf_hub_download(
            self.repo_id, filename=f"videos/{split}.zip", repo_type='dataset', local_dir=self.base_dir
        )
        shutil.unpack_archive(download_path, f"{self.base_dir}/videos/", 'zip')
        os.remove(f"{self.base_dir}/videos/{split}.zip")

        logger.info(f"info: success download at path {target_dir}")


    def prepare_dataset(self, split="action_ordering_v2", use_frame=True, use_robust=False, mr_max_actions=1):
        r'''
        Download dataset and videos if not exist in storage, samples num_frames of frames from each video, tokenize, and return it.
        
        Videos are downloaded to is {base_dir}/videos/{split}

        The dataset is then saved to {base_dir}/{split}/{num_frames}_frames/{dataset_config}

        The dataset_config depends on the split used, for action_ordering_v2, the config are robust 

        Parameters:
            split (`str`): dataset split to prepare. Possible values are "action_ordering_v2", "moment_retrieval_v2"
            use_robust (`bool`): whether to use robust dataset on action_ordering
            use_frame (`bool`): whether to use frame prompts on moment_retrievals
            mr_max_actions (`int`): how many actions to download on moment_retrieval_v2
        '''
        logger = logging.getLogger(__name__)
        
        set_verbosity_error()
        if split != 'moment_retrieval':
            ds = load_dataset(self.repo_id, split).sort("complexity")
        else:
            ds = load_dataset(self.repo_id, split)
        self.download_videos(ds, split)

        logger.info("info: collating")
        if split == "action_ordering_v2":
            removed_columns = ['video_id', 'duration', 'captions_starts', 'captions_ends', 'question_normal', 'question_robust', 'answer', 'complexity']
            default_kwargs = {"batched": False, "num_proc": self.num_worker, "writer_batch_size": 400, "remove_columns": removed_columns}

            ds['train'] = ds['train'].map(
                self.pre_collate_ao_v2,
                fn_kwargs={"train": True, "use_robust": use_robust},
                **default_kwargs
            ).with_format("torch")

            ds['validation'] = ds['validation'].map(
                self.pre_collate_ao_v2,
                **default_kwargs,
                fn_kwargs={"train": False, "use_robust": use_robust},
            ).with_format("torch")

            ds['test'] = ds['test'].map(
                self.pre_collate_ao_v2,
                fn_kwargs={"train": False, "use_robust": use_robust},
                remove_columns=removed_columns
            ).with_format("torch")

        elif split == "moment_retrieval_v2":
            removed_columns = ['video_id', 'duration', 'prompt_frame', 'prompt_timestamp', 'complexity', 'answers', 'actions']
            default_kwargs = {"batched": False, "num_proc": self.num_worker, "writer_batch_size": 400, "remove_columns": removed_columns}
            self.processor.tokenizer.padding_side = "right"
            ds_pool = [
                load_dataset(self.repo_id, split).sort('complexity').filter(lambda e: len(e['answers']) >= i) for i in range(1, mr_max_actions+1)
            ]
            logger.info(ds_pool)

            for i in range(len(ds_pool)):
                fn_kwargs = {"train": True, "use_frame": use_frame, "mr_max_actions": i + 1}

                ds_pool[i]['train'] = ds_pool[i]['train'].map(
                    self.pre_collate_mr_v2, fn_kwargs=fn_kwargs, **default_kwargs,
                ).with_format("torch")

                fn_kwargs['train'] = False
                ds_pool[i]['validation'] = ds_pool[i]['validation'].map(
                    self.pre_collate_mr_v2, fn_kwargs=fn_kwargs, **default_kwargs,
                ).with_format("torch")

                ds_pool[i]['test'] = ds_pool[i]['test'].map(
                    self.pre_collate_mr_v2, fn_kwargs=fn_kwargs, **default_kwargs,
                ).with_format("torch")

            logger.info(ds_pool)

            ds = DatasetDict({
                "train": concatenate_datasets([d['train'] for d in ds_pool]),
                "test": concatenate_datasets([d['test'] for d in ds_pool]),
                "validation": concatenate_datasets([d['validation'] for d in ds_pool])
            })

        else:
            removed_columns = ['video_id', 'duration', 'prompt_frame', 'prompt_timestamp', 'action']
            default_kwargs = {"batched": False, "num_proc": self.num_worker, "writer_batch_size": 400, "remove_columns": removed_columns}
            ds['train'] = ds['train'].map(
                self.pre_collate_mr_ts,
                fn_kwargs={"train": True, "use_frame": use_frame},
                **default_kwargs
            )
            ds['validation'] = ds['validation'].map(lambda e: {**e, "len_action": e['answer'][1] - e['answer'][0]})
            ds['validation'] = ds['validation'].sort('len_action').select(range(64))
            default_kwargs['remove_columns'] = removed_columns + ['len_action']
            ds['validation'] = ds['validation'].map(
                self.pre_collate_mr_ts,
                fn_kwargs={"train": True, "use_frame": use_frame},
                **default_kwargs
            )
            default_kwargs['remove_columns'] = removed_columns
            ds['test'] = ds['test'].map(
                self.pre_collate_mr_ts,
                fn_kwargs={"train": True, "use_frame": use_frame},
                **default_kwargs
            )


        if split == 'action_ordering_v2':
            config_name = 'robust' if use_robust else 'normal'
        else:
            config_name = 'frame' if use_frame else 'timestamp'

        save_dir = f'{self.base_dir}/{self.processed_dir}/{split}/{config_name}/{str(self.num_frames)}_frames'
        logger.info(f'info: saving dataset to {save_dir}')
        ds.save_to_disk(save_dir)
        ds = load_from_disk(save_dir)
        set_verbosity_info()
        return ds


def to_str_timestamp(seconds):
    m, s = divmod(seconds, 60)
    return f"{round(m):02d}:{round(s):02d}"


class MomentRetrievalDataset:
    def __init__(
        self,
        base_dir: str = "datasets",
        processed_dir: str = "processed",
        processor = None,
        num_frames = 14,
        num_worker = 1
    ):
        self.repo_id = "jwnt4/a-temporal-upgrade"
        self.split = 'moment_retrieval'
        self.base_dir = base_dir
        self.processed_dir = processed_dir
        self.processor = processor
        self.num_frames = num_frames 
        self.num_worker = num_worker


    def read_video_decord(self, video_path):
        '''
        Decode the video with Decord decoder.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        vr = VideoReader(uri=video_path, ctx=cpu(0)) # you need to install from source to use gpu ctx
        step = len(vr) / self.num_frames
        indices = np.arange(0, len(vr) - step, len(vr) / (self.num_frames - 1)).astype(int)
        indices = np.append(indices, len(vr)-1)
        frames = vr.get_batch(indices).asnumpy()
        return (frames, np.average(vr.get_frame_timestamp(indices), axis=1))
    

    def download_videos(self, dataset):
        split = self.split
        logger = logging.getLogger("__main__")
        video_ids = set(dataset['train']['video_id'] + dataset['validation']['video_id'] + dataset['test']['video_id'])
        target_dir = f"{self.base_dir}/videos/{split}_v2"
        
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            present_videos = set()
        else:
            present_videos = set([d[:-4] for d in os.listdir(target_dir)])

        if len(video_ids & present_videos) == len(video_ids):
            logger.info("info: all videos are present, skip downloading")
            return
    
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir) # clean up improper downloads
        
        logger.info(f"info: downloading videos for {split}")
        download_path = hf_hub_download(
            self.repo_id, filename=f"videos/{split}.zip", repo_type='dataset', local_dir=self.base_dir
        )
        shutil.unpack_archive(download_path, f"{self.base_dir}/videos/", 'zip')
        os.remove(f"{self.base_dir}/videos/{split}.zip")
        shutil.move(target_dir, f"{self.base_dir}/videos/{split}")
        logger.info(f"info: success download at path {self.base_dir}/videos/{split}")


    def pre_collate_mr_ts(self, e, use_frame=False, train=False):
        # read video
        video_file = e['video_id'] + '.mp4'
        video_clip, v_ts = self.read_video_decord(f'{self.base_dir}/videos/moment_retrieval_v2/{video_file}')

        prompt = e['prompt_frame'] if use_frame else e['prompt_timestamp']
        prompt = prompt.replace("<num_frames>", str(self.num_frames))
        prompt = prompt.replace('<action>', e['action'].strip(' '))
        prompt = prompt.replace('Number of frames sampled in this video', 'Frames sampled')
        prompt = prompt.replace('Duration of the video', 'Video duration')
        prompt = prompt.replace('The action in question is', "Action in question")
        prompt = prompt.replace('The question', 'Question')
        prompt = prompt.replace('Here is the video context', 'Video context')
        prompt = prompt.replace('Here is the frame', 'The frame')

        if not use_frame:
            timestamps = [to_str_timestamp(v_ts[i]) for i in range(len(v_ts))]
            ts_infos = '\n'.join([f"Frame {f[0] + 1} at {f[1]}" for f in enumerate(timestamps)])
            prompt = prompt.replace('<frame_info>', ts_infos)
            prompt = prompt.replace('<duration>', str(round(e['duration'])) + " seconds")

        prompt = f"USER: <video>{prompt} ASSISTANT:"
        st = bisect_left(v_ts, e['answer'][0])
        ed = bisect_right(v_ts, e['answer'][1])
        ed = min(ed, self.num_frames - 1)
        
        if not use_frame: 
            st = to_str_timestamp(v_ts[st])
            ed = to_str_timestamp(v_ts[ed])
        else:
            st = str(st + 1)
            ed = str(ed + 1)

        if train:
            prompt += ', '.join([st, ed])

        batch = self.processor(
            text=prompt,
            videos=video_clip,
            truncation=True,
            max_length=4096,
            return_tensors='pt'
        )

        if not train:
            if not use_frame:
                batch['ts_info'] = timestamps
            batch['answer'] = [st, ed]

        return batch
    

    def prepare_dataset(self, use_frame=True):
        logger = logging.getLogger(__name__)
        set_verbosity_error()

        ds = load_dataset(self.repo_id, self.split)
        self.download_videos(ds)
        removed_columns = ['video_id', 'duration', 'prompt_frame', 'prompt_timestamp', 'action', 'answer']
        default_kwargs = {"batched": False, "num_proc": self.num_worker, "writer_batch_size": 400, "remove_columns": removed_columns}
        ds['train'] = ds['train'].shuffle(seed=18220053).select(range(640)).map(
            self.pre_collate_mr_ts,
            fn_kwargs={"train": True, "use_frame": use_frame},
            **default_kwargs
        ).with_format('torch')
        ds['validation'] = ds['validation'].shuffle(seed=18220053).select(range(48)).map(
            self.pre_collate_mr_ts,
            fn_kwargs={"train": False, "use_frame": use_frame},
            **default_kwargs
        ).with_format('torch')
        default_kwargs['remove_columns'] = removed_columns
        ds['test'] = ds['test'].map(
            self.pre_collate_mr_ts,
            fn_kwargs={"train": False, "use_frame": use_frame},
            **default_kwargs
        ).with_format('torch')

        config_name = 'timestamp' if not use_frame else 'frame'
        save_dir = f'{self.base_dir}/{self.processed_dir}/{self.split}/{config_name}/{str(self.num_frames)}_frames'
        logger.info(f'info: saving dataset to {save_dir}')
        ds.save_to_disk(save_dir)
        ds = load_from_disk(save_dir)
        set_verbosity_info()
        return ds