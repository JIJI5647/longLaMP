import json
from datasets import load_dataset
import os
from torch.utils.data import Dataset


class GeneralSeq2SeqDataset(Dataset):

    def __init__(self, task, use_profile = True, create_prompt = None, dataset_name = "abstract_generation_user", split = "train") -> None:
        super().__init__()
            
        token = os.getenv("HF_TOKEN")
        ds = load_dataset("LongLaMP/LongLaMP", 
                      dataset_name, 
                      token=token)
        self.data = ds[split]
        self.use_profile = use_profile
        self.task = task
        assert not (use_profile ^ (create_prompt != None)), "You should provide a prompt maker function when you use profile"
        self.create_prompt = create_prompt

    def __getitem__(self, index):
        if self.use_profile:
            return {
                "source" : self.create_prompt(self.data[index]['input'], self.data[index]['profile'], self.task),
                "target" : self.data[index]['output']
            }
        else:
            return {
                "source" : self.data[index]['input'],
                "target" : self.data[index]['output']
            }
    
    def __len__(self):
        return len(self.data)


