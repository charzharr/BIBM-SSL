
import sys, os
import pandas as pd
import pickle
import PIL
import torch


def get_df(path, df_file='df.pd'):
    """ Returns the knee cartilage dataset as a df. """
    assert os.path.isdir(path)

    if os.path.isfile(os.path.join(path, df_file)):
        with open(os.path.join(path, df_file), 'rb') as f:
            df = pickle.load(f)
    else:
        raise ValueError(f"DF file not in: {os.path.join(path, df_file)}")

    # Fix paths
    old_path_s = df.iloc[0]['image'].split('/')
    idx = old_path_s.index('train')
    base_path = '/' + os.path.join(*old_path_s[:idx])

    def _replace(it):
        if isinstance(it, str):
            return it.replace(base_path, path)
        new_it = []
        for i in it:
            new_it.append(i.replace(base_path, path))
        return new_it

    df['image'] = df['image'].apply(_replace)
    df['mask'] = df['mask'].apply(_replace)

    return df

        
    


    