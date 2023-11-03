import pandas as pd

def batch_to_hit_df(batch):
    class MyDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __missing__(self, key):
            return 0
            
    l = len(batch.r)
    
    index_to_pid = MyDict({ **dict(zip(batch.track_edges[0].numpy(), batch.particle_id.numpy())), **dict(zip(batch.track_edges[1].numpy(), batch.particle_id.numpy())) })
    
    df = pd.DataFrame({ key: batch[key].numpy() for key in batch.keys if len(batch[key]) == l})
    df["particle_id"] = df.index.map(index_to_pid)
    
    return df
