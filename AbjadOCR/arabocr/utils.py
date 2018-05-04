import numpy as np
import json

class AbjadUtils(object):

    class CollectionEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)

    @classmethod
    def DebugNumpyArray(cls, arr):
        print('')
        if isinstance(arr, np.ndarray):
            print('Content :', arr)
            print('Shape :', arr.shape)
            print('Size :', arr.size)
        else: 
            print('(Not ndarray | Content :', arr)
        print('')
            

