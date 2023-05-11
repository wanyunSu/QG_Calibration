import logging
import joblib 
import numpy as np
import pandas as pd 
from pathlib import Path
import re
from .utils import check_inputpath, check_outputpath

def pkl2predpkl(input, gbdt_path, training_vars=['jet_pt', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1'],
                output_path=None, if_save = False):

    if isinstance(input, pd.DataFrame):
        sample_pd = input
        
    elif isinstance(input, str) or isinstance(input, Path):
        input_path = input if isinstance(input, Path) else Path(input)

        input_path = check_inputpath(input_path)
        sample_pd = joblib.load(input_path)
        if not isinstance(sample_pd, pd.DataFrame):
            raise Exception(f"Check the input format! expect pd.DataFrame in {input_path}")
        
        if if_save:
            if output_path is None:
                output_path = input_path.parent
            output_path = check_outputpath(output_path)
            output_name = f"{input_path.stem}_pred.pkl"

    gbdt = joblib.load(gbdt_path)

    #sample_pd['GBDT_newScore'] = gbdt.predict_proba(sample_pd[training_vars])[:,1]
    sample_pd['GBDT_newScore'] = gbdt.predict(sample_pd[training_vars], raw_score = True)

    if if_save:
        joblib.dump(sample_pd, output_path / output_name)
    return sample_pd
