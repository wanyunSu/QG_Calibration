import joblib 
import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import hist
from hist import Hist 
from uncertainties import ufloat, unumpy
from .utils import * 


def predpkl2hist(input, reweight='event_weight', is_MC = True, output_path=None, if_save = False):
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
            output_name = f"digitized_{input_path.stem}.pkl"

    if is_MC:
        sample_pd = sample_pd[(sample_pd["jet_nTracks"] > 1) & (sample_pd["target"] != '-')] 
    else:
        sample_pd = sample_pd[(sample_pd["jet_nTracks"] > 1)]
    
    histogram_unumpy = digitize_pd(sample_pd, reweight=reweight, is_MC= is_MC)
    
    if if_save:
        joblib.dump(histogram_unumpy, output_path / output_name)

    return histogram_unumpy
