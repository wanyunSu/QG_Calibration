from core.utils import *
from core.root2pkl import * 
from core.pkl2predpkl import *
from core.predpkl2hist import * 
from core.calculate_sf_parallel import * 

from concurrent.futures import ProcessPoolExecutor
import functools 

n_workers = 8

def merge_hists(hists_list:list):
    merged_hists = hists_list[0]
    for hist_to_be_merged in hists_list[1:]:
        for k, v in merged_hists.items():
            v += hist_to_be_merged[k]
    return merged_hists

def final_reweighting(pkl:Path, reweight_factor, do_systs=False, output_path = None):
    logging.info(f"Doing final reweighting on {pkl.stem}...")
    logging.info(f"Doing systematics? {do_systs}")
    sample_pd = joblib.load(pkl)
    is_MC = False if pkl.stem.startswith('data') else True

    if output_path is None:
        output_path = pkl.parent

    sub_samples_pd = np.array_split(sample_pd, n_workers) # split the sample to subsamples for multi-processing 
    del sample_pd

    attach_reweight_factor_mod = functools.partial(attach_reweight_factor, reweight_factor = reweight_factor)
    logging.debug(f"Doing attach_reweight_factor multiprocessing... {pkl.stem}")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        sub_samples_pd = list(executor.map(attach_reweight_factor_mod, sub_samples_pd)) # len: n_workers 

    joblib.dump(pd.concat(sub_samples_pd), pkl) # overwrite the original file 

    reweighted_hists_dict = {}
    if not do_systs: # Note: This includes MC-Non closure and Gluon reweighting uncertainties 
        all_weight_options = ['event_weight'] + \
                     [f'{reweight_var}_{parton}_reweighting_weights' 
                     for reweight_var in reweighting_vars for parton in ['quark', 'gluon']]
        logging.info(f"All the weighting options are: {all_weight_options}")
    else: ## only do the quark reweighting 
        all_weight_options = ['event_weight']+ [f'{reweight_var}_{parton}_reweighting_weights' 
                             for reweight_var in reweighting_vars for parton in ['quark']]
                             
    logging.debug(f"Doing predpkl2hist multiprocessing... {pkl.stem}")
    for weight in all_weight_options:
        logging.debug(f"\tDoing {weight}... {pkl.stem}")
        predpkl2hist_mod = functools.partial(predpkl2hist, reweight = weight, is_MC = is_MC)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            hists_list = list(executor.map(predpkl2hist_mod, sub_samples_pd))

        reweighted_hists_dict[weight] = merge_hists(hists_list)
    logging.debug(f"Done final_reweighting for {pkl.stem}")

    joblib.dump(reweighted_hists_dict, output_path / f"{pkl.stem}_hists.pkl" )
    logging.debug(f"is_MC? {is_MC}, {pkl.stem}")
    if is_MC:
        return_key = 'MC'
    else:
        return_key = 'Data'
    assert not (reweighted_hists_dict is None)
    return {return_key: reweighted_hists_dict}

def _merge_period(hist_list):
    if len(hist_list) == 0:
        logging.error(f"Null hist_list")
        raise Exception(f"Null hist_list passed")
    merged_hist = hist_list[0]
    for to_be_merged_hist in hist_list[1:]:
        for key_reweight, value_reweight in merged_hist.items():
            for key_hist, value_hist in to_be_merged_hist[key_reweight].items():
                merged_hist[key_reweight][key_hist] += value_hist

    return merged_hist

def merge_period(reweighted_hists_dicts:list):
    # reweighted_hists_dicts is a list of dicts.
    # e.g. 6 dicts, 3 'MC' and 3 'Data'
    MC_hist_list = []
    Data_hist_list = []
    for reweighted_hists_dict in reweighted_hists_dicts:
        if [*reweighted_hists_dict][0] == 'MC':
            MC_hist_list.append(reweighted_hists_dict['MC'])
        elif [*reweighted_hists_dict][0] == 'Data':
            Data_hist_list.append(reweighted_hists_dict['Data'])
    
    return _merge_period(MC_hist_list), _merge_period(Data_hist_list)

def make_histogram_parallel(output_path, do_systs=False, if_write_log=False):
    logging_setup(verbosity=3, if_write_log=if_write_log, output_path=output_path, filename='final_plotting')

    predpkl_pattern = "*_pred.pkl"
    predpkl_files = sorted(output_path.rglob(predpkl_pattern))

    logging.info("Attach new weighting to pd.DataFrame and reweighting...")
    reweight_factor = joblib.load(output_path / 'reweight_factor.pkl')
    # The following three blocks failed with 50GB memory, we can do sequencially here but within final reweighting multiprocessing 
    # final_reweighting_mod = functools.partial(final_reweighting, reweight_factor = reweight_factor, output_path=output_path)
    # with ProcessPoolExecutor(max_workers=6) as executor:
    #     reweighted_hists_dicts = list(executor.map(final_reweighting_mod, predpkl_files)) # a list of 6 dicts
    reweighted_hists_dicts = []
    for predpkl_file in predpkl_files:
        reweighted_hists_dicts.append(final_reweighting(predpkl_file, reweight_factor, do_systs=do_systs))

    joblib.dump(reweighted_hists_dicts, output_path / 'reweighted_hists_dicts.pkl')
    # reweighted_hists_dicts = joblib.load(output_path / 'reweighted_hists_dicts.pkl')

    logging.info("Merging the histograms for MC and Data...")
    MC_merged_hist, Data_merged_hist = merge_period(reweighted_hists_dicts)
    joblib.dump(MC_merged_hist, output_path / 'MC_merged_hist.pkl')
    joblib.dump(Data_merged_hist, output_path / 'Data_merged_hist.pkl')

    logging.info("Plotting...")
    plot_dict = {}
    for key in [*MC_merged_hist.keys()]:
        plot_dict[key]={
            "MC":MC_merged_hist[key],
            "Data":Data_merged_hist[key],
        }
    plot_tuple_list = [*plot_dict.items()]
    n_worker_plots = len(plot_tuple_list)
    calculate_sf_parallel_mod = functools.partial(calculate_sf_parallel, output_path=output_path / 'plots')
    with ProcessPoolExecutor(max_workers=n_worker_plots) as executor:
        executor.map(calculate_sf_parallel_mod, plot_tuple_list)
    logging.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', help='the output folder path', type=str)
    parser.add_argument('--do-systs', help='whether do nominal study or systematics', action="store_true")
    parser.add_argument('--write-log', help='whether to write the log to output path', action="store_true")

    args = parser.parse_args()

    output_path = Path(args.output_path)
    do_systs = args.do_systs
    if_write_log = args.write_log

    make_histogram_parallel(output_path=output_path, do_systs=do_systs, 
                            if_write_log=if_write_log)