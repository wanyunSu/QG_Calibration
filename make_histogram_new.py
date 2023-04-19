from core.utils import *
from core.root2pkl import * 
from core.pkl2predpkl import *
from core.predpkl2hist import * 
from core.calculate_sf_parallel import * 
from make_plot_new import make_plots

from concurrent.futures import ProcessPoolExecutor
import functools 

pythia_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_alter/sherpa'
#pythia_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/herwigangle'
#pythia_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_Dec11/pythia/'
data_path = '/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_New/data/'
pythia_path = Path(pythia_path)
data_path = Path(data_path)

# default gbdt path 
# gbdt_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/LightGBM/optuna_tuning/small_dataset/lightgbm_gbdt.pkl'
gbdt_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/LightGBM/4vars/full_dataset/lightgbm_gbdt.pkl'
gbdt_path = Path(gbdt_path)

n_workers = 8 


def root2hist(input_path, output_path = None, is_MC = True, MC_identifier = 'pythia', 
              do_systs=False, systs_type=None, systs_subtype=None, verbosity = 2, write_log = False):
    input_path = check_inputpath(input_path)

    if is_MC:
        period_list = ["A", "D", "E"]
        minitrees_periods = [f"{MC_identifier}{period}" for period in period_list]
        glob_pattern = "*JZ*_minitrees.root/*.root"
    else:
        period_list = ["1516", "17", "18"]
        minitrees_periods = [f"data{period}" for period in period_list]
        glob_pattern = "*data*_13TeV.period?.physics_Main_minitrees.root/*.root"
        
    if output_path is None:
            output_path = input_path
    else:
        output_path = check_outputpath(output_path)

    return_dicts = {}
    for minitrees_period in minitrees_periods:
        logging.info(f"Processing {minitrees_period} minitrees...")
        minitreess = input_path / minitrees_period 
        root_files = sorted(minitreess.rglob(glob_pattern))

        if len(root_files)==0:
            raise Exception("No file found. Check the match pattern!")

        root2pkl_mod = functools.partial(root2pkl, is_MC=is_MC,output_path=output_path, do_systs=do_systs, 
                                         systs_type=systs_type, systs_subtype=systs_subtype,
                                         MC_identifier=MC_identifier, if_save=False)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            pkls = list(executor.map(root2pkl_mod, root_files))
        
        pkls = [x for x in pkls if x is not None] # Important to filter the None! 
        
        pkl2predpkl_mod = functools.partial(pkl2predpkl, output_path = None, gbdt_path=gbdt_path, if_save=False)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            predpkls = list(executor.map(pkl2predpkl_mod, pkls))  
        del pkls # release some memory 
         
        predpkl2hist_mod = functools.partial(predpkl2hist, reweight='event_weight',is_MC = is_MC, output_path = None, if_save = False)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            hists = list(executor.map(predpkl2hist_mod, predpkls)) 
        
        if minitrees_period == f'{MC_identifier}E': # periodE is too big, thus divided by 2 to fit into a Cori Haswell Node. 
            merged_pkls_period = pd.concat(predpkls[:len(predpkls)//2])
            merged_pkls_period_path = output_path / (f"{minitrees_period}_part1_pred.pkl")
            joblib.dump(merged_pkls_period, merged_pkls_period_path)

            del merged_pkls_period, merged_pkls_period_path
            merged_pkls_period = pd.concat(predpkls[len(predpkls)//2:])
            merged_pkls_period_path = output_path / (f"{minitrees_period}_part2_pred.pkl")
            joblib.dump(merged_pkls_period, merged_pkls_period_path)

        else:
            merged_pkls_period = pd.concat(predpkls)
            merged_pkls_period_path = output_path / (f"{minitrees_period}_pred.pkl") # forexample, pythiaA_pred.pkl
            joblib.dump(merged_pkls_period, merged_pkls_period_path)
        del predpkls # release some memory 

        return_dicts[minitrees_period]=hists

    return return_dicts

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

def make_histogram_parallel(input_mc_path, input_data_path, output_path:Path, 
                            do_systs=False, systs_type=None, systs_subtype=None,
                            if_write_log=False, if_do_plotting=False):
    if do_systs and systs_type is None:
        raise Exception(f"You ask to do systematics but its type is not given! systs_type={systs_type}")
    elif (not (systs_type is None)) and systs_subtype is None:
        raise Exception(f"You ask to do systematics {systs_type} but its subtype is not given! systs_subtype={systs_subtype}")
    if (not (do_systs is None)) and (not (systs_subtype is None)):
        nominal_path = output_path / "nominal"
        output_path = output_path / systs_type / systs_subtype
    else:
        # For nominal 
        output_path = output_path / "nominal"
        nominal_path = output_path

    logging_setup(verbosity=3, if_write_log=if_write_log, output_path=output_path)

    logging.info(f"do_systs = {do_systs}, systs_type = {systs_type}, systs_subtype = {systs_subtype}")

    MC_identifier = "pythia"
    if do_systs:
        if systs_type in ['parton_shower', 'hadronization', 'matrix_element']: 
            # If doing such studies, the MC_identifiers are not longer pythia! 
            MC_identifier =  systs_subtype

    logging.info("Doing root2hist for MC...")
    MC_hists = root2hist(input_path=input_mc_path, output_path=output_path, is_MC=True, MC_identifier = MC_identifier,
                         do_systs=do_systs, systs_type=systs_type, systs_subtype=systs_subtype)

    joblib.dump(MC_hists, output_path / 'MC_hists.pkl')
    ## MC_hists = joblib.load(output_path / 'MC_hists.pkl')
    logging.info("Calculate reweighting factor from MC...")
    reweight_factor = get_reweight_factor_hist(MC_hists, if_need_merge=True)
    joblib.dump(reweight_factor, output_path / 'reweight_factor.pkl')

    logging.info("Doing root2hist for Data...")
    _ =  root2hist(input_path=input_data_path, output_path=output_path, is_MC=False)
    predpkl_pattern = "*_pred.pkl"
    predpkl_files = sorted(output_path.rglob(predpkl_pattern))

    reweight_factor = joblib.load(output_path / 'reweight_factor.pkl')
    logging.info("Attach new weighting to pd.DataFrame and reweighting...")
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
    # MC_merged_hist = joblib.load(output_path / 'MC_merged_hist.pkl')
    # Data_merged_hist = joblib.load(output_path / 'Data_merged_hist.pkl')
    # Do final plotting here with multi-processing. 
    if if_do_plotting:
        make_plots(MC_merged_hist=MC_merged_hist, Data_merged_hist=Data_merged_hist,
                   output_path=output_path, nominal_path=nominal_path, 
                   do_systs=do_systs, systs_type=systs_type, systs_subtype=systs_subtype, 
                   if_write_log=if_write_log)

    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mc-path', help='the input folder path for MC', type=str, default=pythia_path)
    parser.add_argument('--input-data-path', help='the input folder path Data', type=str, default=data_path)
    parser.add_argument('--output-path', help='the output folder path', type=str)
    parser.add_argument('--gbdt-path', help='the lightGBM model path', type=str, default=gbdt_path)
    parser.add_argument('--write-log', help='whether to write the log to output path', action="store_true")
    parser.add_argument('--do-systs', help='whether do nominal study or systematics', action="store_true")
    parser.add_argument('--systs-type', help='choose the systematic uncertainty type', default=None, 
                        choices=['trk_eff', 'JESJER', 'pdf_weight', 'scale_variation', 'parton_shower', 
                                 'hadronization', 'matrix_element'])
    parser.add_argument('--systs-subtype', help='choose the systematic uncertainty subtype', default=None, 
                        choices=all_systs_subtypes)
    parser.add_argument('--do-plotting', help='whether to do plotting', action="store_true")


    args = parser.parse_args()

    output_path = Path(args.output_path)
    input_mc_path = Path(args.input_mc_path)
    input_data_path = Path(args.input_data_path)

    do_systs = args.do_systs
    systs_type = args.systs_type
    systs_subtype = args.systs_subtype
    if_write_log = args.write_log
    if_do_plotting = args.do_plotting

    make_histogram_parallel(input_mc_path=input_mc_path, input_data_path=input_data_path,
                            output_path=output_path, do_systs=do_systs, systs_type=systs_type, 
                            systs_subtype=systs_subtype, if_write_log=if_write_log,
                            if_do_plotting=if_do_plotting)
