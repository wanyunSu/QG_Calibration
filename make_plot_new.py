from core.utils import *
from core.root2pkl import * 
from core.pkl2predpkl import *
from core.predpkl2hist import * 
from core.calculate_sf_parallel import * 

from concurrent.futures import ProcessPoolExecutor
import functools 

#input_path = '/global/cfs/projectdirs/atlas/wys/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new_hrzhao/matrix_element/powhegpythia'
input_path = '/global/cfs/projectdirs/atlas/wys/QG_Calibration/note_plots_qg/nominal'
nominal_path = '/global/cfs/projectdirs/atlas/wys/QG_Calibration/note_plots_qg/nominal'
#input_path = '/global/cfs/projectdirs/atlas/wys/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new_hrzhao/nominal'
#nominal_path = '/global/cfs/projectdirs/atlas/wys/QG_Calibration/note_plots/nominal'
#nominal_path = '/global/cfs/projectdirs/atlas/wys/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new_hrzhao/nominal/'

def make_plots(MC_merged_hist, Data_merged_hist, output_path, nominal_path=None, 
               do_systs=None, systs_type=None, systs_subtype=None, if_write_log=True):
    
    # Do final plotting here with multi-processing. 
    logging_setup(verbosity=3, if_write_log=if_write_log, output_path=output_path, filename="plotting")

    logging.info("Plotting...")
    global nominal_keys
    if do_systs:
        logging.info(f"Doing plotting for syst={systs_type}, systs_subtype={systs_subtype}")
        if nominal_path is None:
            raise Exception("You are doing systematics but the nominal path is not given! ")
        
        if (systs_type != "MC_nonclosure") and (systs_type != "gluon_reweight"): 
            plot_dict = {}
            nominal_keys = nominal_keys + ["event_weight"] # Here the event is used as tests. 
            for key in nominal_keys:
                plot_dict[key]={
                    "MC":MC_merged_hist[key],
                    "Data":Data_merged_hist[key],
                }
            plot_tuple_list = [*plot_dict.items()]

            calculate_sf_parallel_syst = functools.partial(calculate_sf_parallel,
                                            is_nominal = False, 
                                            nominal_path = nominal_path,
                                            output_path=output_path / 'plots')

            n_worker_plots = len(nominal_keys)
            # breakpoint()
            # calculate_sf_parallel(plot_tuple=plot_tuple_list[-1], nominal_path=nominal_path,
            #                       output_path=output_path / 'plots')
            with ProcessPoolExecutor(max_workers=n_worker_plots) as executor:
                executor.map(calculate_sf_parallel_syst, plot_tuple_list)

        # make a list of plot_tuples, and pass it to calculate_sf_parallel
        else:
            plot_dict = {}
            if systs_type=="MC_nonclosure":
                # need to use the cut from nominal keys to define, so three versions needed 
                for key in nominal_keys:
                    # plot_dict[key]={
                    #     "MC":MC_merged_hist["event_weight"],
                    #     "Data":Data_merged_hist["event_weight"],
                    # }
                    plot_dict[key]={
                        "MC":MC_merged_hist[key],
                        "Data":Data_merged_hist[key],
                    }

                calculate_sf_parallel_syst = functools.partial(calculate_sf_parallel,
                                        is_nominal = False, 
                                        nominal_path = nominal_path,
                                        output_path = output_path / 'plots',
                                        do_systs = True, systs_type = 'MC_nonclosure')

            elif systs_type=="gluon_reweight":
                for key in nominal_keys:
                    gluon_weigt_key = key.replace("quark", "gluon")
                    plot_dict[key] = {
                        "MC":MC_merged_hist[gluon_weigt_key],
                        "Data":Data_merged_hist[gluon_weigt_key],
                    }
                calculate_sf_parallel_syst = functools.partial(calculate_sf_parallel,
                                        is_nominal = False, 
                                        nominal_path = nominal_path,
                                        output_path=output_path / 'plots',
                                        do_systs = True, systs_type = 'gluon_reweight')

            plot_tuple_list = [*plot_dict.items()]

            n_worker_plots = len(nominal_keys)
            with ProcessPoolExecutor(max_workers=n_worker_plots) as executor:
                executor.map(calculate_sf_parallel_syst, plot_tuple_list)
            
                
    else: # For the nominal keys, just 3 vars and quark reweighting   
        logging.info(f"Doing plotting for nominal!")
        
        # only restricted to the nominal keys 
        plot_dict = {}
        nominal_keys = nominal_keys + ["event_weight"]
        for key in nominal_keys:
            plot_dict[key]={
                "MC":MC_merged_hist[key],
                "Data":Data_merged_hist[key],
            }
        plot_tuple_list = [*plot_dict.items()]

        calculate_sf_parallel_nominal = functools.partial(calculate_sf_parallel,
                                        is_nominal = True, 
                                        output_path=output_path / 'plots')

        n_worker_plots = len(nominal_keys)
        with ProcessPoolExecutor(max_workers=n_worker_plots) as executor:
            executor.map(calculate_sf_parallel_nominal, plot_tuple_list)
    logging.info(f"all keys are: {[a[0] for a in plot_tuple_list]}")

    logging.info("Done Plotting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='the input merged hists path for MC and Data', type=str, default=input_path)
    parser.add_argument('--nominal-path', help='the nominal folder path', type=str, default=nominal_path)
    parser.add_argument('--output-path', help='the output folder path', type=str, default=None)
    parser.add_argument('--write-log', help='whether to write the log to output path', action="store_true")
    parser.add_argument('--do-systs', help='whether do nominal study or systematics', action="store_true")
    parser.add_argument('--systs-type', help='choose the systematic uncertainty type', default=None, 
                        choices=['MC_nonclosure', 'gluon_reweight', 'trk_eff', 'JESJER', 'pdf_weight', 
                                 'scale_variation', 'parton_shower', 'hadronization', 'matrix_element'])
    parser.add_argument('--systs-subtype', help='choose the systematic uncertainty subtype', default=None, 
                        choices=all_systs_subtypes)

    args = parser.parse_args()


    input_path = Path(args.input_path) # MC and Data merged hists should be in the same path 
    nominal_path = Path(args.nominal_path)

    do_systs = args.do_systs
    systs_type = args.systs_type
    systs_subtype = args.systs_subtype
    if_write_log = args.write_log

    if args.output_path is None:
        output_path = Path(args.input_path)
    else:
        output_path = Path(args.output_path)

    if not do_systs: 
        output_path = output_path / "nominal"
        
    else:
        if args.systs_subtype is None:
            output_path = output_path / systs_type 
        else:
            output_path = output_path / systs_type / systs_subtype


#    MC_merged_hist = joblib.load(input_path / "pythiaA_pred_hists.pkl")
#    Data_merged_hist = joblib.load(input_path / "data1516_pred_hists.pkl")
    MC_merged_hist = joblib.load(input_path / "MC_merged_hist.pkl")
    Data_merged_hist = joblib.load(input_path / "Data_merged_hist.pkl")
    
    make_plots(MC_merged_hist=MC_merged_hist, Data_merged_hist=Data_merged_hist,
               output_path=output_path, nominal_path=nominal_path, 
               do_systs=do_systs, systs_type=systs_type, systs_subtype=systs_subtype, 
               if_write_log=if_write_log)
