# This script is used in the end of make_histogram.py for plotting with multicores. 
from .Calculate_SF import  * 
from .utils import check_outputpath
from concurrent.futures import ProcessPoolExecutor
import functools 

def calculate_sf_parallel(plot_tuple:tuple, output_path, is_nominal=False, nominal_path=None, 
                          period='ADE', do_systs=False, systs_type=None):
    
    if not is_nominal and nominal_path is None:
        logging.error("No WP_cut_path is passed for systematics.")
        raise Exception("No WP_cut_path is passed for systematics.")

    if (not is_nominal) and not (nominal_path is None):
        if plot_tuple[0] == 'event_weight':
            nominal_key='none'+'_'+'event_weight'
        else:
            nominal_key = plot_tuple[0]

        
        WP_cut_path = nominal_path / "plots" / "ADE" / "WP_cuts_pkls" / nominal_key / "WP_cuts.pkl"
        logging.info(f"Doing plotting for systematic, using WP_cut from nominal.\n Path is {WP_cut_path}")
        WP_cut = joblib.load(WP_cut_path)

    hists_MC = plot_tuple[1]['MC']
    hists_Data = plot_tuple[1]['Data']
    output_path = check_outputpath(output_path)

    if plot_tuple[0] == 'event_weight':
        reweighting_var = 'none'
        weight_option = plot_tuple[0]
    else:
        reweighting_var = '_'.join(str.split(plot_tuple[0], '_')[:2])
        weight_option = '_'.join(str.split(plot_tuple[0], '_')[2:])
    
    HistMap_MC_unumpy = convert_histdict2unumpy(hists_MC)
    HistMap_Data_unumpy = convert_histdict2unumpy(hists_Data)

    #### Draw pt spectrum
    Plot_Pt_Spectrum(HistMap_MC_unumpy, HistMap_Data_unumpy, output_path, reweighting_var, weight_option)

    Extraction_Results = Extract(HistMap_MC_unumpy, HistMap_Data_unumpy)
    extraction_output_path = check_outputpath(output_path / period / "Extraction_Results") 
    joblib.dump(Extraction_Results, extraction_output_path /  f"{reweighting_var}_Extraction_Results.pkl")

    #### Draw fraction
    Plot_Fraction(Extraction_Results=Extraction_Results, output_path=output_path, period=period, 
                  reweighting_var=reweighting_var, reweighting_option=weight_option)
                  
    #### Draw ROC plot 
    Plot_ROC(Extraction_Results, output_path, period, reweighting_var, reweighting_option=weight_option)

    # Doing the extraction plots...
    for var in label_var:
        for i_pt, l_pt in enumerate(label_ptrange[:-1]):
            Extraction_var_pt =  Extraction_Results[var][l_pt]
            #### Draw Forward vs Central plots 
            Plot_ForwardCentral_MCvsData(i_pt=i_pt,pt = l_pt, var= var, output_path= output_path, 
                                period= period, reweighting_var = reweighting_var,
                                reweighting_option= weight_option,
                                Forward_MC= Extraction_var_pt['Forward_MC'], 
                                Central_MC= Extraction_var_pt['Central_MC'],
                                Forward_Data= Extraction_var_pt['Forward_Data'], 
                                Central_Data= Extraction_var_pt['Central_Data'],
                                if_norm=False, show_yields=False)

            Plot_ForwardCentral_MCvsData(i_pt=i_pt,pt = l_pt, var = var, output_path = output_path, 
                                period = period, reweighting_var = reweighting_var,
                                reweighting_option= weight_option,
                                Forward_MC= Normalize_unumpy(Extraction_var_pt['Forward_MC']), 
                                Central_MC= Normalize_unumpy(Extraction_var_pt['Central_MC']),
                                Forward_Data= Normalize_unumpy(Extraction_var_pt['Forward_Data']), 
                                Central_Data= Normalize_unumpy(Extraction_var_pt['Central_Data']),
                                if_norm=True, show_yields=False)
            
            Plot_Parton_ForwardvsCentral(i_pt=i_pt,pt = l_pt, var = var, output_path = output_path,
                                period = period, reweighting_var = reweighting_var,
                                reweighting_option = weight_option, 
                                p_Forward_Quark = Normalize_unumpy(Extraction_var_pt['Forward_Quark']),  
                                p_Central_Quark = Normalize_unumpy(Extraction_var_pt['Central_Quark']), 
                                p_Forward_Gluon = Normalize_unumpy(Extraction_var_pt['Forward_Gluon']), 
                                p_Central_Gluon = Normalize_unumpy(Extraction_var_pt['Central_Gluon'])
                                )

            #### Draw extraction plots 
            Plot_Extracted_unumpy(i_pt=i_pt,pt = l_pt, var= var, output_path= output_path, 
                                    period= period, reweighting_var = reweighting_var,
                                    reweighting_factor= weight_option,
                                    p_Quark=Extraction_var_pt['p_Quark'], p_Gluon=Extraction_var_pt['p_Gluon'],
                                    extract_p_Quark = Extraction_var_pt['extract_p_Quark_MC'], extract_p_Gluon = Extraction_var_pt['extract_p_Gluon_MC'],
                                    extract_p_Quark_Data = Extraction_var_pt['extract_p_Quark_Data'], extract_p_Gluon_Data = Extraction_var_pt['extract_p_Gluon_Data'],
                                    show_yields=False, 
                                    n_Forward_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_MC'])), 
                                    n_Central_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_MC'])), 
                                    variances_Forward_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Forward_MC'])**2),
                                    variances_Central_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Central_MC'])**2), 
                                    n_Forward_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_Data'])), 
                                    n_Central_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_Data'])),
                                    variances_Forward_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Forward_Data'])**2),
                                    variances_Central_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Central_Data'])**2)
                                    )
            Plot_Closure_unumpy(i_pt=i_pt,pt = l_pt, var= var, output_path= output_path, 
                                    period= period, reweighting_var = reweighting_var,
                                    reweighting_factor= weight_option,
                                    p_Quark=Extraction_var_pt['p_Quark'], p_Gluon=Extraction_var_pt['p_Gluon'],
                                    extract_p_Quark = Extraction_var_pt['extract_p_Quark_MC'], extract_p_Gluon = Extraction_var_pt['extract_p_Gluon_MC'],
                                    extract_p_Quark_Data = Extraction_var_pt['extract_p_Quark_Data'], extract_p_Gluon_Data = Extraction_var_pt['extract_p_Gluon_Data'],
                                    show_yields=False, 
                                    n_Forward_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_MC'])), 
                                    n_Central_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_MC'])), 
                                    variances_Forward_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Forward_MC'])**2),
                                    variances_Central_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Central_MC'])**2), 
                                    n_Forward_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_Data'])), 
                                    n_Central_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_Data'])),
                                    variances_Forward_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Forward_Data'])**2),
                                    variances_Central_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Central_Data'])**2)
                                    )
    WPs = [0.5, 0.6, 0.7, 0.8]
    SF_label_vars = ['jet_nTracks', 'GBDT_newScore']

    SFs = {}
    if is_nominal:
        WP_cut = dict.fromkeys(SF_label_vars)
        for var in SF_label_vars:
            WP_cut[var] = dict.fromkeys(WPs)
            for WP in WPs:
                WP_cut[var][WP] = dict.fromkeys(label_ptrange[:-1])

    for var in SF_label_vars:
        SFs[var] = {}
        #### Draw working points 
        for WP in WPs:
            SFs[var][WP] = {}
            quark_effs_at_pt = []
            gluon_rejs_at_pt = []
            quark_effs_data_at_pt = []
            gluon_rejs_data_at_pt = []
            for ii, l_pt in enumerate(label_ptrange[:-1]):
                extract_p_Quark_MC =  Extraction_Results[var][l_pt]['extract_p_Quark_MC']
                extract_p_Gluon_MC =  Extraction_Results[var][l_pt]['extract_p_Gluon_MC']
                extract_p_Quark_Data =  Extraction_Results[var][l_pt]['extract_p_Quark_Data']
                extract_p_Gluon_Data =  Extraction_Results[var][l_pt]['extract_p_Gluon_Data']

                if is_nominal:
                    extract_p_Quark_cum_sum = np.cumsum(unumpy.nominal_values(extract_p_Quark_MC))
                    cut = np.where(extract_p_Quark_cum_sum >= WP)[0][0]+1
                else:
                    cut = WP_cut[var][WP][l_pt]['idx']
                
                if do_systs and systs_type == "MC_nonclosure":
                    # for MC_nonclosure, compare MC truth vs MC extracted after reweighting
                    p_Quark = Extraction_Results[var][l_pt]['p_Quark']
                    p_Gluon = Extraction_Results[var][l_pt]['p_Gluon']
                    quark_effs_at_pt.append(np.sum(p_Quark[:cut])) 
                    gluon_rejs_at_pt.append(np.sum(p_Gluon[cut:]))
                    quark_effs_data_at_pt.append(np.sum(extract_p_Quark_MC[:cut]))
                    gluon_rejs_data_at_pt.append(np.sum(extract_p_Gluon_MC[cut:]))

                else: 
                    # for others, compare MC extracted vs Data extracted after reweighting 
                    quark_effs_at_pt.append(np.sum(extract_p_Quark_MC[:cut])) 
                    gluon_rejs_at_pt.append(np.sum(extract_p_Gluon_MC[cut:]))
                    quark_effs_data_at_pt.append(np.sum(extract_p_Quark_Data[:cut]))
                    gluon_rejs_data_at_pt.append(np.sum(extract_p_Gluon_Data[cut:]))

                if is_nominal:
                    WP_cut[var][WP][l_pt] = {
                        'idx' : cut,
                        'value' : HistBins[var][cut],
                    }

            SF_quark, SF_gluon = Plot_WP(WP = WP, var= var, output_path= output_path, 
                    period= period, reweighting_var = reweighting_var,
                    reweighting_factor= weight_option,
                    quark_effs= quark_effs_at_pt, gluon_rejs = gluon_rejs_at_pt,
                    quark_effs_data=quark_effs_data_at_pt, gluon_rejs_data = gluon_rejs_data_at_pt)
            SFs[var][WP]["Quark"] = SF_quark
            SFs[var][WP]["Gluon"] = SF_gluon

    #WriteSFtoPickle(Hist_SFs = SFs, output_path=output_path, period=period, 
    #                reweighting_var = reweighting_var, reweighting_factor= weight_option)

    if is_nominal:
        WriteWPcuttoPickle(WP_cuts= WP_cut, output_path=output_path, period=period, 
                            reweighting_var = reweighting_var, reweighting_factor= weight_option)
        return WP_cut

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script does the MC Closure test. ')
    parser.add_argument('--path-mc', help='The path to the merged MC histogram file(.pkl file).',\
        default='/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/nominal/MC_merged_hist.pkl')
    parser.add_argument('--path-data', help='The path to the merged Data histogram file(.pkl file).',\
        default='/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/nominal/Data_merged_hist.pkl')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E', "ADE"],\
        default='ADE')
    parser.add_argument('--output-path', help='Output path',\
        default='./test_calculate_sf_parallel')

    args = parser.parse_args()
    
    MC_merged_hist_path = Path(args.path_mc)
    Data_merged_hist_path = Path(args.path_data)
    period = Path(args.period)
    output_path = Path(args.output_path)

    MC_merged_hist = joblib.load(MC_merged_hist_path)
    Data_merged_hist = joblib.load(Data_merged_hist_path)

    plot_dict = {}
    for key in [*MC_merged_hist.keys()]:
        plot_dict[key]={
            "MC":MC_merged_hist[key],
            "Data":Data_merged_hist[key],
        }
    plot_tuple_list = [*plot_dict.items()]

    n_worker_plots = len(plot_tuple_list)
    calculate_sf_parallel_mod = functools.partial(calculate_sf_parallel, period = period, output_path=output_path)
    with ProcessPoolExecutor(max_workers=n_worker_plots) as executor:
        executor.map(calculate_sf_parallel_mod, plot_tuple_list)
    pass
