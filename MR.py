import scipy.stats
import pandas as pd
import math
import os
import xml.etree.ElementTree as ET
import statsmodels.api as sm


class MR:
    def __init__(self):
        self.estimation_data = []
        self.estimation_snps = []

    # Calculate Odds Ratios from sample
    # Params:
    #     datasource: directory that contains the generated models and samples
    #     testcase: which "data_<testcase>.csv" and "model_<testcase>.xml" file we are working with
    #     trait_node_id_range: exposure and outcome nodes of the sample
    #     numOfGenes: gene nodes of the sample

    def calc_or(self, datasource, trait_node_id_range, gene_node_id_range):
        #Read samples csv file
        df = pd.read_csv(f"{datasource}/data.csv")

        #Create output directory
        if not os.path.exists(f"{datasource}/MR_results"):
            os.makedirs(f"{datasource}/MR_results")

        # We calculate the ratios for every non-gene node, therefore exposure and outcome nodes
        for trait_node_id in trait_node_id_range:
            #Create file
            with open(f"{datasource}/MR_results/node_{trait_node_id}_stat.csv", "w") as output_handle:

                #Writing header
                output_handle.write("node_id,OR0,SE_OR0,CONF_INT_OR0_START,CONF_INT_OR0_END,OR1,SE_OR1,CONF_INT_OR1_START,CONF_INT_OR1_END,OR2,SE_OR2,CONF_INT_OR2_START,CONF_INT_OR2_END,OR_allele_counting,SE_OR_allele_counting,CONF_INT_OR_allele_counting_START,CONF_INT_OR_allele_counting_END,logistic_beta,logistic_SE\n")

                #Going through every gene node
                for predictor_node_index in gene_node_id_range:
                    # --- Logistic regression's beta ---
                    # log(P(Y=1)/P(Y=0)) = beta * X + epsilon
                    x = df['node_'+str(predictor_node_index)].values.reshape(-1,1) # reshape
                    y = df['node_'+str(trait_node_id)].values

                    log_reg = sm.Logit(y, x).fit(disp=False)

                    # --- Simple Odds calculation ---
                    # (P(Y=1|X=1)/P(Y=0|X=1)) / (P(Y=1|X=0)/P(Y=0|X=0))
                    # Y = 1 | X = 0, Y = 0 | X = 0 --> ezekből Odds()
                    Y0X0 = len(df[(df['node_'+str(trait_node_id)]==0) & (df['node_'+str(predictor_node_index)]==0)])
                    Y1X0 = len(df[(df['node_'+str(trait_node_id)]==1) & (df['node_'+str(predictor_node_index)]==0)])
                    if(Y0X0 == 0):
                        Y0X0 = 0.5 # np.finfo(np.double).tiny
                    if (Y1X0 == 0):
                        Y1X0 = 0.5

                    odds0 = Y1X0/Y0X0

                    # Y = 1 | X = 1, Y = 0 | X = 1
                    Y0X1 = len(df[(df['node_'+str(trait_node_id)]==0) & (df['node_'+str(predictor_node_index)]==1)])
                    Y1X1 = len(df[(df['node_'+str(trait_node_id)]==1) & (df['node_'+str(predictor_node_index)]==1)])
                    if (Y0X1 == 0):
                        Y0X1 = 0.5
                    if (Y1X1 == 0):
                        Y1X1 = 0.5

                    odds1 = Y1X1/Y0X1

                    # Y = 1 | X = 2, Y = 0 | X = 2
                    Y0X2 = len(df[(df['node_'+str(trait_node_id)]==0) & (df['node_'+str(predictor_node_index)]==2)])
                    Y1X2 = len(df[(df['node_'+str(trait_node_id)]==1) & (df['node_'+str(predictor_node_index)]==2)])
                    if (Y0X2 == 0):
                        Y0X2 = 0.5
                    if (Y1X2 == 0):
                        Y1X2 = 0.5

                    odds2 = Y1X2/Y0X2

                    # odds0, odds1, odds2
                    OR0 = odds1 / odds0
                    OR1 = odds2 / odds0
                    OR2 = odds2 / odds1

                    # standard_error
                    standard_error_OR0 = math.sqrt(1 / Y0X0 + 1 / Y1X0 + 1 / Y0X1 + 1 / Y1X1)
                    standard_error_OR1 = math.sqrt(1 / Y0X0 + 1 / Y0X2 + 1 / Y1X0 + 1 / Y1X2)
                    standard_error_OR2 = math.sqrt(1 / Y0X1 + 1 / Y0X2 + 1 / Y1X1 + 1 / Y1X2)

                    """Egy másik eshetőség, hogy a homozigóta vad és a homozigóta mutáns is duplán számít, mert azokban 2db ugyanolyan allél van  (allele counting method)
                    (2CC+TC) / (2TT+TC)
                    Tehát pl: (C - hatás és mutáns allél)
                    0: TT  1v0
                    1: TC  2v0
                    2: CC  2v1 """
                    OR_allele_counting = ((2*Y1X2 + Y1X1) /  (2*Y0X2 + Y0X1)) / ((2*Y1X0 + Y1X1) /  (2*Y0X0 + Y0X1))
                    #  Y=0 | 2*TT+TC , Y=1 | 2*TT+TC, Y=0 | 2*CC+TC, Y=1 | 2*CC+TC

                    standard_error_OR_allele_counting= 1 / (Y0X0*2 + Y0X1) + 1 / (Y1X0*2 + Y1X1) +  1 / (Y0X0*2 + Y0X1) + 1 / (Y1X0*2 + Y1X1)

                    conf_int_OR0 = (OR0 - 1.96*standard_error_OR0, OR0 + 1.96*standard_error_OR0)
                    conf_int_OR1 = (OR1 - 1.96*standard_error_OR1, OR1 + 1.96*standard_error_OR1)
                    conf_int_OR2 = (OR2 - 1.96*standard_error_OR2, OR2 + 1.96*standard_error_OR2)
                    conf_int_OR_allele_counting = (OR_allele_counting - 1.96*standard_error_OR_allele_counting, OR_allele_counting + 1.96*standard_error_OR_allele_counting)

                    # node_id, OR0, SE_OR0, CONF_INT_OR0_START, CONF_INT_OR0_END, OR1, SE_OR1, CONF_INT_OR1_START, CONF_INT_OR1_END, OR2, SE_OR2, CONF_INT_OR2_START, CONF_INT_OR2_END, OR_allele_counting, SE_OR_allele_counting, CONF_INT_OR_allele_counting_START, CONF_INT_OR_allele_counting_END, logistic_beta
                    output_handle.write(f"node_{predictor_node_index},{OR0},{standard_error_OR0},{conf_int_OR0[0]},{conf_int_OR0[1]},{OR1},{standard_error_OR1},{conf_int_OR1[0]},{conf_int_OR1[1]},{OR2},{standard_error_OR2},{conf_int_OR2[0]},{conf_int_OR2[1]},{OR_allele_counting},{standard_error_OR_allele_counting},{conf_int_OR_allele_counting[0]},{conf_int_OR_allele_counting[1]},{log_reg.params[0]},{log_reg.bse[0]}\n")

            
    def calc_beta(self, gene_node_id, exposure_id, outcome_id, save_dir ,stat_data_source="data/example_1_sample_100/0", modelfile="example_1_sample_100/model.xml"):
        df_exp = pd.read_csv(stat_data_source + "/node_"+str(exposure_id)+"_stat.csv")
        df_out = pd.read_csv(stat_data_source + "/node_"+str(outcome_id)+"_stat.csv")

        xml_root = ET.parse(modelfile).getroot()
        # .tag, .attrib, .text
        edge_attrib_array = []
        for tag in xml_root[0]:
            if('edge' in tag.tag):
                edge_attrib_array.append(tag.attrib)

        gene_exp_edge_there = {'source': 'node_'+str(gene_node_id), 'target': 'node_'+str(exposure_id)} in edge_attrib_array
        exp_out_edge_there = {'source': 'node_'+str(exposure_id), 'target': 'node_'+str(outcome_id)} in edge_attrib_array

        # betaY = df_exp['ln(OR_allele_counting)']['node_'+str(gene_node_id)]
        betaX = math.log(df_exp[df_exp['node_id'] == 'node_'+str(gene_node_id)]['OR_allele_counting'].iloc[0])
        # betaX = df_out["ln(OR_allele_counting)"]['node_'+str(gene_node_id)]
        betaY = math.log(df_out[df_out['node_id'] == 'node_'+str(gene_node_id)]['OR_allele_counting'].iloc[0])

        if betaX == 0:
            print("beta is zero:", betaX)
            print(gene_node_id, exposure_id, outcome_id)
            betaX = 1e-5


        # Y - risk factor
        # X - outcome
        # se_betaY = df_exp['SE']['node_'+str(gene_node_id)]
        se_betaY = df_out[df_out['node_id'] == 'node_'+str(gene_node_id)]['SE_OR_allele_counting'].iloc[0]
        beta = betaY/betaX
        se = se_betaY/math.exp(betaX)

        wald_p_val = scipy.stats.norm.sf(abs(beta / se)) * 2

        betaX_logreg = df_exp[df_exp['node_id'] == 'node_'+str(gene_node_id)]['logistic_beta'].iloc[0]
        betaY_logreg = df_out[df_out['node_id'] == 'node_'+str(gene_node_id)]['logistic_beta'].iloc[0]
        se_betaY_logreg = df_out[df_out['node_id'] == 'node_' + str(gene_node_id)]['logistic_SE'].iloc[0]
        beta_logreg = betaY_logreg/betaX_logreg
        se_logreg = se_betaY_logreg/math.exp(betaX_logreg)

        wald_p_val_logreg = scipy.stats.norm.sf(abs(beta_logreg / se_logreg)) * 2

        with open(f"{save_dir}/results_of_nodes_{gene_node_id}_{exposure_id}_{outcome_id}.txt", "w") as output_handle:
            output_handle.write("beta\tSE\twald_p_val\tbeta_logreg\tSE_logreg\twald_p_val_logreg\n")
            output_handle.write(f"{beta}\t{se}\t{wald_p_val}\t{beta_logreg}\t{se_logreg}\t{wald_p_val_logreg}\n")
            output_handle.write(f"Edge from gene to exposure in original model: {gene_exp_edge_there}\n")
            output_handle.write(f"Edge from exposure to outcome in original model: {exp_out_edge_there}\n")


    def calc_beta_by_alg(self, gene_node_id, exposure_id, outcome_id, alg, stat_data_source="data/example_1_sample_100/0", modelfile="example_1_sample_100/model.xml"):
        df_exp = pd.read_csv(stat_data_source + "/node_"+str(exposure_id)+"_stat.csv")
        df_out = pd.read_csv(stat_data_source + "/node_"+str(outcome_id)+"_stat.csv")
        # /node_0_stat.csv
        # exp_line = df_exp[df_exp["node_id"] == "node_"+str(gene_node_id)]
        # out_line = df_out[df_out["node_id"] == "node_"+str(gene_node_id)]

        xml_root = ET.parse(modelfile).getroot()
        # .tag, .attrib, .text
        edge_attrib_array = []
        for tag in xml_root[0]:
            if('edge' in tag.tag):
                edge_attrib_array.append(tag.attrib)

        gene_exp_edge_there = {'source': 'node_'+str(gene_node_id), 'target': 'node_'+str(exposure_id)} in edge_attrib_array
        exp_out_edge_there = {'source': 'node_'+str(exposure_id), 'target': 'node_'+str(outcome_id)} in edge_attrib_array

        # betaY = df_exp['ln(OR_allele_counting)']['node_'+str(gene_node_id)]
        betaX = math.log(df_exp[df_exp['node_id'] == 'node_'+str(gene_node_id)]['OR_allele_counting'].iloc[0])
        # betaX = df_out["ln(OR_allele_counting)"]['node_'+str(gene_node_id)]
        betaY = math.log(df_out[df_out['node_id'] == 'node_'+str(gene_node_id)]['OR_allele_counting'].iloc[0])

        # Y - risk factor
        # X - outcome
        # se_betaY = df_exp['SE']['node_'+str(gene_node_id)]
        se_betaY = df_out[df_out['node_id'] == 'node_'+str(gene_node_id)]['SE_OR_allele_counting'].iloc[0]

        beta = betaY/betaX
        se = se_betaY/math.exp(betaX)

        wald_p_val = scipy.stats.norm.sf(abs(beta / se)) * 2

        betaX_logreg = df_exp[df_exp['node_id'] == 'node_'+str(gene_node_id)]['logistic_beta'].iloc[0]
        betaY_logreg = df_out[df_out['node_id'] == 'node_'+str(gene_node_id)]['logistic_beta'].iloc[0]
        se_betaY_logreg = df_out[df_out['node_id'] == 'node_' + str(gene_node_id)]['logistic_SE'].iloc[0]
        beta_logreg = betaY_logreg/betaX_logreg
        se_logreg = se_betaY_logreg/math.exp(betaX_logreg)

        wald_p_val_logreg = scipy.stats.norm.sf(abs(beta_logreg / se_logreg)) * 2

        with open(f"{stat_data_source}/{alg}/results_of_nodes_{gene_node_id}_{exposure_id}_{outcome_id}.txt", "w") as output_handle:
            output_handle.write("beta\tSE\twald_p_val\tbeta_logreg\tSE_logreg\twald_p_val_logreg\n")
            output_handle.write(f"{beta}\t{se}\t{wald_p_val}\t{beta_logreg}\t{se_logreg}\t{wald_p_val_logreg}\n")
            output_handle.write(f"Edge from gene to exposure in original model: {gene_exp_edge_there}\n")
            output_handle.write(f"Edge from exposure to outcome in original model: {exp_out_edge_there}\n")

    def do_ivw_estimation(self, result_file_name):
        """
        Does IVW estimation on all the methods
        :return: tuple of floats: beta, se, wald_p_val of the estimate.
        """
        output_handle = open("data/"+result_file_name, "w")
        # output_handle.write(str(self.estimation_data))
        output_handle.write("beta standard_error wald_p_val\n")

        beta, se, wald_p_val = self.do_ivw_estimation_on_estimate_vector(
            self.estimation_data)

        output_handle.write(str(beta) + " " + str(se) + " " + str(wald_p_val))

        return beta, se, wald_p_val

    def do_ivw_estimation_on_estimate_vector(self, estimation_vec):
        """
        Estimates IVW on a specified estimate vector
        :param estimation_vec: list of (beta,se) tuples
        :return: tuple of floats: beta, se, wald_p_val of the estimate.
        """
        if len(estimation_vec) == 0:
            raise RuntimeError('No estimates supplied to do estimation')

        beta_top = 0.0
        beta_bottom = 0.0

        ivw_intermediate_top = []
        ivw_intermediate_bottom = []

        i = 0

        for smr_result in estimation_vec:

            # make sure the standard error cannot be zero.
            """if smr_result[1] == 0:
                smr_result[1] = np.nextafter(0, 1)"""

            ivw_intermediate_top.append(smr_result[0] * (smr_result[1] ** -2))
            ivw_intermediate_bottom.append((smr_result[1] ** -2))

            beta_top += ivw_intermediate_top[i]
            beta_bottom += ivw_intermediate_bottom[i]
            i += 1

        beta_ivw = beta_top / beta_bottom
        se_ivw = math.sqrt(1 / beta_bottom)

        p_value = scipy.stats.norm.sf(abs(beta_ivw / se_ivw)) * 2

        return beta_ivw, se_ivw, p_value
