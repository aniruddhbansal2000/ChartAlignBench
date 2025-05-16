import os
import pandas as pd
import numpy as np
import json
import csv

from collections import defaultdict
from itertools import permutations

import argparse

import io
import Levenshtein
from scipy.stats import kendalltau, spearmanr

# chart_type_list = ['bar_', 'bar_num_', '3D-Bar_', 'line_', 'line_num_', 'radar_', 'rose_', 'box_', 'multi-axes_']
# chart_dir_names_type_list = ['bar_chart', 'bar_chart_num', '3D-Bar', 'line_chart', 'line_chart_num', 'radar', 'rose', 'box', 'multi-axes']

chart_topic_list = ['Human Resources and Employee Management', 'Healthcare and Health', 'Arts and Culture', 'Business and Finance', 'Charity and Nonprofit Organizations', 'Tourism and Hospitality', 'Food and Beverage Industry', 'Technology and the Internet', 'Manufacturing and Production', 'Environment and Sustainability', 'Social Media and the Web', 'Retail and E-commerce', 'Science and Engineering', 'Law and Legal Affairs', 'Social Sciences and Humanities', 'Education and Academics', 'Energy and Utilities', 'Sports and Entertainment', 'Government and Public Policy', 'Agriculture and Food Production', 'Real Estate and Housing Market', 'Transportation and Logistics']

json_path = os.path.join(os.getcwd(), "_datasets", "ChartX_annotation.json")
with open(json_path, encoding='utf-8') as f:
    data = json.load(f)


# data values for a chart data: row_cnt X column_cnt (in csv)
def get_data_count(chart_name):

    for data_element in data:
        if data_element["imgname"] == chart_name:
            # found chart
            csv_str = data_element["csv"]
            csv_str = csv_str.replace(" \\t ", " \t ").replace(" \\n ", " \n ")
            df_chart_csv = pd.read_csv(io.StringIO(csv_str), sep="\t") 
            # data_count = (df_chart_csv.shape[0]*df_chart_csv.shape[1], df_chart_csv.shape[0], df_chart_csv.shape[1])
            return df_chart_csv.shape #data_count[0]
    return None


def get_chart_info_json(chart_name):

    for data_element in data:
        if data_element["imgname"] == chart_name:
            return data_element
    return None

# units (million, $ etc removed from COLUMN NAME)
    # units: sometimes represented in heading, sometimes in value hence UNCERTAIN
def remove_units_in_brackets(val):

    print(f"val before unit removal: {val}")

    idx_start = val.find('(')
    if idx_start == int(-1):
        # bracket doesn't exist
        return val    

    if val[idx_start-1] == ' ':
        idx_start -= 1          # space, remove it

    idx_end = val.find(')') + 1
    modified_val = val[:idx_start]
    if idx_end < len(val):
        modified_val += val[idx_end:]
    # modified_val = re.sub('([^>]+)', '', val)

    print(f"val after unit removal: {modified_val}")
    return modified_val

# If value in string form (contain %, or '$' currency), extract float value
def get_float_val(string_val):

    float_val = None
    try:    
        string_val_final = string_val.replace("%", "").replace("$", "")
        float_val = eval(string_val_final)
    except:
        float_val = float('inf')

    return float_val

# similarity for a JSON ground-truth vs generated (compare row-name, column-name, chart-1 val ie initial, chart-2 val ie modified)
def get_json_similarity(gt_json, gr_json):

    # ensure values are in int/flot form (some string form)
    try:
    # ROW COMPARISON
        # Levenshtein ratio -> floating number in range [0.0, 1.0] inversely proportional to corresponding distance
        base_score = 3 * Levenshtein.ratio(remove_units_in_brackets(gt_json["row name"]).lower(), remove_units_in_brackets(gr_json["row name"]).lower())
        # base_score = 3 * int(gt_json["row name"].lower() == gr_json["row name"].lower()) 

    # COLUMN COMPARISON
        # Levenshtein ratio & unit removal from brackets
        base_score += 3 * Levenshtein.ratio(remove_units_in_brackets(gt_json["column name"]).lower(), remove_units_in_brackets(gr_json["column name"]).lower())
        print(f"base score is: {base_score}")

    # VALUE COMPARISON
     # handle no value (like in LlaVa) issues
        if gr_json["value in chart 1"] == None: 
            gr_json["value in chart 1"] = float('inf')
        if gr_json["value in chart 2"] == None: 
            gr_json["value in chart 2"] = float('inf')
     
     # string to float (GT)
        if isinstance(gt_json["value in chart 1"], str):
            gt_json["value in chart 1"] = get_float_val(gt_json["value in chart 1"])
        if isinstance(gt_json["value in chart 2"], str):
            gt_json["value in chart 2"] = get_float_val(gt_json["value in chart 2"])
        if isinstance(gr_json["value in chart 1"], str):
            gr_json["value in chart 1"] = get_float_val(gr_json["value in chart 1"])
        if isinstance(gr_json["value in chart 2"], str):
            gr_json["value in chart 2"] = get_float_val(gr_json["value in chart 2"])

        try:
            chart_1_val_diff_percentage = abs(gr_json["value in chart 1"] - gt_json["value in chart 1"]) * 100. / gt_json["value in chart 1"]
        except:
            chart_1_val_diff_percentage = float('inf')

        try: 
            chart_2_val_diff_percentage = abs(gr_json["value in chart 2"] - gt_json["value in chart 2"]) * 100. / gt_json["value in chart 2"]
        except:
            chart_2_val_diff_percentage = float('inf')

        # print(f"chart 1 val diff: {chart_1_val_diff_percentage}%")
        # print(f"chart 2 val diff: {chart_2_val_diff_percentage}%")

        avg_val_diff = (chart_1_val_diff_percentage + chart_2_val_diff_percentage) / 2

        final_score = base_score + 4 * max(0, (1 - (avg_val_diff/100.)))
    except:
        print(f"ERROR, extracting json value!")
        return 0.
    print(f"final score: {final_score}")
    return final_score

def get_list_from_dict(json_dict):
    json_list = []
    for idx in json_dict.keys():
        json_element = json_dict[idx]
        json_list.append(json_element)
    return json_list

# net difference, ie for all cells (1/2/3)
def get_net_cell_changes_score(chart_diff_gt_json_dict, chart_diff_gr_json_dict, cells_change_cnt):

    gt_json_list = get_list_from_dict(chart_diff_gt_json_dict)
    gr_json_list = get_list_from_dict(chart_diff_gr_json_dict)

    # try all permutations (ground-truth list vs generated list)
    index_list_perm = permutations(range(cells_change_cnt))

    max_net_score = 0.
    for index_list in list(index_list_perm):

        print(index_list)
        net_score = 0.
        for idx in range(cells_change_cnt):
            gt_idx = idx
            gr_idx = index_list[idx]
            net_score += get_json_similarity(gt_json_list[gt_idx], gr_json_list[gr_idx])

        print(f"net score is: {net_score}")    
        max_net_score = max(max_net_score, net_score)
        print(f"Net max score is: {max_net_score}")
    # breakpoint()

    return max_net_score / cells_change_cnt

# def get_random_score(data_count, cells_change_cnt):

#     score = 0.

#     # random row score
#     score += 3. / data_count[0] 
#     # random column score
#     score += 3./data_count[1]

#     # random probability due to values
#     score += 2.
#     return score

def chart_type_change_update_stats(chart_type_cnt_list, chart_type_mean_std_dev_corr_list, chart_type_mean_of_mean_score_list, chart_type_std_dev_of_mean_score_list, chart_type_mean_of_std_dev_of_score_list, chart_type_std_dev_of_std_dev_of_score_list, current_chart_type_mean_and_std_dev_list, chart_type_mean_score_list_list, chart_type_std_dev_score_list_list):
    print(f"change of chart type\n")
    print(f"mean/std-dev list is: {current_chart_type_mean_and_std_dev_list}")

    # count of chart-type images
    chart_type_cnt_list.append(len(current_chart_type_mean_and_std_dev_list))

    # extract mean list
    current_chart_type_mean_list = [tmp[0] for tmp in current_chart_type_mean_and_std_dev_list]
    # extract std-dev list
    current_chart_type_std_dev_list = [tmp[1] for tmp in current_chart_type_mean_and_std_dev_list]

    # calculate correlation-coefficient between mean and std-dev
    
    # Kendall
    # corr, p_value = kendalltau(np.array(current_chart_type_mean_list), np.array(current_chart_type_std_dev_list))
    #Spearman
    corr, p_value = spearmanr(np.array(current_chart_type_mean_list), np.array(current_chart_type_std_dev_list))
    print(f"corr is: {corr}, p-value is: {p_value}")
    chart_type_mean_std_dev_corr_list.append([corr, p_value])

    # pearson test (linear)
    # mean_std_dev_corr = np.corrcoef(current_chart_type_mean_list, current_chart_type_std_dev_list)[0, 1]
    # print(f"correlation is: {mean_std_dev_corr}")
    # chart_type_mean_std_dev_corr_list.append(mean_std_dev_corr)

    ### LIST BY REFERENCE (here new created, hence no overlap conflict)
    # (mean-score list for chart_type) list
    chart_type_mean_score_list_list.append(current_chart_type_mean_list)
    # (std-dev of score list for chart_type) list
    chart_type_std_dev_score_list_list.append(current_chart_type_std_dev_list)

    # mean
    mean_of_mean_score = round(np.mean(current_chart_type_mean_list), 1)
    std_dev_of_mean_score = round(np.std(current_chart_type_mean_list), 2)

    chart_type_mean_of_mean_score_list.append(mean_of_mean_score)
    chart_type_std_dev_of_mean_score_list.append(std_dev_of_mean_score)

    print(f"mean_of_mean_score: {mean_of_mean_score}, std_dev_of_mean_score: {std_dev_of_mean_score}")

    # standard deviation
    mean_of_std_dev_of_score = round(np.mean(current_chart_type_std_dev_list), 1)
    std_dev_of_std_dev_of_score = round(np.std(current_chart_type_std_dev_list), 2)
    
    chart_type_mean_of_std_dev_of_score_list.append(mean_of_std_dev_of_score)
    chart_type_std_dev_of_std_dev_of_score_list.append(std_dev_of_std_dev_of_score)

    print(f"mean_of_std_dev_of_score: {mean_of_std_dev_of_score}, std_dev_of_std_dev_of_score: {std_dev_of_std_dev_of_score}")

    current_chart_type_mean_and_std_dev_list.clear()
    # breakpoint()

    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--cells_change_cnt", type=int, default=1, help="no of cells altered in chart")
    parser.add_argument('-t', "--charts_per_type", type=int, default=50, help="charts to process per type: max is 200 each")
    parser.add_argument('-a', "--attribute_altered", type=str, default="color", help="attribute altered, to check for robustness, color/legend/text_style")
    parser.add_argument('-m', "--model_name", type=str, default="phi3", help="model used for inference to give current results")

    args = parser.parse_args()
    print(f"args are: {args}")

    cells_change_cnt = args.cells_change_cnt
    model_name = args.model_name
    charts_per_type = args.charts_per_type
    attribute_altered = args.attribute_altered

    file_path = os.path.join(os.getcwd(), f"_infer_outputs/robustness/{model_name}/{attribute_altered}/{cells_change_cnt}_cells_{charts_per_type}_charts.json")
    robustness_infer_data = None
    with open(file_path, encoding='utf-8') as f:
        robustness_infer_data = json.load(f)
    print(f"robustness response length is: {len(robustness_infer_data)}")

    os.makedirs(os.path.join(os.getcwd(), "_eval_results", "robustness" , model_name, attribute_altered), exist_ok=True)
    file_to_save_res = os.path.join(os.getcwd(), "_eval_results", "robustness", model_name, attribute_altered, str(cells_change_cnt) + "_cells" + ".csv")
    # file_to_save_res = os.path.join(os.getcwd(), "_eval_results", "robustness", model_name, attribute_altered, str(cells_change_cnt) + "_cells_" + str(charts_per_type) + "_charts.csv")


    current_chart_type_mean_and_std_dev_list = []

    chart_type_name_list = []
    chart_type_cnt_list = []

    # correlation b/w mean and std-dev of sets (5 chart images), calculated for each chart type
    chart_type_mean_std_dev_corr_list = []

    chart_type_mean_of_mean_score_list = []
    chart_type_std_dev_of_mean_score_list = []
    
    chart_type_mean_of_std_dev_of_score_list = []
    chart_type_std_dev_of_std_dev_of_score_list = []

    chart_type_mean_score_list_list = []
    chart_type_std_dev_score_list_list = []


    for idx in range(len(robustness_infer_data)):

        robustness_infer_data_element = robustness_infer_data[idx]

        chart_name = robustness_infer_data_element["imgname"]
        ground_truth_json_dict = robustness_infer_data_element["cell_change_ground_truth_JSON"]
        print(f"chart: {chart_name}")
        print(f"ground truth: {ground_truth_json_dict}")
        
        chart_info = get_chart_info_json(chart_name)
        chart_type = chart_info["chart_type"]

        chart_pair_imgs_set = robustness_infer_data_element["chart_pair_imgs_set"]
        # robustness set - parse it
        current_robustness_set_score_list = []
        for set_idx in chart_pair_imgs_set.keys():

            attribute_val_json_dict = chart_pair_imgs_set[set_idx]["attribute_value"]
            print(f"{set_idx} | attribute_value is: {attribute_val_json_dict}")

            generated_res_json_dict = chart_pair_imgs_set[set_idx]["cell_change_generated_JSON"]
            print(f"{set_idx} | generated_res: {generated_res_json_dict}")
            score = get_net_cell_changes_score(ground_truth_json_dict, generated_res_json_dict, cells_change_cnt)
            print(f"CALCULATED SCORE: {score}")
            # breakpoint()

            current_robustness_set_score_list.append(score)
        
        # calculate mean and std-dev for the set | and append to current chart type stats
        mean_of_robustness_set = np.mean(current_robustness_set_score_list)
        # std_dev_of_robustness_set = np.std(current_robustness_set_score_list)
        std_dev_of_robustness_set = np.max(current_robustness_set_score_list) - np.min(current_robustness_set_score_list)
        current_chart_type_mean_and_std_dev_list.append((mean_of_robustness_set, std_dev_of_robustness_set))

        # check if chart-type change (next chart different OR end of loop)
        isChartTypeChange = (idx+1 == len(robustness_infer_data)) or chart_type != get_chart_info_json(robustness_infer_data[idx+1]["imgname"])["chart_type"]

        if isChartTypeChange:
            chart_type_name_list.append(chart_type)
            chart_type_change_update_stats(chart_type_cnt_list, chart_type_mean_std_dev_corr_list, chart_type_mean_of_mean_score_list, chart_type_std_dev_of_mean_score_list, chart_type_mean_of_std_dev_of_score_list, chart_type_std_dev_of_std_dev_of_score_list, current_chart_type_mean_and_std_dev_list, chart_type_mean_score_list_list, chart_type_std_dev_score_list_list)

    print(f"mean-score list list is: {chart_type_mean_score_list_list}")
    print(f"std-dev-score list list is: {chart_type_std_dev_score_list_list}")

    # save (chart type score stats) to csv
    with open(file_to_save_res, 'w') as file:
        temp_writer = csv.writer(file, delimiter=';')
        # csv header
        temp_writer.writerow(["chart_type", "mean_std_dev_corr", "mean_of_mean_score", "std_dev_of_mean_score", "mean_of_std_dev_of_score", "std_dev_of_std_dev_of_score", "chart_images_count", "mean_score_list", "std_dev_score_list"])

        for idx in range(len(chart_type_cnt_list)):
            info_to_write = [chart_type_name_list[idx], chart_type_mean_std_dev_corr_list[idx], chart_type_mean_of_mean_score_list[idx], chart_type_std_dev_of_mean_score_list[idx], chart_type_mean_of_std_dev_of_score_list[idx], chart_type_std_dev_of_std_dev_of_score_list[idx], chart_type_cnt_list[idx], chart_type_mean_score_list_list[idx], chart_type_std_dev_score_list_list[idx]]
            temp_writer.writerow(info_to_write)        


    # df_wanted = df[df["data_count"] != -1]
    # correlation = df_wanted["score"].corr(df_wanted["data_count"])
    # print(f"\n\ncorrelation is: {correlation}")

    ###----------------------------------------------------
