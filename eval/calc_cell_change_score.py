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

import logging
import copy

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

    final_score = 0
    # ensure values are in int/flot form (some string form)
    try:
    # ROW COMPARISON
        # Levenshtein ratio -> floating number in range [0.0, 1.0] inversely proportional to corresponding distance
        row_score = 3 * Levenshtein.ratio(remove_units_in_brackets(gt_json["row name"]).lower(), remove_units_in_brackets(gr_json["row name"]).lower())

    # COLUMN COMPARISON
        # Levenshtein ratio & unit removal from brackets
        column_score = 3 * Levenshtein.ratio(remove_units_in_brackets(gt_json["column name"]).lower(), remove_units_in_brackets(gr_json["column name"]).lower())

        base_score = row_score + column_score

        print(f"row score is: {base_score}")
        print(f"column score is: {column_score}")
        print(f"base score is: {base_score}")
        
        # if (row_score > 2.) and (column_score > 2.):
        #     print(f"Correct row&column guess.")
        # else:
        #     print(f"Incorrect row&column guess.")
        #     raise Exception("Incorrect row/column match for the cell.")

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
            inidividual_cell_score = get_json_similarity(gt_json_list[gt_idx], gr_json_list[gr_idx])
            net_score += inidividual_cell_score
            
            # if inidividual_cell_score < 0.1:
            #     net_score = 0.
            #     break 
            # want all cells to be guessed correctly (incorrect guess, end loop)

        print(f"net score is: {net_score}")    
        max_net_score = max(max_net_score, net_score)
        print(f"Net max score is: {max_net_score}")
    # breakpoint()

    return max_net_score / cells_change_cnt


def chart_type_change_update_stats(chart_type_cnt_list, chart_type_mean_score_list, chart_type_std_dev_of_score_list, current_chart_type_score_list, chart_type_score_list_list):
    print(f"change of chart type\n")
    print(f"score list is: {current_chart_type_score_list}")

    current_chart_type_score_list_to_append = copy.deepcopy(current_chart_type_score_list)
    chart_type_score_list_list.append(current_chart_type_score_list_to_append)

    # count of chart-type images
    chart_type_cnt_list.append(len(current_chart_type_score_list))

    # mean
    score_mean = round(np.mean(current_chart_type_score_list), 3)
    chart_type_mean_score_list.append(score_mean)

    # standard deviation
    score_std_dev = round(np.std(current_chart_type_score_list), 3)
    chart_type_std_dev_of_score_list.append(score_std_dev)
    print(f"mean: {score_mean}, std_dev: {score_std_dev}")

    current_chart_type_score_list.clear()
    
    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--cells_change_cnt", type=int, default=1, help="no of cells altered in chart")
    parser.add_argument('-t', "--charts_per_type", type=int, default=200, help="charts to process per type: max is 200 each")
    parser.add_argument('-m', "--model_name", type=str, default="phi3", help="model used for inference to give current results")
    parser.add_argument('-ab', "--ablation_type", type=str, default="none", help="ablation: stitched(single stage) images, separate(single stage) image")

    args = parser.parse_args()

    # logging.basicConfig(filename=f'_{args.cells_change_cnt}_cells_{args.charts_per_type}_charts_eval.log', level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"args are: {args}")

    cells_change_cnt = args.cells_change_cnt
    model_name = args.model_name
    charts_per_type = args.charts_per_type
    ablation_type = args.ablation_type

    model_name_dir_part = model_name
    if ablation_type != "none": 
        model_name_dir_part = model_name_dir_part + f"/{ablation_type}"

    file_path = os.path.join(os.getcwd(), f"_infer_outputs/cell_edit/{model_name_dir_part}/{cells_change_cnt}_cells_{charts_per_type}_charts.csv")
    df = pd.read_csv(file_path, sep=';')
    print(f"data frame length is: {len(df)}")

    os.makedirs(os.path.join(os.getcwd(), "_eval_results", "cell_edit" , model_name_dir_part), exist_ok=True)
    file_to_save_res = os.path.join(os.getcwd(), "_eval_results", "cell_edit", model_name_dir_part, str(cells_change_cnt) + "_cells_" + str(charts_per_type) + "_charts.csv")


    chart_topic_score_list_dict = defaultdict(list)

    current_chart_type_score_list = []
    score_net_list = []

    chart_type_name_list = []
    chart_type_cnt_list = []
    chart_type_mean_score_list = []
    chart_type_std_dev_of_score_list = []

    chart_type_score_list_list = []

    for idx in range(len(df)):

        chart_name = df.iloc[idx, 0]
        ground_truth = df.iloc[idx, 1]
        generated_res = df.iloc[idx, 2]
        print(f"chart: {chart_name}")
        print(f"ground truth: {ground_truth}")
        print(f"generated response: {generated_res}")

        ground_truth_json_dict = json.loads(ground_truth)
        generated_res_json_dict = json.loads(generated_res)

        chart_info = get_chart_info_json(chart_name)
        chart_type = chart_info["chart_type"]


        score = get_net_cell_changes_score(ground_truth_json_dict, generated_res_json_dict, cells_change_cnt)
        print(f"CALCULATED SCORE: {score}")

        current_chart_type_score_list.append(score)
        score_net_list.append(score)    # score-list for all charts in the file
        

        chart_topic = chart_info["topic"].replace("\n", "")
        chart_topic_score_list_dict[chart_topic].append(score)


        # check if chart-type change (next chart different OR end of loop)
        isChartTypeChange = (idx+1 == len(df)) or chart_type != get_chart_info_json(df.iloc[idx+1, 0])["chart_type"]

        if isChartTypeChange:
            chart_type_name_list.append(chart_type)
            chart_type_change_update_stats(chart_type_cnt_list, chart_type_mean_score_list, chart_type_std_dev_of_score_list, current_chart_type_score_list, chart_type_score_list_list)

    print(f"score list list is: {chart_type_score_list_list}")
    # save (chart type score stats) to csv
    with open(file_to_save_res, 'w') as file:
        temp_writer = csv.writer(file, delimiter=';')
        # csv header
        temp_writer.writerow(["chart_type", "mean_score", "std_dev_of_score", "chart_images_count", "score_list"])

        for idx in range(len(chart_type_cnt_list)):
            info_to_write = [chart_type_name_list[idx], chart_type_mean_score_list[idx], chart_type_std_dev_of_score_list[idx], chart_type_cnt_list[idx], chart_type_score_list_list[idx]]
            temp_writer.writerow(info_to_write)        

    # net list (to same csv as reponses)
    score_net_list = [round(val, 1) for val in score_net_list]
    df["score"] = score_net_list 
    # df.to_csv(file_path, sep = ';', index=False)

    # chart-TOPIC stats
    for chart_topic in chart_topic_score_list_dict.keys():
        chart_topic_score_list = chart_topic_score_list_dict[chart_topic]
        print(f"chart topic: {chart_topic}")
        print(f"chart score list: {len(chart_topic_score_list)}")
        chart_topic_score_mean = np.mean(chart_topic_score_list)
        chart_topic_score_std_dev = np.std(chart_topic_score_list)

        chart_topic_score_list_dict[chart_topic] = [chart_topic_score_mean, chart_topic_score_std_dev]

    print(chart_topic_score_list_dict)


    # df_wanted = df[df["data_count"] != -1]
    # correlation = df_wanted["score"].corr(df_wanted["data_count"])
    # print(f"\n\ncorrelation is: {correlation}")

    ###----------------------------------------------------

