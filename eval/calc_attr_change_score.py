import os

import pandas as pd
import numpy as np
import json

import csv

from collections import defaultdict
from itertools import permutations

import logging
import argparse

# N to N matching for color attribute mapping
from scipy.optimize import linear_sum_assignment
# string comparison
from Levenshtein import distance  

import copy

chart_topic_list = ['Human Resources and Employee Management', 'Healthcare and Health', 'Arts and Culture', 'Business and Finance', 'Charity and Nonprofit Organizations', 'Tourism and Hospitality', 'Food and Beverage Industry', 'Technology and the Internet', 'Manufacturing and Production', 'Environment and Sustainability', 'Social Media and the Web', 'Retail and E-commerce', 'Science and Engineering', 'Law and Legal Affairs', 'Social Sciences and Humanities', 'Education and Academics', 'Energy and Utilities', 'Sports and Entertainment', 'Government and Public Policy', 'Agriculture and Food Production', 'Real Estate and Housing Market', 'Transportation and Logistics']

json_path = os.path.join(os.getcwd(), "_datasets", "ChartX_annotation.json")
with open(json_path, encoding='utf-8') as f:
    data = json.load(f)


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


def get_score_text_section(gt_text_section, gr_text_section):

    # dict keys: chart title, chart legend, chart axes labels, chart axes ticks
    # within each: size, weight, fontfamily (1.0, 0.75, 0.75)
    # each part - 2.5 points
    # size: % difference | weight, fontfamily: direct (or lenenshtein distance threshold)

    size_score, weight_score, fontfamily_score = 0., 0., 0.
    # size
    try:
        gt_size_info, gr_size_info = gt_text_section['size'], gr_text_section['size']


        gr_size_initial_val = eval(str(gr_size_info['initial value']).replace("pt", ""))
        gr_size_modified_val = eval(str(gr_size_info['modified value']).replace("pt", ""))

        initial_size_fractional_diff = abs(gr_size_initial_val - gt_size_info['initial value']) / gt_size_info['initial value']
        modified_val_fractional_diff = abs(gr_size_modified_val - gt_size_info['modified value']) / gt_size_info['modified value']

        print(f"gt sizes: {gt_size_info['initial value']}, {gt_size_info['modified value']}")
        print(f"gr sizes: {gr_size_initial_val}, {gr_size_modified_val}")
        
        print(f"initial size frac difference: {initial_size_fractional_diff}")
        print(f"modified size frac difference: {modified_val_fractional_diff}")
        # breakpoint()
        
        size_score = 0.5 * ((1 - initial_size_fractional_diff) + (1 - modified_val_fractional_diff)) 

        if size_score < 0.9:
            size_score = 0.

    except:
        if ('size' not in gr_text_section) and (gt_text_section['size']['initial value'] == gt_text_section['size']['modified value']):
            # breakpoint()
            size_score = 1.
        else:
            print(f"Error, text's size information not generated")

    # weight
    try:
        gt_weight_info, gr_weight_info = gt_text_section['weight'], gr_text_section['weight']
        initial_weight_correctly_detected = gt_weight_info['initial value'] == gr_weight_info['initial value']
        modified_weight_correctly_detected = gt_weight_info['modified value'] == gr_weight_info['modified value']
        weight_score = 0.375 * (initial_weight_correctly_detected + modified_weight_correctly_detected)

        if weight_score < 0.7:
            weight_score = 0.
    except:
        if ('weight' not in gr_text_section) and (gt_text_section['weight']['initial value'] == gt_text_section['weight']['modified value']):
            # breakpoint()
            weight_score = 0.75
        else:
            print(f"Error, text's weight information not generated")

    # font family
    try:
        gt_fontfamily_info, gr_fontfamily_info = gt_text_section['fontfamily'], gr_text_section['fontfamily']
        initial_fontfamily_correctly_detected = gt_fontfamily_info['initial value'] == gr_fontfamily_info['initial value']
        modified_fontfamily_correctly_detected = gt_fontfamily_info['modified value'] == gr_fontfamily_info['modified value']
        fontfamily_score = 0.375 * (initial_fontfamily_correctly_detected + modified_fontfamily_correctly_detected)
    
        if fontfamily_score < 0.7:
            fontfamily_score = 0.
    except:
        if ('fontfamily' not in gr_text_section) and (gt_text_section['fontfamily']['initial value'] == gt_text_section['fontfamily']['modified value']):
            # breakpoint()
            fontfamily_score = 0.75
        else:
            print(f"Error, text's fontfamily information not generated")

    return size_score, weight_score, fontfamily_score


def get_color_attr_mapping(L1, L2):
    n1, n2 = len(L1), len(L2)
    max_len = max(n1, n2)
    DUMMY = "<DUMMY>"

    # Pad the shorter list with dummy elements
    L1_padded = L1 + [DUMMY] * (max_len - n1)
    L2_padded = L2 + [DUMMY] * (max_len - n2)

    # Build the cost matrix
    cost_matrix = []
    for item1 in L1_padded:
        row = []
        for item2 in L2_padded:
            if DUMMY in (item1, item2):
                row.append(1000)  # High cost for matching real with dummy
            else:
                row.append(distance(item1, item2))
        cost_matrix.append(row)

    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Extract the mapping, ignoring dummy-dummy pairs
    mapping = []
    for row, col in zip(row_indices, col_indices):
        if L1_padded[row] != DUMMY and L2_padded[col] != DUMMY:
            mapping.append((L1_padded[row], L2_padded[col]))
        elif L1_padded[row] != DUMMY:  # Unmatched L1 item
            mapping.append((L1_padded[row], None))
        elif L2_padded[col] != DUMMY:  # Extra L2 item (not mapped to L1)
            pass  # Can also log if needed

    total_cost = sum(cost_matrix[row][col] for row, col in zip(row_indices, col_indices)
                     if L1_padded[row] != DUMMY and L2_padded[col] != DUMMY)

    print("Optimal Mapping:", mapping)
    print("Total Cost:", total_cost)

    return mapping


def get_rgb_from_hex(color_hex):
    color_rgb = None
    try:
        color_hex_val = color_hex.replace('#', '')
        color_rgb = tuple(int(color_hex_val[i:i+2], 16) for i in (0, 2, 4))
    except: 
        # color not in hex format
        print(f"color: {color_hex} not in hex format")
    return color_rgb

def calc_color_similarity(color_1, color_2):
    
    # rgb from hex color
    color_1_rgb, color_2_rgb = get_rgb_from_hex(color_1), get_rgb_from_hex(color_2)
    if not color_1_rgb or not color_2_rgb:
        return 0
    print(f"colors are: {color_1_rgb}, {color_2_rgb}")

    red_diff = abs(color_1_rgb[0] - color_2_rgb[0]) / 255.
    green_diff = abs(color_1_rgb[1] - color_2_rgb[1]) / 255.
    blue_diff = abs(color_1_rgb[2] - color_2_rgb[2]) / 255.
    avg_diff = (red_diff + green_diff + blue_diff) / 3

    color_similarity = 1 - avg_diff
    
    print(f"color similarity is: {color_similarity}")
    return color_similarity

# Define positions on a 3x3 grid
position_grid = {
    'upper left': (0, 0),
    'upper center': (1, 0),
    'upper right': (2, 0),
    'center left': (0, 1),
    'center': (1, 1),
    'center right': (2, 1),
    'lower left': (0, 2),
    'lower center': (1, 2),
    'lower right': (2, 2),
}

# Function to compute Manhattan distance between two positions
def manhattan_distance(pos1, pos2):
    coord1 = position_grid.get(pos1)
    coord2 = position_grid.get(pos2)
    if coord1 is None or coord2 is None:
        return None  # Handle invalid labels
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

# Function to convert distance to score
def score_from_distance(distance):
    if distance is None or distance > 4:
        return 0
    return max(5 - distance, 0)
        # making distance penalty higher -> better statistical significance (else all scores in range 8 to 10)


def get_net_attr_change_score(ground_truth_json_dict, generated_res_json_dict, attribute_altered):
    
    if not generated_res_json_dict:
        # json not extracted as per format
        return 0

    score = None
    # legend change
    if attribute_altered == 'legend':
        score = 0.

        initial_gt = ground_truth_json_dict['position']['initial value']
        initial_gen = generated_res_json_dict['position']['initial value']
        modified_gt = ground_truth_json_dict['position']['modified value']
        modified_gen = generated_res_json_dict['position']['modified value']

        score += score_from_distance(manhattan_distance(initial_gt, initial_gen))
        score += score_from_distance(manhattan_distance(modified_gt, modified_gen))

        # if ground_truth_json_dict['position']['initial value'] == generated_res_json_dict['position']['initial value']:
        #     score += 5
        # if ground_truth_json_dict['position']['modified value'] == generated_res_json_dict['position']['modified value']:
        #     score += 5        

    # font change
    elif attribute_altered == 'text_style':
        score = 0.
        section_scored_correctly = 0.
        # 4 text sections (title, legend, axes labels, axes ticks)
        for text_section in ground_truth_json_dict:
            section_score = 0.

            print(f"ground truth values for section are: {ground_truth_json_dict[text_section]}")
            try:
                print(f"generated values for section are: {generated_res_json_dict[text_section]}")
                size_score, weight_score, fontfamily_score = get_score_text_section(ground_truth_json_dict[text_section], generated_res_json_dict[text_section])
                section_score = size_score + weight_score + fontfamily_score
                print(f"{text_section}, size_score: {size_score}, weight_score: {weight_score}, fontfamily_score: {fontfamily_score}, total section score: {section_score}")    
            
                section_scored_correctly += 1
            except:
                print(f"generated dictionary doesn't have the text section.")
            
            score += 4*section_score
            # breakpoint()

        # divide by section count
        if section_scored_correctly != 0:
            score /= section_scored_correctly

    # color change
    else:
        score = 0.
 
        gt_color_attr_list = list(ground_truth_json_dict.keys())
        gr_color_attr_list = list(generated_res_json_dict.keys())

        # mapping between attribute names
        color_attr_mapping = get_color_attr_mapping(gt_color_attr_list, gr_color_attr_list)
        print(color_attr_mapping)

        # number of colors in the chart
        color_count = len(gt_color_attr_list)

        for idx in range(len(color_attr_mapping)):
            gt_color_attr, gr_color_attr = color_attr_mapping[idx][0], color_attr_mapping[idx][1]
            print(f"gt attribute: {gt_color_attr}, gr attribute: {gr_color_attr}")
            turn_score = None


            if gr_color_attr == None:
                # generated response, didn't generate for this attribute
                # if no color change (ground truth -> initial & final color same) then full score
                
                # check similarity more than 0.95
                if calc_color_similarity(ground_truth_json_dict[gt_color_attr]['initial value'], ground_truth_json_dict[gt_color_attr]['modified value']) > 0.95:
                    turn_score = (1./color_count)
                else: 
                    turn_score = 0.    # attribute not detected, zero 

            # very rare to happen
            elif gt_color_attr == None:
                # extra color (not in attribute), penalty 
                turn_score = -(1./color_count)

            # color comparison (between ground-truth and generated response)
            else:
                try:
                    initial_color_detection_correctness = calc_color_similarity(ground_truth_json_dict[gt_color_attr]['initial value'], generated_res_json_dict[gr_color_attr]['initial value'])
                    modified_color_detection_correctness = calc_color_similarity(ground_truth_json_dict[gt_color_attr]['modified value'], generated_res_json_dict[gr_color_attr]['modified value'])

                    # on scale of 0 to 1
                    color_detection_correctness = (initial_color_detection_correctness + modified_color_detection_correctness) / 2
                    turn_score = (1./color_count) * color_detection_correctness
                except:
                    # json structure incorrect (initial value, modified value NOT present)
                    turn_score = 0.

            score += turn_score
            print(turn_score)
        score *= 10   

    # final score (after either of if condition ie color/legend/text style done)
    return score

    
def chart_type_change_update_stats(chart_type_cnt_list, chart_type_mean_score_list, chart_type_std_dev_of_score_list, current_chart_type_score_list, chart_type_score_list_list):
    print(f"change of chart type\n")
    print(f"score list is: {current_chart_type_score_list}")

    current_chart_type_score_list_to_append = []
    for _, el in enumerate(current_chart_type_score_list):
        if el != -1:
            current_chart_type_score_list_to_append.append(el)
    # current_chart_type_score_list_to_append = copy.deepcopy(current_chart_type_score_list)
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

def is_chart_pair_identical(json_dict, attribute_altered):
    # check if chart pair is identical

    if attribute_altered in ["color", "legend"]:
        try:
            for attr in json_dict: # legend position or encoding (which is colored)
                if json_dict[attr]["initial value"] != json_dict[attr]["modified value"]:
                    return False       # different JSON
        except:
            return False               # incorrect formatting
        return True

    elif attribute_altered == "text_style":
        try:
            for chart_region in json_dict:
                chart_region_dict = json_dict[chart_region]
                for text_property in chart_region_dict:
                    if chart_region_dict[text_property]["initial value"] != chart_region_dict[text_property]["modified value"]:
                        return False    # different JSON
        except:
            return False                # incorrect formatting
        return True
    
    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--attribute_altered", type=str, default="legend", help="attribute altered in chart")
    parser.add_argument('-t', "--charts_per_type", type=int, default=200, help="charts to process per type: max is 200 each")
    parser.add_argument('-m', "--model_name", type=str, default="phi3", help="model used for inference to give current results")

    args = parser.parse_args()

    logging.basicConfig(filename=f'_{args.attribute_altered}_altered_{args.charts_per_type}_charts_eval.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"args are: {args}")

    attribute_altered = args.attribute_altered
    model_name = args.model_name
    charts_per_type = args.charts_per_type

    file_path = os.path.join(os.getcwd(), f"_infer_outputs/attr_edit/{model_name}/{attribute_altered}_altered_{charts_per_type}_charts.csv")
    df = pd.read_csv(file_path, sep=';')
    print(f"data frame length is: {len(df)}")

    os.makedirs(os.path.join(os.getcwd(), "_eval_results", "attr_edit" , model_name), exist_ok=True)
    file_to_save_res = os.path.join(os.getcwd(), "_eval_results", "attr_edit", model_name, attribute_altered + "_altered_" + str(charts_per_type) + "_charts.csv")


    chart_topic_score_list_dict = defaultdict(list)

    current_chart_type_score_list = []
    score_net_list = []

    chart_type_name_list = []
    chart_type_cnt_list = []
    chart_type_mean_score_list = []
    chart_type_std_dev_of_score_list = []
    
    chart_type_score_list_list = []

    chart_type_identical_pairs_list = []
    chart_type_detected_identical_pairs_list = []

    identical_pairs_cnt = 0
    detected_identical_pairs_cnt = 0

    for idx in range(len(df)):

        chart_name = df.iloc[idx, 0]
        ground_truth = df.iloc[idx, 1]
        generated_res = df.iloc[idx, 2]
        logging.info(f"chart: {chart_name}")
        print(f"ground truth: {ground_truth}")
        print(f"generated response: {generated_res}")

        ground_truth_json_dict = json.loads(ground_truth)

        generated_res_json_dict = None
        try:
            generated_res_json_dict = json.loads(generated_res)
        except: 
            print(f"No JSON for generated response")

        chart_info = get_chart_info_json(chart_name)
        chart_type = chart_info["chart_type"]

        # check if IDENTICAL IMAGES in the pair
        if is_chart_pair_identical(ground_truth_json_dict, attribute_altered):
            print(ground_truth_json_dict)
            identical_pairs_cnt += 1

            # check if generated also identical
            if (generated_res_json_dict) != None and ((not generated_res_json_dict) or is_chart_pair_identical(generated_res_json_dict, attribute_altered)):
                # if 1. no JSON (assuming "empty" meant model found no difference) OR empty-dict -> not generated_res_json_dict
                # OR 2. generated dict has identical info (initial == modified) 
                print(generated_res_json_dict)
                detected_identical_pairs_cnt += 1
                # breakpoint()
            
            continue

        # exclude empty JSON cases (compare between present ones)
        if (attribute_altered == "text_style") and (isinstance(generated_res_json_dict, dict) and (not generated_res_json_dict)):
            score = -1
        else:
            score = get_net_attr_change_score(ground_truth_json_dict, generated_res_json_dict, attribute_altered)
        logging.info(f"CALCULATED SCORE: {score}")

        current_chart_type_score_list.append(score)
        score_net_list.append(score)    # score-list for all charts in the file
        

        chart_topic = chart_info["topic"].replace("\n", "")
        chart_topic_score_list_dict[chart_topic].append(score)


        # check if chart-type change (next chart different OR end of loop)
        isChartTypeChange = (idx+1 == len(df)) or chart_type != get_chart_info_json(df.iloc[idx+1, 0])["chart_type"]

        if isChartTypeChange:
            chart_type_name_list.append(chart_type)
            chart_type_change_update_stats(chart_type_cnt_list, chart_type_mean_score_list, chart_type_std_dev_of_score_list, current_chart_type_score_list, chart_type_score_list_list)
            
            chart_type_identical_pairs_list.append(identical_pairs_cnt)
            identical_pairs_cnt = 0
            chart_type_detected_identical_pairs_list.append(detected_identical_pairs_cnt)
            detected_identical_pairs_cnt = 0

    print(f"score list list is: {chart_type_score_list_list}")
    # save (chart type score stats) to csv
    with open(file_to_save_res, 'w') as file:
        temp_writer = csv.writer(file, delimiter=';')
        # csv header
        temp_writer.writerow(["chart_type", "mean_score", "std_dev_of_score", "chart_images_count", "identical_pairs_count", "detected_identical_pairs_count", "score_list"])

        for idx in range(len(chart_type_cnt_list)):
            info_to_write = [chart_type_name_list[idx], chart_type_mean_score_list[idx], chart_type_std_dev_of_score_list[idx], chart_type_cnt_list[idx], chart_type_identical_pairs_list[idx], chart_type_detected_identical_pairs_list[idx], chart_type_score_list_list[idx]]
            temp_writer.writerow(info_to_write)        

    # net list (to same csv as reponses)
    score_net_list = [round(val, 1) for val in score_net_list]

# skipping identical examples, hence can't rewrite in same file
    # df["score"] = score_net_list 
    # modified_file_path = file_path.split('.')[0] + "_evaluated.csv"
    # df.to_csv(modified_file_path, sep = ';', index=False)

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

