import os

ChartX_imgs_from_code_run_path = os.path.join(os.getcwd(), "_datasets", "ChartX_imgs_from_code_run") 
# ChartX_data_path = os.path.join(os.getcwd(), "_datasets", "ChartX")
robustness_dataset_path = os.path.join(os.getcwd(), "_datasets", "ChartX_robustness")

json_path = os.path.join(os.getcwd(), "_datasets", 'ChartX_annotation.json')


from model_inference_helper import Phi3, LlaVa, InternVL2  # Import the models
from qwen_inference_helper import Qwen

from pathlib import Path
import csv
# import pandas as pd
import json

import logging
    
from PIL import Image 
import requests 

import argparse

from tqdm import tqdm
from itertools import islice

with open(json_path, encoding='utf-8') as ChartX_data:
    data = json.load(ChartX_data)


# Chart types
chart_type_cnt = 9
chart_dir_names_type_list = ['bar_chart', 'bar_chart_num', '3D-Bar', 'line_chart', 'line_chart_num', 'radar', 'rose', 'box', 'multi-axes']

chart_file_names_type_prefix_list = ['bar_', 'bar_num_', '3D-Bar_', 'line_', 'line_num_', 'radar_', 'rose_', 'box_', 'multi-axes_']

chart_type_tag_list = ['bar', 'bar', 'bar', 'line', 'line', 'radar', 'rose', 'box', 'bar or line']
# chart_entity_tag_list = ['bar', 'bar', 'bar', 'point', 'point', 'axis', 'axis', 'box quantity', 'bar or point']

prompt_expr_chart_template = "Given table (csv format) for first chart_type_tag chart:-\nimage_1_csv_tag\nGiven table (csv format) for second chart_type_tag chart:-\nimage_2_csv_tag\nThe second chart differs from first due to change in value of cells_change_cnt_tag of the cell(s). Can you identify the cells_change_cnt_tag cell(s)? Mention final answer of form: \"[<cell i json> for for all i cells with value change]\" where json is of form: {\"row name\": <row name for the cell>, \"column name\": <column name for the cell>, \"value in chart 1\": <cell value in chart 1>, \"value in chart 2\": <cell value in chart 2>}. Only cells_change_cnt_tag cell(s) hence output only cells_change_cnt_tag json in list, no explaination needed."


# get csv info for an IMAGE from ChartX_json
def get_img_csv_from_ChartX_json(imgname):
    for data_element in data:
        if data_element['imgname'] == imgname:
            chart_csv = data_element["csv"]
            chart_csv = chart_csv.replace(" \\t ", " \t ").replace(" \\n ", " \n ")
            return chart_csv
    return None


# construct csv (printed string) from file | same format as LLM model output
def get_string_from_csv_list(list_l):

    s = ""
    for idx in range(len(list_l)):
        if idx != 0:
            s += ","
        s += list_l[idx]
    return s

# json extracted from LLM response
### COMMON for all models
def extract_json_from_res(val, cells_change_cnt):

    val = val.replace('\n', '')
    chart_change_json_dict = {}

    for i in range(cells_change_cnt):
        
        chart_change_json_element = None
        try:
            idx_start = val.find('{')
            idx_end = val.find('}') + 1
            json_val = val[idx_start:idx_end]

            val = val.replace(json_val, "*", 1)

            chart_change_json_element = json.loads(json_val)
            # remove units  -> DO DURING EVAL
            # if '(' in chart_change_json_element["column name"]:
            #     chart_change_json_element["column name"] = remove_units_in_brackets(chart_change_json_element["column name"])

        except:
            logging.info("couldn't extract json, building a dummy -ve.")
            chart_change_json_element = {}
            chart_change_json_element["row name"] = "sample row"
            chart_change_json_element["column name"] = "sample column"
            chart_change_json_element["value in chart 1"] = float('inf')
            chart_change_json_element["value in chart 2"] = float('inf')


        chart_change_json_dict[i] = chart_change_json_element
    
    return chart_change_json_dict


# image input into LLM, get csv output
def get_csv_from_image(image, model_obj):
    
    prompt_text_template = "Given a chart image: image_1_tag\nGenerate the table (csv format) for the chart data. Only output the table directly."
    image_list = [image]

    csv_response = model_obj.infer(prompt_text_template=prompt_text_template, images_list=image_list)

    if isinstance(model_obj, LlaVa): #LlaVA: double quotes printed, but handle in json evaluation
        csv_response = csv_response.replace('\"', '')

    return csv_response

# LLM (compare 2 images, output differences)
def get_robustness_response(chart_name, prompt_expr_chart, image_1, image_1_csv, image_2, cells_change_cnt, model_obj):

    image_1_csv_generated = get_csv_from_image(image_1, model_obj)
    image_2_csv_generated = get_csv_from_image(image_2, model_obj)

    logging.info(f"image-1 GT csv is: {image_1_csv}\n")
    logging.info(f"image-1 generated csv is: {image_1_csv_generated}\n")

    logging.info(f"image-2 generated csv is: {image_2_csv_generated}\n")   

    final_prompt = prompt_expr_chart.replace("image_1_csv_tag", image_1_csv_generated).replace("image_2_csv_tag", image_2_csv_generated)
    logging.info(f"final prompt is: {final_prompt}")
    
    response = model_obj.infer(final_prompt)

    logging.info(f"RESPONSE IS: {response}\n")
    # response = response.replace('\n', '')

    chart_change_json_dict = extract_json_from_res(response, cells_change_cnt)
    logging.info(f"Extracted JSON IS: {chart_change_json_dict}\n")

    return chart_change_json_dict

# extract json for image (its robustness set) from JSON
def get_img_robustness_info_json(robustness_cell_change_attribute_altered_info_json, chart_name):

    for img_robustness_info_json in robustness_cell_change_attribute_altered_info_json:
        if img_robustness_info_json["imgname"] == chart_name:
            return img_robustness_info_json
    
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--cells_change_cnt", type=int, default=1, help="no of cells altered in chart")
    parser.add_argument('-a', "--attribute_altered", type=str, default="legend", help="attribute altered for robustness")
    parser.add_argument('-t', "--charts_per_type", type=int, default=1, help="charts to process per type: max is 200 each")
    parser.add_argument('-m', "--model_name", type=str, default="phi3", help="model to inference")
    parser.add_argument('-g', "--gpu_no", type=int, default=0, help="device on which model will run")
    parser.add_argument('-s', "--starting_chart_type_idx", type=int, default=0, help="in list of chart-types, position from where start inference")
    
    args = parser.parse_args()
    # Configure logging
    os.makedirs(os.path.join(os.getcwd(), "_infer_outputs", "robustness_logs" , args.model_name, args.attribute_altered), exist_ok=True)
    logging.basicConfig(filename=f'_infer_outputs/robustness_logs/{args.model_name}/{args.attribute_altered}/{args.cells_change_cnt}_cells_{args.charts_per_type}_charts.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"arguments are: {args}")
    
    # # Log messages
    # logging.debug('This is a debug message') # Will not be printed because level is INFO
    logging.info('This is an info message')
    # logging.warning('This is a warning message')
    # logging.error('This is an error message')
    # logging.critical('This is a critical message')


    ### All class (inference), 1. initialize model 2. replace image_tag with appropriate format used in the model 3. inference
    model_name = args.model_name
    device = f"cuda:{args.gpu_no}"
    model_obj = None
    # create model object (model to inference)
    if model_name == "phi3":
        model_obj = Phi3(device)
    elif model_name == "llava":
        model_obj = LlaVa(device)
    elif model_name == "internvl2":
        model_obj = InternVL2(device)
    elif model_name == "qwen":
        model_obj = Qwen(device)    
    else:
        raise Exception("Invalid model!")


    # how many cells in charts different 
    cells_change_cnt = args.cells_change_cnt
    prompt_expr_chart = prompt_expr_chart_template.replace("cells_change_cnt_tag", str(cells_change_cnt))
    if cells_change_cnt == 1: # singular
        prompt_expr_chart = prompt_expr_chart.replace("cell(s)", "cell")
    else:                     # plural
        prompt_expr_chart = prompt_expr_chart.replace("cell(s)", "cells")


    # plot-attribute in chart which has been altered. (to analyze data-change identification robustness against the attribute)
    attribute_altered = args.attribute_altered


    # how many charts (per type ie bar/line/3D) to process | max possible: 500 charts per type
    charts_per_type = args.charts_per_type


    # dir to access modified charts
    robustness_cell_change_attribute_altered_dir_path = os.path.join(robustness_dataset_path, f"imgs_{cells_change_cnt}_cell_change_{attribute_altered}_altered")
    robustness_cell_change_attribute_altered_info_json_path = os.path.join(robustness_dataset_path, f"info_{cells_change_cnt}_cell_change_{attribute_altered}_altered.json")
    robustness_cell_change_attribute_altered_info_json = None
    with open(robustness_cell_change_attribute_altered_info_json_path, "r", encoding = "utf-8") as f:
        robustness_cell_change_attribute_altered_info_json = json.load(f)
    
    # results saved in JSON
    os.makedirs(os.path.join(os.getcwd(), "_infer_outputs", "robustness" , model_name, attribute_altered), exist_ok=True)
    file_to_save_res = os.path.join(os.getcwd(), "_infer_outputs", "robustness" , model_name, attribute_altered, str(cells_change_cnt) + "_cells_" + str(charts_per_type) + "_charts.json")

    starting_chart_type_idx = args.starting_chart_type_idx
    if starting_chart_type_idx == 0:
        with open(file_to_save_res, 'w') as f:
            pass
    if (starting_chart_type_idx > 0) and (not os.path.exists(file_to_save_res)):
        raise Exception("Results for initial chart-types (being skipped) not present.")

    for idx, (chart_type, chart_file_names_type) in tqdm(
    enumerate(islice(zip(chart_dir_names_type_list, chart_file_names_type_prefix_list),starting_chart_type_idx, None), start = starting_chart_type_idx),
    total=chart_type_cnt - starting_chart_type_idx,  # Adjust total count to reflect the skipped elements
    desc=f"Processing, for each chart-type, No of chart image-pairs: {charts_per_type}, Cells altered: {cells_change_cnt}"
    ):
    # for idx, (chart_type, chart_file_names_type) in tqdm(enumerate(zip(chart_dir_names_type_list, chart_file_names_type_prefix_list), start=0), total=chart_type_cnt, desc=f"Processing, for each chart-type robustness-set, No of chart image-pairs: {charts_per_type}, Cells altered: {cells_change_cnt}"):
    # for idx, chart_type, chart_file_names_type in zip(range(chart_type_cnt), chart_dir_names_type_list, chart_file_names_type_prefix_list):

        # 3D-bar and box charts don't have LEGENDS
        if attribute_altered == "legend" and chart_type in ["3D-Bar", "box"]:
            continue
        
        # org_chart_type_csv_dir_path = ChartX_data_path + '/' + chart_type + "/csv" 

        prompt_expr_chart_type_specific = prompt_expr_chart.replace('chart_type_tag', chart_type_tag_list[idx]) #.replace('chart_entity_tag', chart_entity_tag_list[idx])
        logging.info(f"final prompt is: {prompt_expr_chart_type_specific}")
        # chart_file_list = glob.glob(robustness_cell_change_attribute_altered_dir_path + '/' + chart_file_names_type + '*' + '.png')
        # logging.info(f"chart file list is: {chart_file_list[:20]}")
        # ** alphabetical: 102 before 23

        robustness_cell_change_attribute_altered_chart_type_dir_list = [dirname for dirname in sorted(os.listdir(robustness_cell_change_attribute_altered_dir_path)) if dirname.startswith(chart_file_names_type)]
        # logging.info(f"list is: {robustness_cell_change_attribute_altered_chart_type_dir_list}")

        # info for current chart-type
        inference_json_list = []

        imags_processed_for_type_cnt  =0
        for i in range(len(robustness_cell_change_attribute_altered_chart_type_dir_list)):

            chart_name = robustness_cell_change_attribute_altered_chart_type_dir_list[i]
            logging.info(f"img name is: {chart_name}")

            # org_chart_csv_path = org_chart_type_csv_dir_path + '/' + chart_name + '.csv'
            org_chart_csv = get_img_csv_from_ChartX_json(chart_name)


            # for image: get JSON info & dir-path (for its robustness set)
            img_robustness_set_path = os.path.join(robustness_cell_change_attribute_altered_dir_path, chart_name)
            
            img_robustness_info_json = get_img_robustness_info_json(robustness_cell_change_attribute_altered_info_json, chart_name)
            if not img_robustness_info_json:    # no JSON exists
                # breakpoint()
                continue
            #?** Check later (bar_112) - images but JSON not generated

            chart_pair_imgs_set_info_json = img_robustness_info_json["chart_pair_imgs_set"]
            chart_pair_cell_change_json = img_robustness_info_json["cell_change_JSON"]
            logging.info(f"Data-difference in between chart pairs is: {chart_pair_cell_change_json}\n")

            # build json (inference results) to save
            img_robustness_infer_results_json = {}
            img_robustness_infer_results_json["imgname"] = img_robustness_info_json["imgname"]
            img_robustness_infer_results_json["cell_change_ground_truth_JSON"] = img_robustness_info_json["cell_change_JSON"]
            img_robustness_infer_results_json["chart_pair_imgs_set"] = {}

            # loop on robustness set (pair of images differing in data ie cell change | and multiple pairs differing in plot attribute - attribute_altered)
            for idx in chart_pair_imgs_set_info_json.keys():

                img_robustness_infer_results_json["chart_pair_imgs_set"][idx] = {}

                attribute_value = chart_pair_imgs_set_info_json[idx]["attribute_value"]

                initial_chart_name = chart_pair_imgs_set_info_json[idx]["initial_chart_name"]
                modified_chart_name = chart_pair_imgs_set_info_json[idx]["modified_chart_name"]

                initial_chart_path = os.path.join(img_robustness_set_path, initial_chart_name + ".png")
                modified_chart_path = os.path.join(img_robustness_set_path, modified_chart_name + ".png")
                # logging.info(initial_chart_path)
                # logging.info(modified_chart_path)

                if Path(initial_chart_path).is_file() and Path(modified_chart_path).is_file():
                    pass
                else:
                    logging.info("ERROR, INCORRECT FILE PATH")
                    continue

                logging.info(f"image paths are - image_1: {initial_chart_path}, image_2: {modified_chart_path}")

                image_1_csv = org_chart_csv #"\n"
                # with open(org_chart_csv_path, mode ='r') as file:    
                #     csvFile = csv.reader(file)
                #     for lines in csvFile:
                #         # logging.info(lines)
                #         image_1_csv += get_string_from_csv_list(lines) + '\n'
                # logging.info(f"final csv info is: {image_1_csv}")

                image_1 = Image.open(initial_chart_path) 
                image_2 = Image.open(modified_chart_path)

                cell_change_generated_JSON = get_robustness_response(chart_name, prompt_expr_chart_type_specific, image_1, image_1_csv, image_2, cells_change_cnt, model_obj)
                # if model fail: dummy json -ve created (which scores zero)

                img_robustness_infer_results_json["chart_pair_imgs_set"][idx]["attribute_value"] = attribute_value
                img_robustness_infer_results_json["chart_pair_imgs_set"][idx]["cell_change_generated_JSON"] = cell_change_generated_JSON #inference result JSON
                
            inference_json_list.append(img_robustness_infer_results_json)

            imags_processed_for_type_cnt += 1
            if imags_processed_for_type_cnt == charts_per_type:
                break

        # Check JSON (if storage, extract the list)
        inference_json_list_in_file = []
        if os.path.exists(file_to_save_res) and (os.stat(file_to_save_res).st_size != 0):
            with open(file_to_save_res, encoding = 'utf-8') as f:
                inference_json_list_in_file = json.load(f)

        # concatenate lists & re-write to JSON file
        net_inference_json_list_to_write = inference_json_list_in_file + inference_json_list
        with open(file_to_save_res, 'w') as f:
            json.dump(net_inference_json_list_to_write, f, indent=5)
        
    # with open(file_to_save_res, 'w') as f:
    #     json.dump(inference_json_list, f, indent=5)
