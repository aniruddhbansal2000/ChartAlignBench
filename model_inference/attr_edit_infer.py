import os

ChartX_imgs_from_code_run_path = os.path.join(os.getcwd(), "_datasets", "ChartX_imgs_from_code_run") 
# ChartX_data_path = os.path.join(os.getcwd(), "_datasets", "ChartX")
ChartX_attr_edit_path = os.path.join(os.getcwd(), "_datasets", "ChartX_attr_edit")

json_path = os.path.join(os.getcwd(), "_datasets", 'ChartX_annotation.json')


from model_inference_helper import Phi3_multi, LlaVa_multi, InternVL2_multi # Import the models
from qwen_inference_helper import Qwen_multi, Qwen

from pathlib import Path
import csv
# import pandas as pd
import json
import re

import logging


from PIL import Image 
import requests 

from tqdm import tqdm
from itertools import islice
import argparse

import copy

with open(json_path, encoding='utf-8') as ChartX_data:
    data = json.load(ChartX_data)

# Chart types
chart_type_cnt = 9
chart_dir_names_type_list = ['bar_chart', 'bar_chart_num', '3D-Bar', 'line_chart', 'line_chart_num', 'radar', 'rose', 'box', 'multi-axes']

chart_file_names_type_prefix_list = ['bar_', 'bar_num_', '3D-Bar_', 'line_', 'line_num_', 'radar_', 'rose_', 'box_', 'multi-axes_']


###---------------------------------------
prompt_info_extraction_template_tuple = {}

prompt_info_extraction_template_tuple["color"] = (
    "Can you list the attributes with unique colors in the chart image:- image_tag? Mention final answer of form: list[for all attributes 'i' -> {\"attribute\": <attribute 'i' name>, \"color\": <attribute 'i' color <hex color value>>}]. Mention only final answer, no explanation required.",
    "[{\"Retail Sales\": \"#0000FF\"}, {\"E-commerce Sales\": \"#FF0000\"}]",
    "Great! Can you do the same for following chart image:- image_tag?"
)

prompt_info_extraction_template_tuple["legend"] = (
    "Can you identify the legend position for the chart image:- image_tag? Mention answer of form json: {\"legend position\": <position of legend in the chart>}\". Position of legend is not the order of items in legend but instead position of the legend box in chart. There are 9 possible values:- ['upper right', 'upper left', 'lower left', 'lower right', 'center left', 'center right', 'lower center', 'upper center', 'center'].",
    "{\"legend position\": \"upper left\"}",
    "Great! Can you do the same for following chart image:- image_tag?"
)

text_extraction_message_content = """Can you identify the text style for the chart image image_tag. Provide the answer as JSON in this format:-
{
  "chart title": {
    "size": <numerical value (pt)>,
    "weight": <"light" | "normal" | "bold">,
    "fontfamily": <"sans-serif" | "serif" | "cursive" | "fantasy" | "monospace">
  },
  "chart legend": {
    "size": <numerical value (pt)>,
    "weight": <"light" | "normal" | "bold">,
    "fontfamily": <"sans-serif" | "serif" | "cursive" | "fantasy" | "monospace">
  },
  "chart axes labels": {
    "size": <numerical value (pt)>,
    "weight": <"light" | "normal" | "bold">,
    "fontfamily": <"sans-serif" | "serif" | "cursive" | "fantasy" | "monospace">
  },
  "chart axes ticks": {
    "size": <numerical value (pt)>,
    "weight": <"light" | "normal" | "bold">,
    "fontfamily": <"sans-serif" | "serif" | "cursive" | "fantasy" | "monospace">
  }
}
"""

prompt_info_extraction_template_tuple["text_style"] = (
    text_extraction_message_content,
    "{\"chart title\": {\"size\": 14, \"weight\": \"normal\", \"fontfamily\": \"sans-serif\"}, \"chart legend\": {\"size\": 12, \"weight\": \"normal\", \"fontfamily\": \"sans-serif\"}, \"chart axes labels\": {\"size\": 12, \"weight\": \"normal\", \"fontfamily\": \"sans-serif\"}, \"chart axes ticks\": {\"size\": 12, \"weight\": \"normal\", \"fontfamily\": \"sans-serif\"}}",
    "Great! Can you do the same for following chart image:- image_tag?"
)


###---------------------------------------
attr_change_ans_format_dict = {
    "color": "JSON {<attribute 1 json object>.....<attribute k json object> for all attributes i = 1 to k with COLOR CHANGE} where the <attribute i json object> format is: {\"<attribute name>\": {\"initial value\": <color from chart-1>, \"modified value\": <color from chart-2>}} Only include attributes where 'initial value' differs from 'modified value'",    
    "legend": "{\"position\": {\"initial value\": <legend position in chart-1>, \"modified value\": <legend position in chart-2>}}",
    "text_style": "{<chart-section-i>: <chart-section-i change json> for chart-section-i in [\"chart title\", \"chart legend\", \"chart axes\", \"chart ticks\"]} where <chart-section-i change json> is json form: {<text characteristic>: {\"initial value\": <value in chart-1>, \"modified value\": <value in chart-2>} for <text characteristic> in [\"size\", \"weight\", \"fontfamily\"]}." 
}
#Include only those text characteristics where the 'initial value' and 'modified value' differ. Do not include any unchanged attributes or additional explanations in the output"

prompt_attr_comparison_template_string = "Given attr_change_gt_type_tag information for 2 chart images. Chart-1:-initial_img_attr_info_tag\nChart-2:-modified_img_attr_info_tag\nThe charts differ in attr_change_gt_type_tag. Can you identify the change? Mention the final answer strictly of form: attr_change_ans_format_tag, no explaination required."

###---------------------------------------
sample_image_path = ChartX_imgs_from_code_run_path + '/' + "bar_100" + '.png' 
sample_image = Image.open(sample_image_path)



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

# image input into LLM, get csv output
def get_attr_info_from_image(image, attribute_altered, model_obj):
    
    prompt_info_extraction_list = list(copy.deepcopy(prompt_info_extraction_template_tuple[attribute_altered]))
    images_list_list = [[sample_image], [], [image]]

    if not in_context_example:
        # remove in-context example format
        # PROMPT -> REMOVE LATER PART
        prompt_info_extraction_list = prompt_info_extraction_list[:1]
        # IMAGE LIST -> REMOVE INITIAL 
        images_list_list = images_list_list[2:]

    attr_info_response = None
    if isinstance(model_obj, Qwen):
        attr_info_response = model_obj.infer(prompt_info_extraction_list[0].replace("image_tag", "image_1_tag"), images_list_list[0])
    else:
        attr_info_response = model_obj.infer(prompt_text_template_list = prompt_info_extraction_list, images_list_list = images_list_list)

    return attr_info_response


# LLM (compare 2 images, output differences)
# def get_response(chart_name, change_prompt, prompt_expr_chart, image_1, image_1_csv, image_2, cells_change_cnt, model_obj):
def get_response(chart_name, attr_change_gt_json, prompt_attr_comparison_attr_specific_string, image_1, image_2, attribute_altered, model_obj):

    attr_change_gt_json_str = json.dumps(attr_change_gt_json)
    response_list = [chart_name, attr_change_gt_json_str]

    image_1_attr_info_extracted = get_attr_info_from_image(image_1, attribute_altered, model_obj)
    image_2_attr_info_extracted = get_attr_info_from_image(image_2, attribute_altered, model_obj)
    
    logging.info(f"image-1 {attribute_altered} extracted info is: {image_1_attr_info_extracted}\n")
    logging.info(f"image-2 {attribute_altered} extracted info is: {image_2_attr_info_extracted}\n")   

    final_prompt_content = prompt_attr_comparison_attr_specific_string.replace("initial_img_attr_info_tag", image_1_attr_info_extracted).replace("modified_img_attr_info_tag", image_2_attr_info_extracted)
    final_prompt = [final_prompt_content]
    logging.info(f"final prompt is: {final_prompt}")

    response = None
    if isinstance(model_obj, Qwen):
        response = model_obj.infer(final_prompt[0])
    else:
        response = model_obj.infer(final_prompt)


    logging.info(f"RESPONSE IS: {response}\n")
    # response = response.replace('\n', '')

    attr_change_detected_json_str = 'empty'

    attr_diff_info_str = response
    attr_diff_info_str = attr_diff_info_str.replace("```","")
    
    # LLM hallucinate, extra [] brackets, extract json from it..
    if attribute_altered == "color":
        attr_diff_info_str = attr_diff_info_str.replace('[', '').replace(']', '')

        if isinstance(model_obj, InternVL2_multi):
            # Need to remove additional brackets, ie change format
            # GET ->     {<attr_name>: {initial_value: <>, modified_value: <> }}, <attr_name>: {initial_value: <>, modified_value: <> }}, <attr_name>: {initial_value: <>, modified_value: <>}}
            # DESIRED -> {<attr_name>: {initial_value: <>, modified_value: <> },  <attr_name>: {initial_value: <>, modified_value: <> },  <attr_name>: {initial_value: <>, modified_value: <>}}

            logging.info("instance of InternVL-2, with color change")
            pos_st = attr_diff_info_str.find('{') + 1            # part of string after first {
            pos_end = attr_diff_info_str.rfind('}')           # part of string before last }
            attr_diff_info_str_part = attr_diff_info_str[pos_st : pos_end]

            attr_diff_info_str_part = attr_diff_info_str_part.replace("    {", "").replace("}}", "}")
            attr_diff_info_str = attr_diff_info_str[: pos_st] + attr_diff_info_str_part + attr_diff_info_str[pos_end : ]


    logging.info(f"Preprocessed string is: {attr_diff_info_str}")
    # breakpoint()
#--- PREPROCESS (extract json from attribute info)
    try:
        json_starting_pos = attr_diff_info_str.find('{')            # label (like 'json' before json string)
        json_ending_pos = attr_diff_info_str.rfind('}') + 1           # last occurence of }
        attr_diff_info_str = attr_diff_info_str[json_starting_pos : json_ending_pos]

        attr_diff_info_dict = json.loads(attr_diff_info_str)
        attr_diff_info_str_back = json.dumps(attr_diff_info_dict)
        logging.info(f"Attribute change json: {attr_diff_info_dict}")
        attr_change_detected_json_str = attr_diff_info_str_back
    except:
        logging.info("json extraction failed!")
        logging.info(f"attribute change string: {attr_diff_info_str}")

#---
    response_list.append(attr_change_detected_json_str)

    return response_list

def get_attr_change_info(imgname, attribute_altered, plot_attr_change_info_json):

    for data_element in plot_attr_change_info_json:
        if data_element["imgname"] == imgname and data_element["attr_change_type"] == attribute_altered:
            return data_element["attr_change_json"]     # return only the GT part
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--attribute_altered", type=str, default="legend", help="attribute altered for robustness")
    parser.add_argument('-t', "--charts_per_type", type=int, default=1, help="charts to process per type: max is 200 each")
    parser.add_argument('-m', "--model_name", type=str, default="phi3", help="model to inference")
    parser.add_argument('-s', "--starting_chart_type_idx", type=int, default=0, help="in list of chart-types, position from where start inference")
    parser.add_argument('-g', "--gpu_no", type=int, default=0, help="device on which model will run")
    # parser.add_argument('-tmp', "--model_temperature", type=float, default=0.0, help="temperature of sampling in model softmax")
    parser.add_argument('-ic', "--in_context_example", type=bool, default=False, help="device on which model will run")
    
    args = parser.parse_args()
    # model_temperature = args.model_temperature
    in_context_example = args.in_context_example

    os.makedirs(os.path.join(os.getcwd(), "_infer_outputs", "attr_edit_logs" , args.model_name), exist_ok=True)
    logging.basicConfig(filename=f'_infer_outputs/attr_edit_logs/{args.model_name}/{args.attribute_altered}_altered_{args.charts_per_type}_charts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"arguments are: {args}")

    ### All class (inference), 1. initialize model 2. replace image_tag with appropriate format used in the model 3. inference
    model_name = args.model_name

    model_obj = None
    device = f"cuda:{args.gpu_no}"
    # create model object (model to inference)
    if model_name == "phi3":
        model_obj = Phi3_multi(device)
    elif model_name == "llava":
        model_obj = LlaVa_multi(device)
    elif model_name == "internvl2":
        model_obj = InternVL2_multi(device)
    elif model_name == "qwen":
        model_obj = Qwen_multi(device)
    elif model_name == "gpt4":
        model_obj = Qwen(device)
    else:
        raise Exception("Invalid model!")

    # how many charts (per type ie bar/line/3D) to process | max possible: 200 charts per type
    charts_per_type = args.charts_per_type

    # altered - plot attribute
    attribute_altered = args.attribute_altered

    # dir to access modified charts
    modified_chart_dir_path = os.path.join(ChartX_attr_edit_path, f"{attribute_altered}_altered")

    plot_attr_change_info_json = None
    # ground-truth JSON
    plot_attr_change_info_json_path = os.path.join(ChartX_attr_edit_path, "attribute_edit_info.json")
    with open(plot_attr_change_info_json_path, encoding = 'utf-8') as f:
        plot_attr_change_info_json = json.load(f)

    # csv file to save
    os.makedirs(os.path.join(os.getcwd(), "_infer_outputs", "attr_edit" , model_name), exist_ok=True)
    file_to_save_res = os.path.join(os.getcwd(), "_infer_outputs", "attr_edit", model_name, str(attribute_altered) + "_altered_" + str(charts_per_type) + "_charts" + ".csv")
    
    starting_chart_type_idx = args.starting_chart_type_idx
    if starting_chart_type_idx == 0:
        with open(file_to_save_res, 'w') as f:
            pass
    elif (starting_chart_type_idx > 0) and (not os.path.exists(file_to_save_res)):
        raise Exception("Results for initial chart-types (being skipped) not present.")

    for idx, (chart_type, chart_file_names_type) in tqdm(
    enumerate(islice(zip(chart_dir_names_type_list, chart_file_names_type_prefix_list),starting_chart_type_idx, None), start = starting_chart_type_idx),
    total=chart_type_cnt - starting_chart_type_idx,  # Adjust total count to reflect the skipped elements
    desc=f"Processing, for each chart-type, No of chart image-pairs: {charts_per_type}, attribute altered: {attribute_altered}"
    ):
        
        prompt_attr_comparison_attr_specific_string = prompt_attr_comparison_template_string.replace("attr_change_gt_type_tag", attribute_altered).replace("attr_change_ans_format_tag", attr_change_ans_format_dict[attribute_altered])
        logging.info(f"final template is: {prompt_attr_comparison_attr_specific_string}")
        

        modified_chart_file_list = [filename for filename in sorted(os.listdir(modified_chart_dir_path)) if (filename.startswith(chart_file_names_type) and filename.endswith(".png"))]
        # logging.info(f"list is: {modified_chart_file_list}")

        imgs_processed_for_type_cnt = 0
        responses_list_for_chart_type = []
        for i in range(len(modified_chart_file_list)):

            chart_name = modified_chart_file_list[i].split('.')[0]

            org_chart_path = ChartX_imgs_from_code_run_path + '/' + chart_name + '.png' 
            modified_chart_path = modified_chart_dir_path + '/' + chart_name + '.png'

            if Path(org_chart_path).is_file() and Path(modified_chart_path).is_file():
                pass
            else:
                logging.info("ERROR, INCORRECT FILE PATH")
                continue
            
            logging.info(f"image paths are - image_1: {org_chart_path}, image_2: {modified_chart_path}")

            image_1 = Image.open(org_chart_path)
            image_2 = Image.open(modified_chart_path)

            attr_change_gt_json = get_attr_change_info(chart_name, attribute_altered, plot_attr_change_info_json)
            logging.info(f"Attribute difference between charts is: {attr_change_gt_json}\n")

            # images = [image_1, image_2]
            response_list = get_response(chart_name, attr_change_gt_json, prompt_attr_comparison_attr_specific_string, image_1, image_2, attribute_altered, model_obj)

            logging.info(response_list)
            responses_list_for_chart_type.append(response_list)
            
            # breakpoint()

            imgs_processed_for_type_cnt += 1
            if imgs_processed_for_type_cnt == charts_per_type:
                break

        try:
            with open(file_to_save_res, 'a', newline='\n') as file:
                temp_writer = csv.writer(file, delimiter=';')
                # csv header
                if os.stat(file_to_save_res).st_size == 0:
                    temp_writer.writerow(["chart name", "ground truth", "generated"])
                # inference results for just finished chart-type
                for idx in range(len(responses_list_for_chart_type)):
                    temp_writer.writerow(responses_list_for_chart_type[idx])
        except Exception as e:
            logging.info(f"Error writing to temp file: {e}")

