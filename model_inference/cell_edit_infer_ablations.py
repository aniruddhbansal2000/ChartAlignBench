import os

ChartX_imgs_from_code_run_path = os.path.join(os.getcwd(), "_datasets", "ChartX_imgs_from_code_run") 
# ChartX_data_path = os.path.join(os.getcwd(), "_datasets", "ChartX")
ChartX_cell_edit_path = os.path.join(os.getcwd(), "_datasets", "ChartX_cell_edit")

json_path = os.path.join(os.getcwd(), "_datasets", 'ChartX_annotation.json')

# os.environ['CUDA_PATH'] = "/opt/common/.modulefiles/cuda/10.1.243"

from model_inference_helper import Phi3, LlaVa, InternVL2  # Import the models
# from qwen_inference_helper import Qwen

from pathlib import Path
import csv
# import pandas as pd
import json

import logging

    
from PIL import Image 
import requests 

from tqdm import tqdm
from itertools import islice
import argparse

with open(json_path, encoding='utf-8') as ChartX_data:
    data = json.load(ChartX_data)

# Chart types
chart_type_cnt = 9
chart_dir_names_type_list = ['bar_chart', 'bar_chart_num', '3D-Bar', 'line_chart', 'line_chart_num', 'radar', 'rose', 'box', 'multi-axes']
chart_file_names_type_prefix_list = ['bar_', 'bar_num_', '3D-Bar_', 'line_', 'line_num_', 'radar_', 'rose_', 'box_', 'multi-axes_']




prompt_text_template = "Given a chart image: image_1_tag\nGenerate the table (csv format) for the chart data. Only output the table directly."



# prompt_expr_chart_template = "Given table (csv format) for first chart:-\nimage_1_csv_tag\nGiven table (csv format) for second chart:-\nimage_2_csv_tag\nThe second chart differs from first due to change in value of cells_change_cnt_tag of the cell(s). Can you identify the cells_change_cnt_tag cell(s)? Mention final answer of form: \"[<cell i json> for for all i cells with value change]\" where json is of form: {\"row name\": <row name for the cell>, \"column name\": <column name for the cell>, \"value in chart 1\": <cell value in chart 1>, \"value in chart 2\": <cell value in chart 2>}. Only cells_change_cnt_tag cell(s) hence output only cells_change_cnt_tag json in list, no explaination needed."

prompt_expr_chart_template_dict = {}
prompt_expr_chart_template_dict["stitched"] = "Given image has two charts stitched side by side: image_1_tag\nIf we compare the data of the charts i.e. csv table corresponding to the charts, there is difference in value of cells_change_cnt_tag cell(s). Can you identify the cells_change_cnt_tag cell(s)? Mention final answer of form: \"[<cell i json> for for all i cells with value change]\" where json is of form: {\"row name\": <row name for the cell>, \"column name\": <column name for the cell>, \"value in chart 1\": <cell value in chart 1>, \"value in chart 2\": <cell value in chart 2>}. Only cells_change_cnt_tag cell(s) hence output only cells_change_cnt_tag json in list, no explaination needed."
prompt_expr_chart_template_dict["separated"] = "Given two images each representing a chart, chart-1: image_1_tag, and chart-2: image_2_tag\nIf we compare the data of the charts i.e. csv table corresponding to the charts, there is difference in value of cells_change_cnt_tag cell(s). Can you identify the cells_change_cnt_tag cell(s)? Mention final answer of form: \"[<cell i json> for for all i cells with value change]\" where json is of form: {\"row name\": <row name for the cell>, \"column name\": <column name for the cell>, \"value in chart 1\": <cell value in chart 1>, \"value in chart 2\": <cell value in chart 2>}. Only cells_change_cnt_tag cell(s) hence output only cells_change_cnt_tag json in list, no explaination needed."

# prompt_expr_chart_template = "Given table (csv format) corresponding the first chart_type_tag_list chart - csv_content. And an image for the second chart_type_tag_list chart - image_2. The second chart varies from the first because a chart_entity_tag has been altered. Could you pinpoint which is the chart_entity_tag?"

# stitched images
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

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
            # remove units -> DO DURING EVAL
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


# # image input into LLM, get csv output
# def get_csv_from_image(image, model_obj):
    
#     prompt_text_template = "Given a chart image: image_1_tag\nGenerate the table (csv format) for the chart data. Only output the table directly."
#     image_list = [image]

#     csv_response = model_obj.infer(prompt_text_template=prompt_text_template, images_list=image_list)

#     if isinstance(model_obj, LlaVa): #LlaVA: double quotes printed, but handle in json evaluation
#         csv_response = csv_response.replace('\"', '')

#     return csv_response

# LLM (compare 2 images, output differences)
def get_response_ablation(chart_name, change_prompt, prompt_expr_chart, images_list, cells_change_cnt, model_obj):

    change_prompt_str = json.dumps(change_prompt)
    response_list = [chart_name, change_prompt_str]

    final_prompt = prompt_expr_chart
    logging.info(f"final prompt is: {final_prompt}")

    response = model_obj.infer(final_prompt, images_list)

    logging.info(f"RESPONSE IS: {response}\n")
    # response = response.replace('\n', '')

    chart_change_json_dict = extract_json_from_res(response, cells_change_cnt)
    logging.info(f"Extracted JSON IS: {chart_change_json_dict}\n")
    # breakpoint()

    json_str = json.dumps(chart_change_json_dict)
    response_list.append(json_str)

    return response_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--cells_change_cnt", type=int, default=1, help="no of cells altered in chart")
    parser.add_argument('-t', "--charts_per_type", type=int, default=1, help="charts to process per type: max is 200 each")
    parser.add_argument('-m', "--model_name", type=str, default="phi3", help="model to inference")
    parser.add_argument('-s', "--starting_chart_type_idx", type=int, default=0, help="in list of chart-types, position from where start inference")
    parser.add_argument('-g', "--gpu_no", type=int, default=0, help="device on which model will run")
    
    parser.add_argument('-ab', "--ablation_type", type=str, default="separated", help="ablation: stitched(single stage) images, separate(single stage) image")

    args = parser.parse_args()
    
    os.makedirs(os.path.join(os.getcwd(), "_infer_outputs", "cell_edit_logs" , args.model_name, args.ablation_type), exist_ok=True)
    logging.basicConfig(filename=f'_infer_outputs/cell_edit_logs/{args.model_name}/{args.ablation_type}/{args.cells_change_cnt}_cell_change_{args.charts_per_type}_charts.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"arguments are: {args}")

    ### All class (inference), 1. initialize model 2. replace image_tag with appropriate format used in the model 3. inference
    model_name = args.model_name
    model_obj = None
    device = f"cuda:{args.gpu_no}"
    # create model object (model to inference)
    if model_name == "phi3":
        model_obj = Phi3(device)
    elif model_name == "llava":
        model_obj = LlaVa(device)
    elif model_name == "internvl2":
        model_obj = InternVL2(device)
    # elif model_name == "qwen":
    #     model_obj = Qwen(device)
    else:
        raise Exception("Invalid model!")

    # applied ablation
    ablation_type = args.ablation_type
    prompt_expr_chart_template = prompt_expr_chart_template_dict[ablation_type]

    # how many cells in charts different 
    cells_change_cnt = args.cells_change_cnt
    prompt_expr_chart = prompt_expr_chart_template.replace("cells_change_cnt_tag", str(cells_change_cnt))
    if cells_change_cnt == 1: # singular
        prompt_expr_chart = prompt_expr_chart.replace("cell(s)", "cell")
    else:                     # plural
        prompt_expr_chart = prompt_expr_chart.replace("cell(s)", "cells")


    # how many charts (per type ie bar/line/3D) to process | max possible: 500 charts per type
    charts_per_type = args.charts_per_type

    # dir to access modified charts
    modified_chart_dir_path = ChartX_cell_edit_path + "/" + str(cells_change_cnt) + "_cell_edit"
    # csv file to save
    os.makedirs(os.path.join(os.getcwd(), "_infer_outputs", "cell_edit" , model_name, ablation_type), exist_ok=True)
    file_to_save_res = os.path.join(os.getcwd(), "_infer_outputs", "cell_edit", model_name, ablation_type, str(cells_change_cnt) + "_cells_" + str(charts_per_type) + "_charts" ".csv")
    
    starting_chart_type_idx = args.starting_chart_type_idx
    if starting_chart_type_idx == 0:
        with open(file_to_save_res, 'w') as f:
            pass
    elif (starting_chart_type_idx > 0) and (not os.path.exists(file_to_save_res)):
        raise Exception("Results for initial chart-types (being skipped) not present.")

    for idx, (chart_type, chart_file_names_type) in tqdm(
    enumerate(islice(zip(chart_dir_names_type_list, chart_file_names_type_prefix_list),starting_chart_type_idx, None), start = starting_chart_type_idx),
    total=chart_type_cnt - starting_chart_type_idx,  # Adjust total count to reflect the skipped elements
    desc=f"Processing, for each chart-type, No of chart image-pairs: {charts_per_type}, Cells altered: {cells_change_cnt}"
    ):

        # org_chart_type_dir_path = ChartX_data_path + '/' + chart_type + "/png"
        # org_chart_type_csv_dir_path = ChartX_data_path + '/' + chart_type + "/csv" 
        
        # no tags in current case (not required in 2 stage pipeline)
        prompt_expr_chart_type_specific = prompt_expr_chart #.replace('chart_type_tag', chart_type_tag_list[idx]).replace('chart_entity_tag', chart_entity_tag_list[idx])
        logging.info(f"final prompt template is: {prompt_expr_chart_type_specific}")
        # chart_file_list = glob.glob(modified_chart_dir_path + '/' + chart_file_names_type + '*' + '.png')
        # logging.info(f"chart file list is: {chart_file_list[:20]}")
        # ** alphabetical: 102 before 23

        modified_chart_file_list = [filename for filename in sorted(os.listdir(modified_chart_dir_path)) if (filename.startswith(chart_file_names_type) and filename.endswith(".png"))]
        # logging.info(f"list is: {modified_chart_file_list}")

        imags_processed_for_type_cnt = 0
        responses_list_for_chart_type = []
        for i in range(len(modified_chart_file_list)):

            chart_name = modified_chart_file_list[i].split('.')[0]

            org_chart_path = ChartX_imgs_from_code_run_path + '/' + chart_name + '.png' #org_chart_type_dir_path + '/' + chart_name + '.png'
            # org_chart_csv_path = org_chart_type_csv_dir_path + '/' + chart_name + '.csv'
            org_chart_csv = get_img_csv_from_ChartX_json(chart_name)

            modified_chart_path = modified_chart_dir_path + '/' + chart_name + '.png'
            change_prompt_file_path = modified_chart_dir_path + '/' + chart_name + '.txt'

            # logging.info(org_chart_path)
            # logging.info(modified_chart_path)
            # logging.info(change_prompt_path)

            if Path(org_chart_path).is_file() and Path(modified_chart_path).is_file() and Path(change_prompt_file_path).is_file():
                pass
            else:
                logging.info("ERROR, INCORRECT FILE PATH")
                continue
            
            logging.info(f"image paths are - image_1: {org_chart_path}, image_2: {modified_chart_path}")

            image_1 = Image.open(org_chart_path)
            image_1_csv = org_chart_csv #"\n"
            # with open(org_chart_csv_path, mode ='r') as file:    
            #     csvFile = csv.reader(file)
            #     for lines in csvFile:
            #         # logging.info(lines)
            #         image_1_csv += get_string_from_csv_list(lines) + '\n'
            # logging.info(f"final csv info is: {image_1_csv}")


            image_2 = Image.open(modified_chart_path)
            with open(change_prompt_file_path) as f: #, encoding="utf-8") as f:
                change_prompt = json.load(f) #["chart_change_ground_truth"]
            # change_prompt = open(change_prompt_file_path, "r").read() 
            logging.info(f"DIFFERENCE BETWEEN CHARTS IS: {change_prompt}\n")

            images_list = None
            if ablation_type == "stitched":
                images_list = [get_concat_h(image_1, image_2)]
            elif ablation_type == "separated":
                images_list = [image_1, image_2]
            else:
                raise Exception("Incorrect ablation type selected")

            response_list = get_response_ablation(chart_name, change_prompt, prompt_expr_chart_type_specific, images_list, cells_change_cnt, model_obj)

            # if model fail: dummy json -ve created (which scores zero)
            logging.info(response_list, "\n\n\n")
            responses_list_for_chart_type.append(response_list)
            # breakpoint()

            imags_processed_for_type_cnt += 1
            if imags_processed_for_type_cnt == charts_per_type:
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

    # with open(file_to_save_res, 'w', newline='\n') as file:
    #     writer = csv.writer(file, delimiter=';')
    #     field = ["chart name", "ground truth", "generated"] #"step by step", "reasoning", "multiple Qs"]
        
    #     writer.writerow(field)
    #     for idx in range(len(responses_list_list)):
    #         writer.writerow(responses_list_list[idx])

