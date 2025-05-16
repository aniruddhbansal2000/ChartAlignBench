import os

import pandas as pd 
import numpy as np
import re

import random
from io import StringIO
import json

import shutil

from collections import Counter
import argparse

from copy import deepcopy

# import functions, to correct chart-save location in the image
from savefig_helper import update_savefig_location, remove_plt_show, update_fig_loc_val_in_var, generate_chart


json_path = os.path.join(os.getcwd(), 'ChartX_annotation.json')
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

dataset_dir = os.path.join(os.getcwd(), "_datasets", "ChartX_robustness_modified_2") 
os.makedirs(dataset_dir, exist_ok=True)

random.seed(42)

###--------------------------###--------------------------
# CELL CHANGE

random_consts_list = [0.5, 1.5, 0.3]

# 1. ASSIGN VALUES TO MODIFY, Parse csv (initial value -> modified value + row-column name)
###-------------------------- FUNCTION + HELPER FUNCTIONS

### return list of cells (row_no, col_no) with unique values
def get_single_occur_value_row_col_list(df):

    df_excluding_entity_col = df.loc[:, df.columns != df.columns[0]]
    values_in_df_list = df_excluding_entity_col.stack().unique()
    # print(f"values are: {values_in_df_list}")

    value_cnt_dict = Counter(values_in_df_list)
    print(f"value cnt dict is: {value_cnt_dict}")

    row_col_list = []

    for value in value_cnt_dict.keys():
        if value_cnt_dict[value] == 1:
            # single occurence, can replace
            res = df.where(df == value)        
            final_res = res.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all')
            # print(f"final response is: {final_res}")

            row_no = final_res.index.to_numpy()[0]

            column_name = final_res.columns[0]
            column_no = df.columns.get_loc(column_name)
            # print(f"row no is: {row_no}")
            # print(f"column no is: {column_no}")
            row_col_list.append([row_no, column_no])

    # jumble: 2 cells should be randomly selected from anywhere
    final_jumbled_row_col_list = sorted(row_col_list, key = lambda x: random.random())
    return final_jumbled_row_col_list

# generate modified value for the cell
def get_modified_val(initial_val, col_vals_list, random_const):

    row_cnt = len(col_vals_list)

    modified_val = -1.
    # print(f"column values are: {col_vals_list}")
 
    isDataProcessed = True
    try: 
        if isinstance(initial_val, str):
            if initial_val[-1] == '%' or initial_val[0] == '$':
                print("HAVE % or $ VALUES!") 
                modified_val = 0.   
                isInt = True

                for val in col_vals_list:
                    val_numerical = val.replace('%','').replace('$','')
                    if '.' in val_numerical:
                        isInt = False

                    val_float = eval(val_numerical)
                    modified_val += val_float / row_cnt

                if isInt:
                    modified_val = int(modified_val * random_const)
                else:
                    modified_val = round(modified_val * random_const, 3)
    
                modified_val = str(modified_val) + ('%' if initial_val.find('%') != -1 else '$')
                # print(f"modified %/$ is: {modified_val}")
        
            elif string_is_fraction(initial_val):
                print("FLOAT in form of string")
                for val in col_vals_list:
                    # if -ve value, the minus sign (sometimes bigger one) outside domain, hence replace by hyphen
                    if val[0] != '.' and val[0].isdigit() == False:
                        val = list(val)
                        val[0] = '-'
                        val = "".join(val)
                    val_float = eval(val)
                    modified_val += val_float / row_cnt

                modified_val = round(modified_val * random_const, 3)
                modified_val = str(modified_val) 
                # print(f"modified float_string is: {modified_val}")

        else:
            modified_val = np.mean(col_vals_list)
            if isinstance(initial_val, (int, np.integer)):    
                print(f"integer: adjusting it")
                modified_val = modified_val * random_const
                modified_val = int(modified_val)
            elif isinstance(initial_val, float):
                print("float in float (not string) form, adjusting it")
                modified_val = round(modified_val * random_const, 3)
    except:
        isDataProcessed = False

    if not isDataProcessed:
        return None
    return modified_val

### parse CSV and mark values (initial + calculate modified & row/column numbers)
def select_values_to_modify(img_json_info, vals_to_modify_cnt):

    # preprocess
    csv_data = img_json_info["csv"]
    csv_data = csv_data.replace('\\t', '\t').replace('\\n', '\n')
    df = pd.read_csv(StringIO(csv_data), delimiter=' \t ')
    rows, cols = df.shape

    search_row_col_list = get_single_occur_value_row_col_list(df)
    # print(f"row col list is: {search_row_col_list}")

    # currently - modify UNIQUE values (duplicate ones excluded)
    if len(search_row_col_list) < vals_to_modify_cnt:
        print(f"Not enough value in csv to replace, skipping chart!")
        return None, None, None

    initial_val_list = []
    modified_val_list = []
    entity_name_attr_name_list = []

    isCsvIndexingFine = True        # csv: columns and row attrbiutes mis-match(Eg: line_107)
    isModifiedCalcSuccess = True
    
    # calculate modified values for cells (from search_row_col_list values)
    for idx in range(vals_to_modify_cnt):

        entity_no, attr_no = search_row_col_list[idx]

        try:
            entity_name = df.iat[entity_no, 0]  
            attr_name = df.columns[attr_no]
            print(f"selected row, column is: {entity_name}, {attr_name}")
        except:
            print(f"ERROR: csv table inconsistent!")
            isCsvIndexingFine = False
            break

        entity_name_attr_name_list.append((entity_name, attr_name))

        initial_val = df.iat[entity_no, attr_no]
        # print(f"initial value is: {initial_val}")
        initial_val_list.append(initial_val)
        
        modified_val = get_modified_val(initial_val, df.loc[:, attr_name], random_consts_list[idx])
        # print(f"modified value is: {modified_val}")
        if modified_val == None:
            print("ERROR: could not process data!")
            isModifiedCalcSuccess = False
            break
        modified_val_list.append(modified_val)

    if not isModifiedCalcSuccess or not isCsvIndexingFine:
        return None, None, None

    return initial_val_list, modified_val_list, entity_name_attr_name_list

###--------------------------


# 2. MAKE CHANGES IN CODE (find INITIAL values, replace with MODIFIED values)
###-------------------------- FUNCTION + HELPER FUNCTIONS

# if substring has only one (dis-joint) instance in a string
def single_instance_of_substring(str, sub_str):
    res = []
    while(str.find(sub_str) != -1):
 
        res.append(str.find(sub_str)) 
        str = str.replace(sub_str, "*"*len(sub_str), 1)
    return len(res) == 1 

### change the code (replace initial value -> modified value)
def modify_values_in_code(img_json_info, vals_to_modify_cnt, initial_val_list, modified_val_list):

    python_script = img_json_info["redrawing"]["output"]

    # Replace value in PYTHON (chart generation) code

    isReplaceSuccess = True
    for idx in range(vals_to_modify_cnt):
        initial_val_to_catch = str(initial_val_list[idx])
        modified_val_to_replace = str(modified_val_list[idx])
        print(f"values to check: {initial_val_to_catch}, replace value: {modified_val_to_replace}")
        
        if img_json_info in ['pie_chart', 'rings']:
            initial_val_to_catch = initial_val_to_catch.replace('%', '')
            modified_val_to_replace = modified_val_to_replace.replace('%', '')

        # replace value in string
        if single_instance_of_substring(python_script, initial_val_to_catch):
            print(f"Single instance: Replacing initial value")
            python_script = python_script.replace(initial_val_to_catch , modified_val_to_replace)
        else:
            print(f"Not single instance, cannot replace value")
            isReplaceSuccess = False
            break

    if not isReplaceSuccess:
        return None
    return python_script

###-------------------------- 


# 3. BUILD JSON (initial values, modified values, row/column ie entity/attribute names)
###-------------------------- FUNCTION + HELPER FUNCTIONS


def to_json_compatible(val):
    if isinstance(val, np.integer):
        return int(val)
    elif isinstance(val, np.floating):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.bool_)):
        return bool(val)
    elif isinstance(val, (str, int, float, bool)) or val is None:
        return val
    else:
        raise TypeError(f"Unsupported type: {type(val)}")

def get_chart_change_json(entity_name, attr_name, initial_val, modified_val):

    # print(f"initial value type is: {type(initial_val)}")
    # print(f"modified value type is: {type(modified_val)}")

    # int (from json-csv): numpy.int64 -> not JSON serializable 
    if isinstance(modified_val, int):
        initial_val = int(initial_val)

    # # remove units (like million, billion etc: placed in brackets)
    # if '(' in attr_name:
    #     attr_name = remove_units_in_brackets(attr_name)

    chart_change_json = {}
    chart_change_json["row name"] = str(entity_name)    # num values (like years)
    chart_change_json["column name"] = str(attr_name)
    chart_change_json["value in chart 1"] = to_json_compatible(initial_val)
    chart_change_json["value in chart 2"] = to_json_compatible(modified_val)

    return chart_change_json

### build json (cell -> row/column name, initial value & modified value)
def get_cell_change_json(vals_to_modify_cnt, initial_val_list, modified_val_list, entity_name_attr_name_list):
    cell_change_json_dict = {}
    
    for idx in range(vals_to_modify_cnt):

        # extracted values
        # entity_no, attr_no = search_row_col_list[idx]
        # print(f"entity_index: {entity_no}, attr_index: {attr_no}")
        # entity_name = df.iat[entity_no, 0]  
        # attr_name = df.columns[attr_no]

        entity_name, attr_name = entity_name_attr_name_list[idx]
        initial_val = initial_val_list[idx]
        modified_val = modified_val_list[idx]

        chart_change_json_element = get_chart_change_json(entity_name, attr_name, initial_val, modified_val) 
        print(f"json dict is: {chart_change_json_element}")
        cell_change_json_dict[idx] = chart_change_json_element
        # chart_change_json_set_to_json = jsonpickle.encode(chart_change_json_set)

    return cell_change_json_dict

###-------------------------- 


### ALTER CELLS in the chart code
def alter_code_cell_change(img_json_info, vals_to_modify_cnt):

    # print(f"values to modify:{vals_to_modify_cnt}")
    # breakpoint()

    img_name = img_json_info['imgname']
    if img_name == "line_107": # columns and row attrbiutes mis-match
        return None, None

    # 1. ASSIGN VALUES TO MODIFY, Parse csv (initial value -> modified value + row-column name)
    initial_val_list, modified_val_list, entity_name_attr_name_list = select_values_to_modify(img_json_info, vals_to_modify_cnt)
    if not initial_val_list: # value selection failed, cell-change can't be done
        return None, None 

    # 2. MAKE CHANGES IN CODE (find INITIAL values, replace with MODIFIED values)
    modified_python_script = modify_values_in_code(img_json_info, vals_to_modify_cnt, initial_val_list, modified_val_list)
    if not modified_python_script: # code-change failed, cell-change can't be done
        return None, None

    # 3. BUILD JSON (initial values, modified values, row/column ie entity/attribute names)
    cell_change_json_dict = get_cell_change_json(vals_to_modify_cnt, initial_val_list, modified_val_list, entity_name_attr_name_list)

    return modified_python_script, cell_change_json_dict

###--------------------------###--------------------------



###--------------------------###--------------------------
# ALTER LEGEND

legend_pos_options_list = ['upper right', 'upper left', 'lower left', 'lower right', 'center']

### replace legend position in given code 
def set_legend_pos_attribute_in_code(legend_pos_pattern, grounded_code, legend_pos_to_set, chart_type):

    # defined outside function, hence better as list (reference-wise)
    legend_pos_replacement_pair = []

    def legend_pos_replacement_function(match):

        legend_pos_exp = match.group(0)
        print(f"expresion is: {legend_pos_exp}")

        initial_legend_pos = None
        if chart_type == 'rose':
            initial_legend_pos = "upper right"
        else:
            if "'" in legend_pos_exp:   # position enclosed in single quotes
                initial_legend_pos = legend_pos_exp.split("'")[1]
            else:                       # position enclosed in double quotes
                initial_legend_pos = legend_pos_exp.split("\"")[1]
        print(initial_legend_pos)

        modified_legend_pos = legend_pos_to_set
        print(modified_legend_pos)

        legend_pos_replacement_pair.append((initial_legend_pos, modified_legend_pos))
    
        return f".legend(loc=\'{modified_legend_pos}\')"

    # Perform the substitution and track replacements
    updated_code = re.sub(legend_pos_pattern, legend_pos_replacement_function, grounded_code)
    
    return updated_code, legend_pos_replacement_pair


### code - search for legend & replace
def alter_chart_legend_pos(grounded_code, legend_pos_to_set, chart_type):

    updated_code, legend_pos_replacement_pair = None, None

    #######################################
    # legend position change in code

    # only 'loc' attribute | comma separated other attributes not taken into account
    """
        ['\"] -> single or double quote (as double quote outside as well, put escape-char)
        [ \w] -> space (' ') or meta-character (english chars, digits)
    """

    if chart_type == 'radar':
        legend_pos_pattern_radar_chart_type = r"\.legend\s*\(\s*(?:[^()]*?,\s*)*?loc\s*=\s*['\"][ \w]+['\"](?:\s*,[^()]*?)*\s*\)"
        legend_pos_check_result = re.search(legend_pos_pattern_radar_chart_type, grounded_code)

        if legend_pos_check_result and ("best" not in legend_pos_check_result.group(0)) and ("bbox_to_anchor" not in legend_pos_check_result.group(0)): # "best": can't determine position, hence skip the legend position        
            print(f"found legend to replace for: {fig_name}")
            updated_code, legend_pos_replacement_pair = set_legend_pos_attribute_in_code(legend_pos_pattern_radar_chart_type, grounded_code, legend_pos_to_set, chart_type) 
            
    elif chart_type == 'rose':
        # detect of form: legend(bbox_to_anchor=(<x-value>, <y-value>)).
        legend_pos_pattern_rose_chart_type = r"\.legend\s*\(\s*bbox_to_anchor\s*=\s*\(\s*[^(),\s]+\s*,\s*[^(),\s]+\s*\)\s*\)"
        legend_pos_check_result = re.search(legend_pos_pattern_rose_chart_type, grounded_code)

        if legend_pos_check_result:
            # if <x-value> and <y-value> greater than 1, then chart on top-right.
            legend_pos_exp = legend_pos_check_result.group(0)
            bbox_anchor_value_str = legend_pos_exp.split('(')[2].split(')')[0]
            
            x_val_str, y_val_str = bbox_anchor_value_str.split(',')
            x_val, y_val = eval(x_val_str), eval(y_val_str)
            if x_val > 1.0 and y_val > 1.0:
                # breakpoint()
                print(f"found legend to replace for: {fig_name}")
                updated_code, legend_pos_replacement_pair = set_legend_pos_attribute_in_code(legend_pos_pattern_rose_chart_type, grounded_code, legend_pos_to_set, chart_type) 
                
    else:
        legend_pos_pattern = r"\.legend\s*\(\s*loc\s*=\s*['\"][ \w]+['\"]\s*\)"
        legend_pos_check_result = re.search(legend_pos_pattern, grounded_code)

        if legend_pos_check_result and ("best" not in legend_pos_check_result.group(0)): # "best": can't determine position, hence skip the legend position        
            updated_code, legend_pos_replacement_pair = set_legend_pos_attribute_in_code(legend_pos_pattern, grounded_code, legend_pos_to_set, chart_type) 

    return updated_code

###--------------------------###--------------------------


###--------------------------###--------------------------
# ALTER COLOR

# pattern match color='x'
color_single_char_pattern = r"color\s?=\s?'([^'])'"
# pattern match color='#xxxxxx'
color_hex_pattern = r"color\s?=\s?'(#[0-9a-fA-F]{6})'"
# pattern match color=letters
color_letters_pattern = r"color\s?=\s?'([a-zA-Z]+)'"
# pattern match color=(x, x, x)
color_rgb_pattern = r"color\s*=\s*\((\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*)\)" # r"color\s?=\s?\(([^)]+)\)"

color_list_pattern = r"color[s]?\s*=\s*\[[^\]]*\]"

### Generate set of 5, each contains K colors (K = number of colors in the chart)
matplotlib_default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def get_color_position_options_list(chart_csv_data_str, chart_type, robustness_set_length):

    csv_data = chart_csv_data_str.replace(' \\t ', ' \t ').replace(' \\n ', ' \n ')
    df = pd.read_csv(StringIO(csv_data), delimiter=' \t ')

    colColored = chart_type in ["bar_chart", "bar_chart_num", "line_chart", "line_chart_num", "3D-Bar"]
    rowColored = chart_type in ["box", "radar", "rose"]

    color_cnt = None
    if rowColored:
        color_cnt = len(np.array(df.values[:, 0])) # df.index - 0, 1, 2... (unless custom defined)
    if colColored:
        color_cnt = len(np.array(df.columns)[1:])

    color_position_options_list = []

    try:
        for i in range(robustness_set_length):
            color_set = random.sample(matplotlib_default_colors, color_cnt)
            color_position_options_list.append(color_set)
    except:
        print(f"Error: mismatch in color count vs there in code")
        color_position_options_list = None

    return color_position_options_list

### replace COLOR position in given code (color in chart-code mentioned separately) | chart_type in ["bar_chart", "bar_chart_num", "line_chart", "line_chart_num"]
def set_colors_attribute_in_code(color_pattern, grounded_code, color_to_set_list):

    isColorChangeSuccess = True
    cur_idx = 0

    def color_set_function(match):
        nonlocal cur_idx  # Declare cur_idx as nonlocal to modify the outer variable
        initial_color_exp = match.group(0)  # Full match (e.g., rgb(255, 87, 51))
        # rgb_values = match.groups()      # Tuple of RGB values (e.g., ('255', '87', '51'))
        
        initial_color = initial_color_exp.replace(' ', '').replace('color=','').replace("'", "")
        
        # set color value
        set_color = None
        try:
            set_color = color_to_set_list[cur_idx]
            # print(f"current idx: {cur_idx}")
    
            cur_idx = cur_idx + 1   # next element (set as color next time)
            # if not define non-local cur_idx, here python assumed local variable 

            # Return the modified color string
            return f"color=\'{set_color}\'"
        except:
            print(f"Error: more colors in chart code apart from core object (i.e. bars/line etc) colors")
            isColorChangeSuccess = False
            return "0"

    # Perform the substitution and track replacements
    updated_code = re.sub(color_pattern, color_set_function, grounded_code)

    # print("\n\n", updated_code, "\n\n")
    # breakpoint()
    
    if not isColorChangeSuccess:
        return None
    return updated_code


### replace COLOR position in given code (color mentioned as list) | chart_type in ["rose", "radar", "box", "3D-Bar"]
def set_colors_list_in_code(initial_color_list, grounded_code, isPlural, color_to_set_list):

    set_color_list_exp = "color = " + str(color_to_set_list)
    if isPlural:
        set_color_list_exp = set_color_list_exp.replace("color", "colors")       

    """
    # color list exp to replace in code
    set_color_list_exp = "color = ["
    if isPlural:
        set_color_list_exp = set_color_list_exp.replace("color", "colors")

    num_colors = len(initial_color_list)
    for idx in range(num_colors):

        initial_color = initial_color_list[idx]
        modified_color = color_to_set_list[idx]

        set_color_list_exp += "\'" + modified_color + "\'"
        if idx == (num_colors-1):
            set_color_list_exp += "]"
        else:
            set_color_list_exp += ", " 
    """
    # replace color list exp in grounded code
    updated_code = re.sub(color_list_pattern, set_color_list_exp, grounded_code)

    return updated_code



### code - search for color & replace
def alter_chart_color(grounded_code, color_to_set_list, chart_type):

    #######################################
    # color change in the code (string form)
    updated_code = None

    if chart_type in ["bar_chart", "bar_chart_num", "line_chart", "line_chart_num"]:

    # bar/bar-num/line/line-num: manual assignment for each attribite (else automatic) - no loop used
        # manually check for color types + replace accordingly
        # Assuming 'in a scipt' - all color patterns same | mixed (different color pattern exp) not handled
        if re.search(color_single_char_pattern, grounded_code):
            updated_code = set_colors_attribute_in_code(color_single_char_pattern, grounded_code, color_to_set_list)
        elif re.search(color_hex_pattern, grounded_code):
            updated_code = set_colors_attribute_in_code(color_hex_pattern, grounded_code, color_to_set_list)
        elif re.search(color_letters_pattern, grounded_code):
            updated_code = set_colors_attribute_in_code(color_letters_pattern, grounded_code, color_to_set_list)
        elif re.search(color_rgb_pattern, grounded_code):
            updated_code = set_colors_attribute_in_code(color_rgb_pattern, grounded_code, color_to_set_list)
        else:
            print("No color parameters found.")

    elif chart_type in ["rose", "radar", "box", "3D-Bar"]:
        # detect list, extract + replace it
        re_color_list_check_result = re.search(color_list_pattern, grounded_code)

        # if 'range' then color list not explicitly defined, instead sequenced from a set
        if re_color_list_check_result and ("range" not in re_color_list_check_result.group(0)):
            color_list_exp = re_color_list_check_result.group(0)
            isPlural = False
            if "colors" in color_list_exp: # color vs color(s)
                isPlural = True
            list_st = color_list_exp.find('[')
            list_end = color_list_exp.find(']')
            color_list = color_list_exp[list_st+1:list_end].split(',')

            # remove blank spaces, quotes   
            color_list = [color_element.replace(" ", "").replace("\'", "").replace('\"', '').replace('\n', '') for color_element in color_list]
            print(f"color list is: {color_list}")
            updated_code = set_colors_list_in_code(color_list, grounded_code, isPlural, color_to_set_list)
        else:
            print(f"No color list (default or Color-set)")

    return updated_code


###--------------------------###--------------------------



###--------------------------###--------------------------
# ALTER TEXT-STYLE (FONT)

chart_regions_with_text_list = ["chart title", "chart legend", "chart axes labels", "chart axes ticks"]

default_text_style_val_json = {
    "chart title": {
        "size": 14,
        "weight": "normal",
        "fontfamily": "sans-serif"
    },
    "chart legend": {
        "size": 12,
        "weight": "normal",
        "fontfamily": "sans-serif"
    },
    "chart axes labels": {
        "size": 12,
        "weight": "normal",
        "fontfamily": "sans-serif"
    },
    "chart axes ticks": {
        "size": 12,
        "weight": "normal",
        "fontfamily": "sans-serif"
    }
}

def get_text_style_options_list():

    ## 5 text-styles
    fontfamily_options_list = ['sans-serif', 'serif', 'cursive', 'fantasy', 'monospace']
    font_family_chosen = random.choice(fontfamily_options_list)
    
    # 1 -> default
    text_style_1 = deepcopy(default_text_style_val_json)

    # 2 -> bold
    text_style_2 = deepcopy(default_text_style_val_json)
    text_style_2["chart title"]["weight"] = "bold"
    text_style_2["chart axes labels"]["weight"] = "bold"
    
    # 3 -> smaller size (larger size may obstruct chart elements)
    text_style_3 = deepcopy(default_text_style_val_json)
    for text_section in chart_regions_with_text_list:
        text_style_3[text_section]["size"] = 8   

    # 4 -> font family
    text_style_4 = deepcopy(default_text_style_val_json)
    for text_section in chart_regions_with_text_list:
        text_style_4[text_section]["fontfamily"] = font_family_chosen

    # 5 -> combine 3, 4, 5
    text_style_5 = deepcopy(default_text_style_val_json)
    text_style_5["chart title"]["weight"] = "bold"
    text_style_5["chart axes labels"]["weight"] = "bold"
    for text_section in chart_regions_with_text_list:
        text_style_5[text_section]["size"] = 8
        text_style_5[text_section]["fontfamily"] = font_family_chosen

    return [text_style_1, text_style_2, text_style_3, text_style_4, text_style_5]

def get_code_to_alter_text_style_from_json(text_style_to_set_json):
    
    rc_params_text_style_size = f"""
plt.rc('axes', titlesize={text_style_to_set_json['chart title']['size']}) # title
plt.rc('legend', fontsize={text_style_to_set_json['chart legend']['size']}) # legend
plt.rc('axes', labelsize={text_style_to_set_json['chart axes labels']['size']}) # x and y labels
plt.rc('xtick', labelsize={text_style_to_set_json['chart axes ticks']['size']}) # x tick labels
plt.rc('ytick', labelsize={text_style_to_set_json['chart axes ticks']['size']}) # y tick labels\n
"""

    rc_params_text_style_weight = f"""
plt.rc('axes', titleweight='{text_style_to_set_json['chart title']['weight']}') # title
plt.rc('axes', labelweight='{text_style_to_set_json['chart axes labels']['weight']}') # x and y labels\n
"""

    rc_params_text_style_family = f"""
plt.rcParams['font.family'] = '{text_style_to_set_json['chart title']['fontfamily']}'\n
"""

    code_to_alter_text_style = rc_params_text_style_size + rc_params_text_style_weight + rc_params_text_style_family
    return code_to_alter_text_style


# check if custom-font defined in chart-code
def check_custom_font(grounded_code):

    # "fontsize" or "size" (size overlap with unrelated usage in script)
    if "fontsize" in grounded_code:
        return True
    if "fontweight" in grounded_code or "weight" in grounded_code:
        return True
    if "fontfamily" in grounded_code or "family" in grounded_code or "fontname" in grounded_code:
        return True

    if "rotation" in grounded_code:
        return True

    return False


### code - search for FONT & REPLACE
def alter_chart_text_style(grounded_code, text_style_to_set_json, chart_type):

    updated_code = None

    # DEFAULT: pattern (rcParams) - to change + add in the script

    isCustomFontDefined = check_custom_font(grounded_code)
    if isCustomFontDefined: # changes ove-ride rcParams change
        return None

    # IF legend in best position - may shift due to font change
    if chart_type not in ["3D-Bar", "box"]: #3D, box charts don't have legends
        # more general pattern
        legend_pos_pattern = r"\.legend\s*\(\s*(?:[^()]*?\([^()]*?\)[^()]*?,\s*)*?loc\s*=\s*['\"][ \w]+['\"](?:\s*,[^()]*?)*\s*\)"
        # legend_pos_pattern = r"\.legend\s*\(\s*loc\s*=\s*['\"][ \w]+['\"]\s*\)"
        legend_pos_check_result = re.search(legend_pos_pattern, grounded_code)
        if not (legend_pos_check_result and ("best" not in legend_pos_check_result.group(0))): # "best": can't determine position, hence skip the legend position        
            return None
    

    plotlib_line_to_find = "import matplotlib.pyplot as plt\n"
    plotlib_line_len = len(plotlib_line_to_find)
    plotlib_line_pos = grounded_code.find(plotlib_line_to_find) + plotlib_line_len
    # breakpoint()

    # font attributes (initial: default values | modified: randomly generated)
    rc_font_attr_code_to_insert = get_code_to_alter_text_style_from_json(text_style_to_set_json)   

    updated_code = grounded_code[:plotlib_line_pos] + rc_font_attr_code_to_insert + grounded_code[plotlib_line_pos:]
    # print(updated_code)    
    # breakpoint()

    return updated_code


###--------------------------###--------------------------



###--------------------------###--------------------------###--------------------------
# ATTRIBUTE CHANGE (Generate robustness set)

### correct save image location & run the script
def set_save_loc_in_code_and_run(python_script, img_file_name, imgs_save_dir):
    # edit figure save location OR remove plt.show
    if "plt.show()" in python_script:
        python_script = remove_plt_show(python_script, img_file_name, imgs_save_dir)
    else:
        python_script = update_savefig_location(python_script, img_file_name, imgs_save_dir)

    # if python_script != None and 'sans-serif' not in python_script:
    #     continue

    # print("\n\n", python_script, "\n\n")
    # breakpoint()
    
    isScriptRun = generate_chart(python_script)
    # try:
    #     exec(python_script)
    #     print("ran the chart generation script")
    # except:
    #     print("ERROR, could not run the chart generation script..")
    #     isScriptRun = False
    #     # print(python_script)
    #     # breakpoint()

    return isScriptRun

def generate_robustness_set(attribute_altered, img_json_info, cell_change_modified_chart_code, cell_change_json, dir_name):
   
    attribute_value_options_list = None
    alter_chart_attribute_function = None
    if attribute_altered == "legend":
        attribute_value_options_list = legend_pos_options_list
        alter_chart_attribute_function = alter_chart_legend_pos

    elif attribute_altered == "color":
        # attribute_value_options_list -> assign on basis of colors in chart
        attribute_value_options_list = get_color_position_options_list(img_json_info["csv"], img_json_info["chart_type"], 5)
        if not attribute_value_options_list:
            return None
        print(attribute_value_options_list)
        alter_chart_attribute_function = alter_chart_color
        # breakpoint()

    elif attribute_altered == "text_style":
        attribute_value_options_list = get_text_style_options_list()
        print(f"text style options list is: {attribute_value_options_list}")
        alter_chart_attribute_function = alter_chart_text_style

    # JSON for robustness set
    robustness_set_json = {}
    robustness_set_json["imgname"] = img_json_info["imgname"]
    # robustness_set_json["attribute_altered"] = attribute_altered
    robustness_set_json["cell_change_JSON"] = cell_change_json
    robustness_set_json["chart_pair_imgs_set"] = {}

    imgname = img_json_info["imgname"]
    cell_change_initial_chart_code = img_json_info["redrawing"]["output"]

    for idx in range(len(attribute_value_options_list)):
        attribute_value_to_set = attribute_value_options_list[idx]
        attribute_altered_cell_change_initial_chart_code = alter_chart_attribute_function(cell_change_initial_chart_code, attribute_value_to_set, chart_type)
        attribute_altered_cell_change_modified_chart_code = alter_chart_attribute_function(cell_change_modified_chart_code, attribute_value_to_set, chart_type)

        # print("\n\n", attribute_altered_cell_change_initial_chart_code, "\n\n")
        # print("\n\n", attribute_altered_cell_change_modified_chart_code, "\n\n")
        # breakpoint()

        imgs_save_subdir = dir_name + "/" + imgname
        os.makedirs(imgs_save_subdir, exist_ok=True)
        
        # name of image pairs (added to JSON)
        image_pairs_name = f"{str(idx)}_initial_chart", f"{str(idx)}_modified_chart"
        # run chart code 
        isChartPairGenerationSuccess = False
        # code for both charts generated (check it isn't NULL)
        if attribute_altered_cell_change_initial_chart_code and attribute_altered_cell_change_modified_chart_code:
            # print(f"image names: {image_pairs_name}")
            # breakpoint()
            isChartPairGenerationSuccess = set_save_loc_in_code_and_run(attribute_altered_cell_change_initial_chart_code, image_pairs_name[0], imgs_save_subdir) and set_save_loc_in_code_and_run(attribute_altered_cell_change_modified_chart_code, image_pairs_name[1], imgs_save_subdir)

        # add image-pair info to JSON (if run for both images a success)
        if isChartPairGenerationSuccess:
            robustness_set_json["chart_pair_imgs_set"][idx] = {"initial_chart_name": image_pairs_name[0], "modified_chart_name": image_pairs_name[1], "attribute_value": attribute_value_to_set}
    
    if not robustness_set_json["chart_pair_imgs_set"]:

        print(robustness_set_json["chart_pair_imgs_set"])
        # breakpoint()        
        shutil.rmtree(imgs_save_subdir)  # no pair generated, ie no robustness set for the image
        return None                 # no JSON to return 
    
    return robustness_set_json

###--------------------------###--------------------------###--------------------------

robustness_set_json_list = []
json_name = f"{dataset_dir}/robustness_info.json"
with open(json_name, 'w') as f: # clean older content
    pass



###--------------------------     MAIN CODE
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--cells_change_cnt", type=int, default=1, help="no of cells altered in chart")
    parser.add_argument('-a', "--attribute_altered", type=str, default="color", help="plot attribute argument altered for robustness")
    
    for idx_cell_vals_to_modify, vals_to_modify_cnt in enumerate([1, 2, 3]):
        for idx_attr_change, attr_change in enumerate(["color", "legend", "text_style"]):

    # vals_to_modify_cnt = args.cells_change_cnt          # 1 or 2 or 3
    # attr_change = args.attribute_altered                # legend or color or text_style

            dir_name = f"{dataset_dir}/imgs_{vals_to_modify_cnt}_cell_change_{attr_change}_altered"

            fraction_of = None
            total_cnt = 0
            success_cnt = 0

            for img_json_info in data:
                # cell_change_initial_chart_code = img_json_info["redrawing"]["output"]
                chart_type = img_json_info['chart_type']
                fig_name = img_json_info['imgname']

                if chart_type not in ["bar_chart", "bar_chart_num", "line_chart", "line_chart_num", "3D-Bar", "box", "radar", "rose", "multi-axes"]:
                    continue    

                if attr_change == "legend" and chart_type in ["3D-Bar", "box", "multi-axes"]: #3D, box charts don't have legends | multi-axes: multiple legends
                    continue

                total_cnt += 1
                # fraction_of = 20
                # if total_cnt % fraction_of:
                #     continue
                print(f"IMAGE TAG - image is: {fig_name}")

                # 1. CELL CHANGE
                cell_change_modified_chart_code, cell_change_json = alter_code_cell_change(img_json_info, vals_to_modify_cnt)

                # print(f"modified code: {cell_change_modified_chart_code}")
                print(f"\ncell change json: {cell_change_json}\n")
                if not cell_change_modified_chart_code:
                    continue

                # 2. ATTRIBUTE CHANGE (on code pairs)
                
                robustness_set_json = generate_robustness_set(attr_change, img_json_info, cell_change_modified_chart_code, cell_change_json, dir_name)
                if robustness_set_json:
                    robustness_set_json["cell_change_cnt"] = vals_to_modify_cnt
                    robustness_set_json["attr_change_type"] = attr_change
                    
                    robustness_set_json_list.append(robustness_set_json)
                    success_cnt += 1
                # breakpoint()

            """
            - attribute (legend/color/text style)
                values to modify (set of X, uniform for all charts)

            - chart pairs - make changes
                + JSON generation

            * choosing the set (color - which colors, font - family/size/breadth which ones alter?)
            * JSON prep - how assign?

            """

            if fraction_of:
                print(f"FRACTION OF CHARTS SEEN!")
                total_cnt = total_cnt / fraction_of

            print(f"total charts: {total_cnt}")
            print(f"Success rate: {(100. * success_cnt)/total_cnt}")


    print(robustness_set_json_list)
    with open(json_name, 'a') as attr_change_json_file:
        json.dump(robustness_set_json_list, attr_change_json_file, indent=4)

    ###-------------------------------

