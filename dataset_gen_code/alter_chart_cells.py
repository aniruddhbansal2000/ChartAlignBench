import os, argparse
import pandas as pd 
import numpy as np
import re

import random
import json

from io import StringIO
from collections import Counter

# import functions, to correct chart-save location in the image
from savefig_helper import update_savefig_location, remove_plt_show, generate_chart

chart_type_names_list = ['bar_chart_num', 'bar_chart', '3D-Bar', 'line_chart_num', 'line_chart', 'radar', 'rose', 'box', 'multi-axes']
vals_to_modify_cnt_list = [1, 2, 3]

json_path = os.path.join(os.getcwd(), 'ChartX_annotation.json')
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

chart_topic_list = ['Human Resources and Employee Management', 'Healthcare and Health', 'Arts and Culture', 'Business and Finance', 'Charity and Nonprofit Organizations', 'Tourism and Hospitality', 'Food and Beverage Industry', 'Technology and the Internet', 'Manufacturing and Production', 'Environment and Sustainability', 'Social Media and the Web', 'Retail and E-commerce', 'Science and Engineering', 'Law and Legal Affairs', 'Social Sciences and Humanities', 'Education and Academics', 'Energy and Utilities', 'Sports and Entertainment', 'Government and Public Policy', 'Agriculture and Food Production', 'Real Estate and Housing Market', 'Transportation and Logistics']

random.seed(42)

# return list of cells (row_no, col_no) with unique values
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

# check if string has fractions (decimal points)
def string_is_fraction(s):
    for i in s:
        if i.isdigit() or i == '.' or (i == 0 and i == '-'):
            continue
        else:
            return False
    return True

def get_random_consts_list(vals_to_modify_cnt):

    # in case cells have same attribute, better to have different constant (diverse outputs)
    random_consts_list = [0.5, 1.5, 0.3]
    return random_consts_list[:vals_to_modify_cnt]

# generate modified value for the cell
    # 0.5 or 1.5 times value of the mean of attrbiute (column) values
def get_modified_val(initial_val, col_vals_list, random_const):

    row_cnt = len(col_vals_list)

    modified_val = -1.
    # print(f"column values are: {col_vals_list}")
 
# random_const_idx = random.randint(0,1)
# random_const = (0.5 if random_const_idx == 0 else 1.5)
    # print(f"random constant is: {random_const}")

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

# if substring has only one (dis-joint) instance in a string
def single_instance_of_substring(str, sub_str):
    res = []
    while(str.find(sub_str) != -1):
 
        res.append(str.find(sub_str)) 
        str = str.replace(sub_str, "*"*len(sub_str), 1)
    return len(res) == 1 

# form exp (telling attribute change for an entity)
def get_nat_lang_expression(entity_name, attr_name, initial_val, modified_val):

    exp = str(attr_name) + ' for ' + str(entity_name) + ' modified from ' + str(initial_val) + ' to ' + str(modified_val)
    return exp

def remove_units_in_brackets(val):

    print(f"val before unit removal: {val}")

    idx_start = val.find('(')
    if val[idx_start-1] == ' ':
        idx_start -= 1          # space, remove it

    idx_end = val.find(')') + 1
    modified_val = val[:idx_start]
    if idx_end < len(val):
        modified_val += val[idx_end:]
    # modified_val = re.sub('([^>]+)', '', val)

    print(f"val after unit removal: {modified_val}")
    return modified_val

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

def get_cell_change_json(entity_name, attr_name, initial_val, modified_val):

    # int (from json-csv): numpy.int64 -> not JSON serializable 
    if isinstance(modified_val, int):
        initial_val = int(initial_val)


    cell_change_json = {}
    cell_change_json["row name"] = str(entity_name)
    cell_change_json["column name"] = str(attr_name)
    cell_change_json["value in chart 1"] = to_json_compatible(initial_val)
    cell_change_json["value in chart 2"] = to_json_compatible(modified_val)
    
    return cell_change_json

#---------------------------------------------------

# list containing cell-altered image, and change ground-truth
cell_change_json_list = []
cell_change_info_json = os.path.join(os.getcwd(), "_datasets", "ChartX_cell_edit_modified", "cell_edit_info.json") 

os.makedirs(os.path.join(os.getcwd(), "_datasets", "ChartX_cell_edit_modified"))
with open(cell_change_info_json, 'w') as f:
    pass

if __name__ == "__main__":

    # run for different cell-change countda
    for vals_to_modify_cnt in vals_to_modify_cnt_list:

        imgs_save_dir = os.path.join(os.getcwd(), "_datasets", "ChartX_cell_edit_modified", f"{vals_to_modify_cnt}_cell_edit") 
        os.makedirs(imgs_save_dir, exist_ok = True)

        # print(data[1])
        cnt = 0
        total_cnt = 0

        print(f"no of values to alter are: {vals_to_modify_cnt}")

        for data_element in data:

            if data_element['chart_type'] not in chart_type_names_list:
                continue

            img_name = data_element['imgname']
            print(f"image name is: {img_name}")

            total_cnt = total_cnt + 1


            # print(f"data element is: {data_element}")
            print(f"chart type is: {data_element['chart_type']}")
            csv_data = data_element["csv"]

            # preprocess 
            csv_data = csv_data.replace('\\t', '\t').replace('\\n', '\n')
            df = pd.read_csv(StringIO(csv_data), delimiter=' \t ')

            rows, cols = df.shape

            search_row_col_list = get_single_occur_value_row_col_list(df)
            print(f"row col list is: {search_row_col_list}")

            #** exhaustive way: modifying duplicate values also possible (if count < vals_to_modify) but complicated code
            if len(search_row_col_list) < vals_to_modify_cnt:
                print(f"Not enough value in csv to replace, skipping chart!")
                continue

            random_consts_list = get_random_consts_list(vals_to_modify_cnt)
        ###*** START HERE 
            # initial_val_list (from row-col list or directly + check index validity)
            # loop: modified val 
            # COMMON LOOP TO REPLACE?

            if img_name == "line_107":
                continue
            print(f"initial data-frame:-\n{df}")

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
                continue

            # Logging (check replaced values)
            # print(f"initial data-frame: {df}")
            # for i in range(vals_to_modify_cnt):
            #     entity_no, attr_no = search_row_col_list[i]
            #     df.iat[entity_no, attr_no] = modified_val_list[i]
            # print(f"modified data-frame:-\n{df}")
            # continue


        #* modify: initial & final data
            # Replace value in PYTHON (chart generation) code
            python_script = data_element["redrawing"]["output"]

            isReplaceSuccess = True
            for idx in range(vals_to_modify_cnt):
                initial_val_to_catch = str(initial_val_list[idx])
                modified_val_to_replace = str(modified_val_list[idx])
                print(f"values to check: {initial_val_to_catch}, replace value: {modified_val_to_replace}")
                
                if data_element in ['pie_chart', 'rings']:
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
                continue

            # edit figure save location OR remove plt.show
            if "plt.show()" in python_script:
                python_script = remove_plt_show(python_script, img_name, imgs_save_dir)
            else:
                python_script = update_savefig_location(python_script, img_name, imgs_save_dir)


            # print(f"modified python script is: {python_script}")

            # if python_script != None and 'sans-serif' not in python_script:
            #     continue

            isScriptRun = generate_chart(python_script)
            if not isScriptRun:
                continue
            all_cells_change_json = {}
            print(f"final data-frame:-\n{df}")
            
            for idx in range(vals_to_modify_cnt):


                entity_name, attr_name = entity_name_attr_name_list[idx]
                initial_val = initial_val_list[idx]
                modified_val = modified_val_list[idx]

                chart_change_json_element = get_cell_change_json(entity_name, attr_name, initial_val, modified_val) 
                print(f"json dict is: {chart_change_json_element}")
                all_cells_change_json[idx] = chart_change_json_element

            chart_change_json = {}
            chart_change_json["imgname"] = img_name
            chart_change_json["cell_change_cnt"] = vals_to_modify_cnt
            chart_change_json["cell_change_json"] = all_cells_change_json

            cell_change_json_list.append(chart_change_json)

            cnt = cnt + 1

        per_success = 100. * (cnt / total_cnt)
        print(f"charts given: {total_cnt}")
        print(f"% processed successfully: {per_success}")

        # save JSON (for 1/2/3 cell change)
    with open(cell_change_info_json, 'a') as cell_change_json_file:
        json.dump(cell_change_json_list, cell_change_json_file, indent=4)
    

