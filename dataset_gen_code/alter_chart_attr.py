import json, re, random, os

from io import StringIO
import numpy as np
import pandas as pd

import matplotlib

from copy import deepcopy # make font json (dict) copy

# import functions, to correct chart-save location in the image
from savefig_helper import update_savefig_location, remove_plt_show, generate_chart

chart_type_names_list = ['bar_chart_num', 'bar_chart', '3D-Bar', 'line_chart_num', 'line_chart', 'radar', 'rose', 'box', 'multi-axes']

# save image (attributes json) 
initial_imgs_dir = os.path.join(os.getcwd(), "_datasets", "ChartX_imgs_from_code_run") 

# modified images + image-json + changes
imgs_save_dir = os.path.join(os.getcwd(), "_datasets", "ChartX_attr_edit_modified") 
if not os.path.exists(imgs_save_dir):
    os.makedirs(imgs_save_dir)

### JSON (containing attribute information)
attribute_change_info_json = os.path.join(imgs_save_dir, "attribute_edit_info.json") #"_combined_attr_change_labels_partial_change_rcparam_defauls.json"

ChartX_json_path = os.path.join(os.getcwd(), "_datasets", 'ChartX_annotation.json')
with open(ChartX_json_path, encoding="utf-8") as f:
    data = json.load(f)

random.seed(42)

# pattern match color='x'
color_single_char_pattern = r"color\s?=\s?'([^'])'"
# pattern match color='#xxxxxx'
color_hex_pattern = r"color\s?=\s?'(#[0-9a-fA-F]{6})'"
# pattern match color=letters
color_letters_pattern = r"color\s?=\s?'([a-zA-Z]+)'"
# pattern match color=(x, x, x)
color_rgb_pattern = r"color\s*=\s*\((\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*)\)" # r"color\s?=\s?\(([^)]+)\)"

# common pattern, check multi-axes chart color duplicacy
color_common_pattern = r"color\s?=\s?'([^']+)'"

color_list_pattern = r"color[s]?\s*=\s*\[[^\]]*\]"

# pattern match rotation=num
rotation_pattern = r"rotation\s?=\s?([0-9]+)"

# pattern match figsize=(x, y)
figsize_pattern = r"figsize\s?=\s?\(([^)]+)\)"

plotlib_default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    
def randomize_rotation(rotation):
    degrees = random.randint(0, 90) #[0, 45]
    print(f"rotation is: {degrees} deg")
    return degrees
    # return random.choice(degrees)

def randomize_figsize(figsize):
    return tuple(random.randint(1, 10) for _ in range(2))

data_pattern = r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\[.*\]\s*'

bar_pattern = r".bar\("

def bar_to_barh(code):
    width_pattern = r"width\s?=\s?([A-Za-z0-9]+)"
    width_to_height = lambda x: f"height={x.group(1)}"
    code = re.sub(width_pattern, width_to_height, code)

    bottom_pattern = r"bottom\s?=\s?([A-Za-z0-9]+)"
    bottom_to_left = lambda x: f"left={x.group(1)}"
    code = re.sub(bottom_pattern, bottom_to_left, code)

    xticks_pattern = "xticks"
    xticks_to_yticks = lambda x: "yticks"
    code = re.sub(xticks_pattern, xticks_to_yticks, code)

    x_label_pattern = "xlabel"
    y_label_pattern = "ylabel"

    # change xlabel to zlabel first
    code = re.sub(x_label_pattern, "zlabel", code)
    code = re.sub(y_label_pattern, "xlabel", code)
    code = re.sub("zlabel", "ylabel", code)

    return re.sub(bar_pattern, ".barh(", code)

def flip_axis(grounded_code):

    modified_ground_code = grounded_code
    # bar chart
    if re.search(bar_pattern, grounded_code):
        width_pattern = r"width\s?=\s?([A-Za-z0-9]+)"
        width_to_height = lambda x: f"height={x.group(1)}"
        modified_ground_code = re.sub(width_pattern, width_to_height, modified_ground_code)

        bottom_pattern = r"bottom\s?=\s?([A-Za-z0-9]+)"
        bottom_to_left = lambda x: f"left={x.group(1)}"
        modified_ground_code = re.sub(bottom_pattern, bottom_to_left, modified_ground_code)

        modified_ground_code = re.sub(bar_pattern, ".barh(", modified_ground_code)


    # Regex patterns to swap x and y axes
    modified_ground_code = re.sub(r'xlabel', r'x_label', modified_ground_code) # temp label
    modified_ground_code = re.sub(r'ylabel', r'xlabel', modified_ground_code)
    modified_ground_code = re.sub(r'x_label', r'ylabel', modified_ground_code)

    # Regex patterns to swap x and y ticks (axis markings) | cases: set_xticks, .xticks, .xtick_labels
    modified_ground_code = re.sub(r'xtick', r'ytick', modified_ground_code) # temp label
    # modified_ground_code = re.sub(r'yticks', r'xticks', modified_ground_code)
    # modified_ground_code = re.sub(r'x_ticks', r'yticks', modified_ground_code)

    # Regex patterns to swap x and y 
    modified_ground_code = re.sub(r'xaxis.set_major_formatter', r'x_axis.set_major_formatter', modified_ground_code) # temp label
    modified_ground_code = re.sub(r'yaxis.set_major_formatter', r'xaxis.set_major_formatter', modified_ground_code)
    modified_ground_code = re.sub(r'x_axis.set_major_formatter', r'yaxis.set_major_formatter', modified_ground_code)

    # Swap the x_data and y_data in plot function
    modified_ground_code = re.sub(r'plot\((.*?), (.*?),', r'plot(\2, \1,', modified_ground_code)

    # Modify axis limits to flip x and y axis ranges
    modified_ground_code = re.sub(r'axis\(\[(.*?), (.*?), (.*?), (.*?)\]\)', r'axis([\3, \4, \1, \2])', modified_ground_code)

    # Swap figsize resolution
    # modified_ground_code = re.sub(r'figsize=\((\d+), (\d+)\)', r'figsize=(\2, \1)', modified_ground_code)

    return modified_ground_code


##### COLOR ALTER |START|

def randomize_color():
    # Generate a random color for demonstration (you can use your own logic)
    # return f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'
    return  "#%06x" % random.randint(0, 0xFFFFFF)

def modify_color_or_donot_randomized_decision():
    modified_color = randomize_color()
    is_color_modified = random.choice([True, False])

    return is_color_modified, (modified_color if is_color_modified else None)

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

def color_single_char_to_rgb(color):
    color = color.lower()
    
    # Convert color names to hex
    web_colors = {
        'b': '0, 0, 1',        # blue 
        'g': '0, 0.5, 0',      # green 
        'r': '1, 0, 0',        # red 
        'c': '0, 0.75, 0.75',  # cyan 
        'm': '0.75, 0, 0.75',  # magenta 
        'y': '0.75, 0.75, 0',  # yellow 
        'k': '0, 0, 0',        # black 
        'w': '1, 1, 1',        # white       
    }
    
    if color in web_colors:
        return web_colors[color]
    
    raise ValueError("Unknown color single char name")

def rgb_string_to_hex(rgb_string):
    r, g, b = map(float, rgb_string.split(','))
    return rgb_to_hex(r*255, g*255, b*255)

def convert_color_to_hex_format(color):
    print(f"color to convert to hex is: {color}")
    if '#' in color:        # already hex
        return color    
    elif ',' in color:      # rgb (has commas)
        return rgb_string_to_hex(color)
    elif len(color) == 1:   # single char
        return rgb_string_to_hex(color_single_char_to_rgb(color))
    else:                   # string color
        hex_color = None
        try:
            hex_color = matplotlib.colors.cnames[color.lower()]
        except:
            print(f"ERROR! Not a color, incorrectly detected")
            # breakpoint()
            return None
        return hex_color

def replace_colors_attribute_in_code(color_pattern, grounded_code):
    # Initialize the list to store the mapping of initial to random colors
    color_replacement_list = []

    def color_replacement_function(match):
        initial_color_exp = match.group(0)  # Full match (e.g., rgb(255, 87, 51))
        # rgb_values = match.groups()      # Tuple of RGB values (e.g., ('255', '87', '51'))
        print(initial_color_exp)

        initial_color = initial_color_exp.replace(' ', '').replace('color=','').replace("'", "")
        # print(f"initial color is: {initial_color}")
        # print(f"color pattern is: {color_pattern}")

        initial_color_hex_format = convert_color_to_hex_format(initial_color)

        # Create a modified color
        is_color_modified, modified_color = modify_color_or_donot_randomized_decision()

        if is_color_modified: # color modified
            color_replacement_list.append((initial_color_hex_format, modified_color))        
            # Return the MODIFIED color string
            return f"color=\'{modified_color}\'"
        
        # color not modified
        color_replacement_list.append((initial_color_hex_format, initial_color_hex_format))  # all comparisons in hex (hence initial color of hex format)        
        # Return the SAME (INITIAL) color string
        return initial_color_exp

    # Perform the substitution and track replacements
    updated_code = re.sub(color_pattern, color_replacement_function, grounded_code)
    
    return updated_code, color_replacement_list 


def replace_colors_list_in_code(color_list, grounded_code, isPlural):

    # color list exp to replace in code
    modified_color_list_exp = "color = ["
    if isPlural:
        modified_color_list_exp = modified_color_list_exp.replace("color", "colors")

    color_replacement_list = []

    num_colors = len(color_list)
    for idx in range(num_colors):

        initial_color = color_list[idx]
        initial_color_hex_format = convert_color_to_hex_format(initial_color)
        if initial_color_hex_format == None:
            # string content incorreclty detected as color
            return None, None
        
        is_color_modified, modified_color = modify_color_or_donot_randomized_decision()
        
        if is_color_modified: # modified color
            color_replacement_list.append((initial_color_hex_format, modified_color))
            modified_color_list_exp += "\'" + modified_color + "\'"

        else: #initial color
            color_replacement_list.append((initial_color_hex_format, initial_color_hex_format)) # all comparisons in hex (hence initial color of hex format)
            modified_color_list_exp += "\'" + initial_color + "\'"                              # maintain same color

        # COLOR LIST 
        if idx == (num_colors-1):
            modified_color_list_exp += "]"
        else:
            modified_color_list_exp += ", " 


    # replace color list exp in grounded code
    updated_code = re.sub(color_list_pattern, modified_color_list_exp, grounded_code)

    return updated_code, color_replacement_list

# return color change jsons
    # color_change_label_json - {attr: {initial color: <>, modified color: <>}....}
    # color value json (for charts) - {attr: <color value>}
def get_color_change_jsons(color_replacement_list, img_json_info):

    print(color_replacement_list)

    csv_data = img_json_info["csv"].replace(' \\t ', ' \t ').replace(' \\n ', ' \n ')
    df = pd.read_csv(StringIO(csv_data), delimiter=' \t ')

    chart_type = img_json_info["chart_type"]
    colColored = chart_type in ["bar_chart", "bar_chart_num", "line_chart", "line_chart_num", "3D-Bar", "area_chart", "multi-axes"]
    rowColored = chart_type in ["box", "radar", "rose"]

    color_replacement_list = np.array(color_replacement_list)

    initial_color_arr = color_replacement_list[:, 0]
    modified_color_arr = color_replacement_list[:, 1]
    
    color_attr_arr = None
    if rowColored:
        color_attr_arr = np.array(df.values[:, 0]) # df.index - 0, 1, 2... (unless custom defined)
    if colColored:
        color_attr_arr = np.array(df.columns)[1:]

    num_colors = len(initial_color_arr)
    num_attributes = len(color_attr_arr)
    if num_colors != num_attributes:
        return None, None, None
    color_change_label_json = {}
    initial_img_colors_json = {}
    modified_img_colors_json = {}
    for idx in range(num_colors):
        attr_name = color_attr_arr[idx]
        
        # color change (GT) label
        color_change_label_json[attr_name] = {"initial value": initial_color_arr[idx], "modified value": modified_color_arr[idx]}

        # colors label for initial image
        initial_img_colors_json[attr_name] = initial_color_arr[idx]
        # colors label for modified image
        modified_img_colors_json[attr_name] = modified_color_arr[idx]

    print(f"color replacement list: {color_replacement_list}")
    print(f"initial img colors: {initial_img_colors_json}")
    print(f"modified img colors: {modified_img_colors_json}")

    return color_change_label_json, initial_img_colors_json, modified_img_colors_json

def change_chart_color(img_json_info):
    """
        randomize color
    """
    grounded_code = img_json_info["redrawing"]["output"]
    fig_name = img_json_info['imgname']
    chart_type = img_json_info['chart_type']

    #######################################
    # color change in the code (string form)
    updated_code, color_replacement_list = None, None

    # multi-axes chart: only single color ie axes label not colored as bar/line
    if chart_type == "multi-axes":
        # multiple cases, regex not working, hence manual check (images with no colored labels/axes ticks)
        # of below, 3 charts where color-replace success..
        if fig_name not in ['multi-axes_2', 'multi-axes_6', 'multi-axes_35', 'multi-axes_154', 'multi-axes_190', 'multi-axes_207', 'multi-axes_213', 'multi-axes_223', 'multi-axes_248', 'multi-axes_277', 'multi-axes_291']:
            return None, None
        # color_pattern_matches_list = re.findall(color_common_pattern, grounded_code)
        # color_pattern_matches_set = set(color_pattern_matches_list)
        # if color_pattern_matches_list and len(color_pattern_matches_list) != len(color_pattern_matches_set):
        #     # unequal length - duplicates
        #     return None, None
       
    if chart_type in ["bar_chart", "bar_chart_num", "line_chart", "line_chart_num", "multi-axes"]:

    # bar/bar-num/line/line-num: manual assignment for each attribite (else automatic) - no loop used
        # manually check for color types + replace accordingly
        # Assuming 'in a scipt' - all color patterns same | mixed (different color pattern exp) not handled
        if re.search(color_single_char_pattern, grounded_code):
            # print(f"Tag: single char pattern")
            updated_code, color_replacement_list = replace_colors_attribute_in_code(color_single_char_pattern, grounded_code)
            # grounded_code = re.sub(color_single_char_pattern, lambda x: f"color='{randomize_color(x.group(1))}'", grounded_code)
        elif re.search(color_hex_pattern, grounded_code):
            # print(f"Tag: hex pattern")
            updated_code, color_replacement_list = replace_colors_attribute_in_code(color_hex_pattern, grounded_code)
            # grounded_code = re.sub(color_hex_pattern, lambda x: f"color='{randomize_color(x.group(1))}'", grounded_code)
        elif re.search(color_letters_pattern, grounded_code):
            # print(f"Tag: letters pattern")
            updated_code, color_replacement_list = replace_colors_attribute_in_code(color_letters_pattern, grounded_code)
            # grounded_code = re.sub(color_letters_pattern, lambda x: f"color='{randomize_color(x.group(1))}'", grounded_code)
        elif re.search(color_rgb_pattern, grounded_code):
            # print(f"Tag: RGB pattern")
            updated_code, color_replacement_list = replace_colors_attribute_in_code(color_rgb_pattern, grounded_code)
        else:
            print("No color parameters found.")

    elif chart_type in ["rose", "radar", "box", "3D-Bar", "area_chart"]:
        # colors as list -> extract + replace it
        re_color_list_check_result = re.search(color_list_pattern, grounded_code)

        # if 'range' then color list not explicitly defined, instead sequenced from a set
        if re_color_list_check_result and ("range" not in re_color_list_check_result.group(0)):
            # print(f"image name is: {fig_name}")
            # print(grounded_code)
            # breakpoint()

            color_list_exp = re_color_list_check_result.group(0)
            print(f"color exp is: {color_list_exp}")
            isPlural = False
            if "colors" in color_list_exp: # color vs color(s)
                isPlural = True
            list_st = color_list_exp.find('[')
            list_end = color_list_exp.find(']')
            color_list = color_list_exp[list_st+1:list_end].split(',')

            # remove blank spaces, quotes   
            color_list = [color_element.replace(" ", "").replace("\'", "").replace('\"', '').replace('\n', '') for color_element in color_list]
            print(f"color list is: {color_list}")
            updated_code, color_replacement_list = replace_colors_list_in_code(color_list, grounded_code, isPlural)
        else:
            print(f"No color list (default or Color-set)")

    # if color_replacement_list:       
            # print("\nColor Replacement Mapping:-")
            # for initial, modified in color_replacement_list:
            #     print(f"initial: {initial} -> modified: {modified}")

    #######################################
    # Mark rows/columns to color change list

    color_change_jsons = None
    
    if color_replacement_list: # colors replaced, can construct labels
        color_change_label_json, initial_img_colors_json, modified_img_colors_json = get_color_change_jsons(color_replacement_list, img_json_info)
        if color_change_label_json: # labels formed successfully
            color_change_jsons = (color_change_label_json, initial_img_colors_json, modified_img_colors_json)

            # breakpoint()

    return updated_code, color_change_jsons
##### COLOR ALTER |END|


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

size_options_list = [8, 10, 12, 14, 16, 18, 20, 22]                    # 2 step - allow detection, range (beyond it too large/small)
weight_options_list = ["light", "normal", "bold"]                      # "normal" - default value, ultrabold/ultralight - skip (disturb chart layout)
fontfamily_options_list = ['sans-serif', 'serif', 'cursive', 'fantasy', 'monospace'] # 'sans-serif' - default font family
# other font attributes: not universal (rotation, horizontal alignment) | color: confuse with bar/line color change (plus not universal)

"""
FONT DEFAULT VALUES
- figure.titlesize: large | axes.titlesize: large
- axes.labelsize: medium 
- legend.fontsize: medium  
- xtick/labelsize/ytick.labelsize: medium
- font.size: small (markings)
    - large: 14 | medium: 12 | small: 10
"""

chart_region_with_font_list = ["chart title", "chart legend", "chart axes labels", "chart axes ticks"]
chart_font_attribute_list = ["size", "weight", "fontfamily"]

default_font_val_json = {
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

def get_rc_font_attr_to_insert():

    initial_img_font_json = deepcopy(default_font_val_json)
    modified_img_font_json = deepcopy(default_font_val_json)

###-----------------------
# Objective: PARTIAL CHANGE (title/legend/axes labels/axes ticks)
# Current code: already accomplish
# random selection (from list): If same as initial -> NO-CHANGE, or different -> CHANGE
# json - all storted (whether changed/not) | during evaluation - only focus on changed sections

    ### FONT MODIFICATIONS
    # label: font size
    modified_font_size_list = random.choices(size_options_list, k = 4)
    modified_img_font_json["chart title"]["size"] = modified_font_size_list[0]
    modified_img_font_json["chart legend"]["size"] = modified_font_size_list[1]
    modified_img_font_json["chart axes labels"]["size"] = modified_font_size_list[2]
    modified_img_font_json["chart axes ticks"]["size"] = modified_font_size_list[3]
    # rc params: font size
    rc_params_font_size = f"""
plt.rc('axes', titlesize={modified_font_size_list[0]}) # title
plt.rc('legend', fontsize={modified_font_size_list[1]}) # legend
plt.rc('axes', labelsize={modified_font_size_list[2]}) # x and y labels
plt.rc('xtick', labelsize={modified_font_size_list[3]}) # x tick labels
plt.rc('ytick', labelsize={modified_font_size_list[3]}) # y tick labels\n
"""

    # label: font weight
    modified_font_weight_list = random.choices(weight_options_list, k = 2)
    modified_img_font_json["chart title"]["weight"] = modified_font_weight_list[0]
    modified_img_font_json["chart axes labels"]["weight"] = modified_font_weight_list[1]
    # rc params: font weight
    rc_params_font_weight = f"""
plt.rc('axes', titleweight='{modified_font_weight_list[0]}') # title
plt.rc('axes', labelweight='{modified_font_weight_list[1]}') # x and y labels\n
"""
    
    # label: font family 
    modified_font_family = random.choice(fontfamily_options_list)
    modified_img_font_json["chart title"]["fontfamily"] = modified_font_family
    modified_img_font_json["chart legend"]["fontfamily"] = modified_font_family
    modified_img_font_json["chart axes labels"]["fontfamily"] = modified_font_family
    modified_img_font_json["chart axes ticks"]["fontfamily"] = modified_font_family
    # rc params: font family
    rc_params_font_family = f"""
plt.rcParams['font.family'] = '{modified_font_family}'\n
"""

###-----------------------

    rc_font_attr_code_to_insert = rc_params_font_size + rc_params_font_weight + rc_params_font_family

    # font change json
    font_change_label_json = {}
    for chart_region_with_font in chart_region_with_font_list:
        chart_region_json = {}
        for chart_font_attribute in chart_font_attribute_list:
            chart_region_json[chart_font_attribute] = {"initial value": initial_img_font_json[chart_region_with_font][chart_font_attribute], "modified value": modified_img_font_json[chart_region_with_font][chart_font_attribute]}
        font_change_label_json[chart_region_with_font] = chart_region_json

    font_change_jsons = (font_change_label_json, initial_img_font_json, modified_img_font_json)
    return rc_font_attr_code_to_insert, font_change_jsons

def change_chart_font(img_json_info):

    grounded_code = img_json_info["redrawing"]["output"]
    chart_type = img_json_info["chart_type"]
    updated_code, font_change_jsons = None, None


    # DEFAULT: pattern (rcParams) - to change + add in the script
    # mapping - default settings vs changed ones

    isCustomFontDefined = check_custom_font(grounded_code)
    if isCustomFontDefined: # changes ove-ride rcParams change
        return None, None
    # breakpoint()

    # IF legend in best position - may shift due to font change
    if chart_type not in ["3D-Bar", "box"]: #3D, box charts don't have legends 
        # more general pattern
        # legend_pos_pattern = r"\.legend\s*\([]\)"
        legend_pos_pattern = r"\.legend\s*\(\s*(?:[^()]*?\([^()]*?\)[^()]*?,\s*)*?loc\s*=\s*['\"][ \w]+['\"](?:\s*,[^()]*?)*\s*\)"
        # legend_pos_pattern = r"\.legend\s*\(\s*loc\s*=\s*['\"][ \w]+['\"]\s*\)"
        legend_pos_check_result = re.search(legend_pos_pattern, grounded_code)
        if not (legend_pos_check_result and ("best" not in legend_pos_check_result.group(0))): # "best": can't determine position, hence skip the legend position        
            # breakpoint()
            return None, None
    

    plotlib_line_to_find = "import matplotlib.pyplot as plt\n"
    plotlib_line_len = len(plotlib_line_to_find)
    plotlib_line_pos = grounded_code.find(plotlib_line_to_find) + plotlib_line_len
    # breakpoint()

    # font attributes (initial: default values | modified: randomly generated)
    rc_font_attr_code_to_insert, font_change_jsons = get_rc_font_attr_to_insert()   

    updated_code = grounded_code[:plotlib_line_pos] + rc_font_attr_code_to_insert + grounded_code[plotlib_line_pos:]
    print(updated_code)    
    # breakpoint()

    return updated_code, font_change_jsons


def change_chart_rotation(grounded_code):
    """
        randomize rotation
    """
    rotation_change = False
    if re.search(rotation_pattern, grounded_code):
        updated_code = re.sub(rotation_pattern, lambda x: f"rotation={randomize_rotation(x.group(1))}", grounded_code)
        rotation_change = True

    if not rotation_change:
        return None
    return updated_code
    

def change_chart_size(grounded_code):
    """
        randomize fig-size
    """
    if re.search(figsize_pattern, grounded_code):
        updated_code = re.sub(figsize_pattern, lambda x: f"figsize={randomize_figsize(x.group(1))}", grounded_code)
        figsize_change = True

    if not figsize_change:
        return None
    return updated_code


def replace_legend_pos_attribute_in_code(legend_pos_pattern, grounded_code, chart_type):

    legend_pos_options_list = ["upper right", "upper left", "lower left", "lower right", "center left", "center right", "lower center", "upper center", "center"]
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

        modified_legend_pos = random.choice(legend_pos_options_list)
        print(modified_legend_pos)

        legend_pos_replacement_pair.append((initial_legend_pos, modified_legend_pos))
    
        return f".legend(loc=\'{modified_legend_pos}\')"

    # Perform the substitution and track replacements
    updated_code = re.sub(legend_pos_pattern, legend_pos_replacement_function, grounded_code)
    
    return updated_code, legend_pos_replacement_pair

def get_legend_pos_change_jsons(legend_pos_replacement_pair):

    legend_initial_pos, legend_modified_pos = legend_pos_replacement_pair[0]

    # legend position change label
    legend_pos_change_label_json = {"position": {"initial value": legend_initial_pos, "modified value": legend_modified_pos}}

    # legend position label (initial image)
    initial_img_legend_pos_json = {"position": legend_initial_pos}
    # legend position label (modified image)
    modified_img_legend_pos_json = {"position": legend_modified_pos}

    print(legend_pos_change_label_json, initial_img_legend_pos_json, modified_img_legend_pos_json)

    return legend_pos_change_label_json, initial_img_legend_pos_json, modified_img_legend_pos_json

def change_chart_legend_pos(img_json_info):

    fig_name = data_element['imgname']
    chart_type = img_json_info['chart_type']
    grounded_code = img_json_info["redrawing"]["output"]
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
            updated_code, legend_pos_replacement_pair = replace_legend_pos_attribute_in_code(legend_pos_pattern_radar_chart_type, grounded_code, chart_type) 
            print(legend_pos_replacement_pair)
            
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
                updated_code, legend_pos_replacement_pair = replace_legend_pos_attribute_in_code(legend_pos_pattern_rose_chart_type, grounded_code, chart_type) 
                print(legend_pos_replacement_pair)
    
    else:
        legend_pos_pattern = r"\.legend\s*\(\s*loc\s*=\s*['\"][ \w]+['\"]\s*\)"
        legend_pos_check_result = re.search(legend_pos_pattern, grounded_code)

        if legend_pos_check_result and ("best" not in legend_pos_check_result.group(0)): # "best": can't determine position, hence skip the legend position        
            updated_code, legend_pos_replacement_pair = replace_legend_pos_attribute_in_code(legend_pos_pattern, grounded_code, chart_type) 

    #######################################
    # legend replacement labels -> return legend change + initial/final image labels

    legend_pos_jsons = None

    if legend_pos_replacement_pair:
        legend_pos_change_label_json, initial_img_legend_pos_json, modified_img_legend_pos_json = get_legend_pos_change_jsons(legend_pos_replacement_pair)
        legend_pos_jsons = (legend_pos_change_label_json, initial_img_legend_pos_json, modified_img_legend_pos_json)

    return updated_code, legend_pos_jsons

def run_chart_code_save_img(chart_code, fig_name, attribute_altered):

    imgs_save_sub_dir = os.path.join(imgs_save_dir, f"{attribute_altered}_altered")
    os.makedirs(imgs_save_sub_dir, exist_ok = True)

    # replace save location
    if "plt.show()" in chart_code:
        chart_code = remove_plt_show(chart_code, fig_name, imgs_save_sub_dir)
    elif "savefig" in chart_code or "write_image" in chart_code:
        chart_code = update_savefig_location(chart_code, fig_name, imgs_save_sub_dir)

    # resetting rcParams at end of script (same execution shell hence values persist, for next chart need to reset)
    
    isCodeRunSuccess = generate_chart(chart_code)
    # isCodeRunSuccess = False
    # try:
    #     exec(chart_code)
    #     print("code run success!")
    #     isCodeRunSuccess = True
    # except:
    #     print("ERROR! execution fail..")
    #     print(f"modified code:-\n{chart_code}\n\n")
    #     # breakpoint()

    return isCodeRunSuccess

def get_chart_change_json(imgname, attr_change_type, attr_change_json):

    chart_change_json = {}
    chart_change_json["imgname"] = imgname
    chart_change_json["attr_change_type"] = attr_change_type
    chart_change_json["attr_change_json"] = attr_change_json
    return chart_change_json

"""
save (image-json + attr-comparison)
"""


###--------------------------     MAIN CODE

# ctype_color_replacement_list = ["bar_chart", "bar_chart_num", "line_chart", "line_chart_num"]
total_cnt = 0
all_3_success_cnt = 0
attr_change_success_count = {"color": 0, "font": 0, "legend_pos": 0}
fraction_of = None

attr_change_json_list = []

initial_img_gt_json_list = []

for data_element in data:
    grounded_code = data_element["redrawing"]["output"]
    fig_name = data_element['imgname']
    chart_type = data_element['chart_type']

    if chart_type not in chart_type_names_list:
        continue

    total_cnt += 1
    # fraction_of = 20
    # if total_cnt % fraction_of:
    #     continue
    print(f"IMAGE TAG - image is: {fig_name}")

    # mark successful attribute changes
    isCodeRunSuccess = {"color": False, "font": False, "legend_pos": False}
    initial_img_gt_json = {"imgname": fig_name}

    # COLOR CHANGE
    color_change_code, color_change_jsons = change_chart_color(img_json_info = data_element)
    if color_change_jsons:
        color_change_label_json, initial_img_colors_json, modified_img_colors_json = color_change_jsons
        print(f"color change json: {color_change_label_json}")
        print(f"initial image colors: {initial_img_colors_json}")
        print(f"modified image colors: {modified_img_colors_json}")
        # breakpoint()    

        # color_change_fig_name = fig_name + "_color_change"
        if run_chart_code_save_img(color_change_code, fig_name, "color"):
            # isCodeRunSuccess["color"] = True
            initial_img_gt_json["color"] = initial_img_colors_json
            attr_change_success_count["color"] += 1
            attr_change_json_list.append(get_chart_change_json(fig_name, "color", color_change_label_json))
        
    # FONT CHANGE
    font_change_code, font_change_jsons = change_chart_font(img_json_info = data_element)
    if font_change_jsons:
        font_change_label_json, initial_img_font_json, modified_img_font_json = font_change_jsons
        print(f"font change json: {font_change_label_json}")
        print(f"initial image font: {initial_img_font_json}")
        print(f"modified image font: {modified_img_font_json}")    

        # font_change_fig_name = fig_name + "_font_change"
        if run_chart_code_save_img(font_change_code, fig_name, "text_style"):
            # isCodeRunSuccess["font"] = True
            initial_img_gt_json["font"] = initial_img_font_json
            attr_change_success_count["font"] += 1
            attr_change_json_list.append(get_chart_change_json(fig_name, "text_style", font_change_label_json))

    legend_pos_change_code, legend_pos_jsons = None, None
    if chart_type not in ["3D-Bar", "box", "multi-axes"]: #3D, box charts don't have legends | multi-axes: multiple
        # LEGEND CHANGE
        legend_pos_change_code, legend_pos_jsons = change_chart_legend_pos(img_json_info = data_element)
        print(legend_pos_jsons)
        if legend_pos_jsons:
            legend_pos_change_label_json, initial_img_legend_pos_json, modified_img_legend_pos_json = legend_pos_jsons
            print(f"legend-pos change json: {legend_pos_change_label_json}")
            print(f"legend position initial image: {initial_img_legend_pos_json}")
            print(f"legend position modified image: {modified_img_legend_pos_json}")
            # breakpoint()

            # legend_pos_change_fig_name = fig_name + "_legend_pos_change"
            if run_chart_code_save_img(legend_pos_change_code, fig_name, "legend"):
                # isCodeRunSuccess["legend_pos"] = True
                initial_img_gt_json["legend_pos"] = initial_img_legend_pos_json
                attr_change_success_count["legend_pos"] += 1
                attr_change_json_list.append(get_chart_change_json(fig_name, "legend", legend_pos_change_label_json))


    # if isCodeRunSuccess["color"] and isCodeRunSuccess["font"] and isCodeRunSuccess["legend_pos"]:
    #     all_3_success_cnt += 1
    if ("color" in initial_img_gt_json) and ("font" in initial_img_gt_json) and ("legend_pos" in initial_img_gt_json):
        all_3_success_cnt += 1
        initial_img_gt_json_list.append(initial_img_gt_json)

# with open(initial_imgs_info_json, 'w') as image_attr_label_json_file:
#     json.dump(initial_img_gt_json_list, image_attr_label_json_file, indent=3)

with open(attribute_change_info_json, 'w') as attr_change_json_file:
    json.dump(attr_change_json_list, attr_change_json_file, indent=4)

if fraction_of:
    print(f"FRACTION OF CHARTS SEEN!")
    total_cnt = total_cnt / fraction_of

print(f"total charts: {total_cnt}")

print(f"color-change Success rate: {(100. * attr_change_success_count['color'])/total_cnt}")
print(f"font-change Success rate: {(100. * attr_change_success_count['font'])/total_cnt}")
print(f"legend_pos-change Success rate: {(100. * attr_change_success_count['legend_pos'])/total_cnt}")

print(f"all 3 Success rate: {(100. * all_3_success_cnt)/total_cnt}")

###-------------------------------

