import re

"""
SINGLE AND DOUBLE QUOTES
* used inter-changeably for storing string 

if one inside another - counted as regular char (storing code in string form)
    Ex: "..............'---'............"

if same type, then need to add escpe char
    Ex: "............\"-----\".........."

regex: same logic

* while writing (if different char) adjusted
    Ex: phrase - '.........."----"....'
    when write within a string "XXXXXXXXXXXX"
    adjusted: "XXXXX\"----\"XXXXXXX"
"""


plt_show_pattern = r"plt\.show\(\)"

""" (,\s*[^)]+)*
    - optional exp - 0 or more times
    - exp: comma followed by whitespaces followed by chars (not brackets)
    Example: plt.savefig('bar chart/png/401.png', bbox_inches='tight')
"""

# single quoted, plt.savefig OR fig.savefig
savefig_all_patterns = r"plt\.savefig\('[^']+'(,\s*[^)]+)*\)|fig\.savefig\('[^']+'(,\s*[^)]+)*\)"
savefig_all_double_quote_patterns = r'plt\.savefig\("[^"]+"(,\s*[^)]+)*\)|fig\.savefig\("[^"]+"(,\s*[^)]+)*\)'

""" single/double quote (inside savefig) - need separate (though mostly single qoute)
    Example
    single -> "....................... plt.savefig('...')......."
    double -> '....................... plt.savefig("...").......'
"""
# plt.savefig, fig.savefig
savefig_location_pattern = r"savefig\('[^']+'(,\s*[^)]+)*\)"  
savefig_location_double_quote_pattern = r'savefig\("[^"]+"(,\s*[^)]+)*\)'

# fig.write_image, pio.write_image
fig_savefig_location_pattern = r"write_image\('[^']+'\)"
fig_savefig_location_double_quote_pattern = r'write_image\("[^"]+"\)'

# update fig location (plt.savefig OR fig.savefig)
def update_savefig_location(code, fig_name, imgs_save_dir):
    code += "\nplt.rcdefaults()\n"
    
    if re.search(savefig_location_pattern, code):
        return re.sub(savefig_location_pattern, f"savefig('{imgs_save_dir}/{fig_name}.png')", code)
    if re.search(savefig_location_double_quote_pattern, code):
        return re.sub(savefig_location_double_quote_pattern, f'savefig("{imgs_save_dir}/{fig_name}.png")', code)
    
    if re.search(fig_savefig_location_pattern, code):
        return re.sub(fig_savefig_location_pattern, f"write_image('{imgs_save_dir}/{fig_name}.png')", code)
    if re.search(fig_savefig_location_double_quote_pattern, code):
        return re.sub(fig_savefig_location_double_quote_pattern, f'write_image("{imgs_save_dir}/{fig_name}.png")', code)
    

# remove plt.show
def remove_plt_show(code, fig_name, imgs_save_dir):
    code += "\nplt.rcdefaults()\n"
    
    if "savefig" in code:
        # check if code has savefig, remove it (probably incorrect location, need to rewrite)    
        code = re.sub(savefig_all_patterns, "", code)
        code = re.sub(savefig_all_double_quote_patterns, '', code)
    # substitute plt.show() with plt.savefig()
    return re.sub(plt_show_pattern, f"plt.savefig('{imgs_save_dir}/{fig_name}.png')", code)
    

def update_fig_loc_val_in_var(code, fig_name, imgs_save_dir, var_name):
    code += "\nplt.rcdefaults()\n"

    save_path_loc_line = re.escape(var_name) + r"\s*=\s*[\'\"]([^\'\"]+)[\'\"]"
    if re.search(save_path_loc_line, code):
        return re.sub(save_path_loc_line, f"{var_name} = '{imgs_save_dir}/{fig_name}.png'", code)

def generate_chart(chart_code):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use a non-GUI backend
    import matplotlib.pyplot as plt

    # Create a dictionary to store the function from the executed code
    local_context = {"np": np, "plt": plt, "matplotlib": matplotlib}

    # Execute the code to define the function
    try:
        exec(chart_code, local_context)
        return True
    except: 
        return False
