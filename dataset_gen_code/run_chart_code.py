import json
import re
import os

# import functions, to correct chart-save location in the image
from savefig_helper import update_savefig_location, remove_plt_show, update_fig_loc_val_in_var, generate_chart

chart_type_names_list = ['bar_chart_num', 'bar_chart', '3D-Bar', 'line_chart_num', 'line_chart', 'radar', 'rose', 'box', 'multi-axes']

json_path = os.path.join(os.getcwd(), 'ChartX_annotation.json')

imgs_save_dir = os.path.join(os.getcwd(), "_datasets", "ChartX_imgs_from_code_run_modified") 
os.makedirs(imgs_save_dir, exist_ok = True)

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

total_cnt = 0
cnt = 0

for data_element in data:

    imgname = data_element["imgname"]
    chart_type = data_element["chart_type"] 

    if chart_type not in chart_type_names_list:
        continue

    print(f"chart name is: {imgname}")
    fig_name = imgname
    # if chart_type in ["histogram", "funnel", "treemap"]:
    #     fig_name = chart_type + "_" + imgname 


    chart_code = data_element["redrawing"]["output"]
    chart_code = chart_code.replace(" \\t ", " \t ").replace(" \\n ", " \n ") # reading csv from text (replace \\t, \\n by \t, \n respectively)
                                                                            # raw_data.strip().split('\\n') - don't want replaced by \n


    if "plt.show()" in chart_code:
        chart_code = remove_plt_show(chart_code, fig_name, imgs_save_dir)
    elif "savefig" in chart_code or "write_image" in chart_code:
        chart_code = update_savefig_location(chart_code, fig_name, imgs_save_dir)

    isCodeRunSuccess = generate_chart(chart_code)

    if isCodeRunSuccess:
       cnt += 1
    total_cnt += 1

print(f"total charts: {total_cnt}")
print(f"% succcess: {(1.* cnt / total_cnt) * 100}")

