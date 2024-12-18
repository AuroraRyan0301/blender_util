
#!/usr/bin/bash

blender_path=/root/blender-3.5.0-linux-x64/blender
# objs=(chair_upup)
# light=vertical_down


# 设置目录
BLENDFILES_DIR="/data/pilot/blendfiles"



parent_dir=/root/render_scripts
python_path1=render_pano.py
out_dir=/data2/new_ris_with_rand_bg
scene=/data/pilot/blendfiles/full_list.txt

${blender_path} --background  --python ${parent_dir}/${python_path1} -- ${scene} ${out_dir} 
