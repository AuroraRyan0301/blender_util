
#!/usr/bin/bash
file_number="$1"

blender_path=/root/blender-3.5.0-linux-x64/blender
# objs=(chair_upup)
# light=vertical_down


# 设置目录
BLENDFILES_DIR="/data/pilot/blendfiles"



scene_dirs=()

cd $BLENDFILES_DIR
txt_file="/data/pilot/blendfiles/list${file_number}.txt"

parent_dir=/root/blender_util
python_path1=pano_envmap.py


final_parent_path=/data2/my_ris

# 检查文件是否存在
if [ -f "$txt_file" ]; then
    # 逐行读取文件并打印
    while IFS= read -r line; do
        echo "process $line"
        scene_filename=$(echo "$line" | sed 's/\.final\.blend$//')
        ${blender_path} --background --factory-startup ${BLENDFILES_DIR}/$line --python ${parent_dir}/${python_path1} -- ${scene_filename} ${final_parent_path}
    done < "$txt_file"
else
    echo "txt file not exist: $txt_file"
fi

gpus=(0)


# for scene in ${scene_dirs[@]}; do
#     echo ${scene}
    ${blender_path} --background --factory-startup ${BLENDFILES_DIR}/${scene_name}.final.blend --python ${parent_dir}/${python_path1} -- ${scene_name} ${final_parent_path}
# done