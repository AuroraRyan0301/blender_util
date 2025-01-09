
#!/usr/bin/bash

blender_path=/root/blender-3.5.0-linux-x64/blender
# objs=(chair_upup)
# light=vertical_down


# 设置目录
BLENDFILES_DIR="/data/pilot/blendfiles"



scene_dirs=()

cd $BLENDFILES_DIR
for dir in */
do
    # 检查文件夹名是否以"scene"开头
    if [[ "$dir" =~ ^scene ]]
    then
        # 将文件夹名添加到数组中
        scene_dirs+=("$dir")
    fi
done

gpus=(0)

parent_dir=/root/blender_util
python_path1=pano_envmap.py
scene_name=scene10
final_parent_path=/data2/my_ris

# for scene in ${scene_dirs[@]}; do
#     echo ${scene}
    ${blender_path} --background --factory-startup ${BLENDFILES_DIR}/${scene_name}.final.blend --python ${parent_dir}/${python_path1} -- ${scene_name} ${final_parent_path}
# done