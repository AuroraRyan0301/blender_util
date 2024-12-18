import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
from collections import defaultdict
import random
from mathutils import Vector

sys.path.append(os.path.dirname(__file__))

argv = sys.argv
argv = argv[argv.index("--") + 1:]

SCENE_Name = argv[0]
OUTDIR = argv[1]

DEBUG = False
RESULTS_PATH = 'outputs'
FORMAT = 'OPEN_EXR'

scene_suffix=SCENE_Name[:-6]

fp = bpy.path.abspath(f"{OUTDIR}/{scene_suffix}")
print(fp)

def enable_cuda():
    ## this function if borrowed from https://github.com/nytimes/rd-blender-docker/issues/3#issuecomment-618459326
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences

    # Calling this purges the device list so we need it
    cprefs.refresh_devices()
    # cuda_devices, opencl_devices = cprefs.devices[:2]
    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL'):
        try:
            cprefs.compute_device_type = compute_device_type
            break
        except TypeError:
            pass

    # Enable all CPU and GPU devices
    for device in cprefs.devices:
        if device.type == 'CPU':
            device.use = False
        else:
            device.use = True

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

if not os.path.exists(fp):
    os.makedirs(fp)

def get_cam_loc_from_light_sources():
    # 获取场景中的所有光源
    lights = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']

    # 统计光源高度
    height_count = defaultdict(list)
    for light in lights:
        height = light.location.z
        if 2 <= height <= 8:
            height_count[height].append(light)

    # 找出高度相同数量最多的组
    max_count = max(len(v) for v in height_count.values())
    target_height = [h for h, lights in height_count.items() if len(lights) == max_count][0]

    # 获取高度相同数量最多且高度在2到8之间的光源组
    target_lights = height_count[target_height]

    total_x = sum(light.location.x for light in target_lights)
    total_y = sum(light.location.y for light in target_lights)
    avg_x = total_x / len(target_lights)
    avg_y = total_y / len(target_lights)
    return avg_x, avg_y, target_height
    
def is_camera_colliding(cam_loc, radius=1.0):
    """
    检测相机位置是否与物体发生碰撞。通过相机位置和给定半径构成一个球体来进行碰撞检测。
    
    :param cam_loc: 相机位置 (Vector)
    :param radius: 检测的半径，默认1.0
    :return: True 如果发生碰撞，False 如果没有碰撞
    """
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.hide_viewport == False:  # 只检测可见的物体
            # 获取物体的包围球（Bounding Sphere）
            obj_location = obj.location
            obj_radius = obj.dimensions.length / 2  # 取物体的半径（假设物体是球形或立方体）
            if isinstance(cam_loc, tuple):
                cam_loc = Vector(cam_loc)
            # 计算相机位置与物体位置的距离
            distance = (obj_location - cam_loc).length
            
            # 判断相机是否与物体相交
            if distance < (radius + obj_radius):
                return True
    return False

def set_camera_position(cam_obj, cam_x, cam_y, height=3.0):
    
    # 设定一个初始位置
    cam_loc = Vector((cam_x, cam_y, height))
    
    # 设定相机的半径（假设为0.5米）
    cam_radius = 0.5
    
    # # 检查相机位置是否发生碰撞
    # if is_camera_colliding(cam_loc, radius=cam_radius):
    #     # 如果碰撞，向上移动相机一段距离，避免与物体发生碰撞
    #     return True

    # 更新相机位置
    cam_obj.location = cam_loc

    # 你可以继续设置相机的旋转和其他属性
    cam_obj.rotation_euler = (1.3467, 0, -1.57079632)
    return False
    
enable_cuda()

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

## # Turn off interreflections
#bpy.context.scene.cycles.max_bounces = 12

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

if not DEBUG:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.format.file_format = 'OPEN_EXR'
    depth_file_output.file_slots[0].path = "depth_normalized"
    depth_file_output.base_path = fp
    normalization_layer = tree.nodes.new("CompositorNodeNormalize")

    links.new(render_layers.outputs['Depth'], normalization_layer.inputs[0])
    links.new(normalization_layer.outputs[0], depth_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

scene = bpy.context.scene


old_cam = scene.objects['Camera']
bpy.data.objects.remove(old_cam, do_unlink=True)

# 创建新的相机对象
cam = bpy.data.cameras.new("NewCamera")
cam_obj = bpy.data.objects.new("NewCamera", cam)

cam.type = "PANO"
cam_x, cam_y, target_height = get_cam_loc_from_light_sources()
collision_judge = True
# while collision_judge == True:
collision_judge=set_camera_position(cam_obj, cam_x, cam_y, target_height-1.0)
    # random_x = random.uniform(-1, 1)
    # random_y = random.uniform(-1, 1)
    # random_height = random.uniform(-1, 1)
    # cam_x += random_x 
    # cam_y += random_y 
    # target_height += random_height
    # print("camera collision happens, try again")


# cam_obj.location = (cam_x, cam_y, target_height-1.0)
# cam_obj.rotation_euler = (1.3467,0,-1.57079632)

scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
scene.cycles.samples = 512
scene.cycles.use_denoising = True
scene.render.resolution_x = 512
scene.render.resolution_y = 256

cam.cycles.panorama_type = "EQUIRECTANGULAR"

scene.render.image_settings.file_format = FORMAT

# 获取当前场景中的相机
cam = bpy.context.scene.camera

# 获取相机的位置和旋转角度
camera_location = cam.location
camera_rotation = cam.rotation_euler

# 获取相机的其他属性
camera_focal_length = cam.data.lens
camera_sensor_width = cam.data.sensor_width
camera_sensor_height = cam.data.sensor_height
camera_clip_start = cam.data.clip_start
camera_clip_end = cam.data.clip_end

# 设置输出文件的路径
output_dir = "//camera_info"
output_file = os.path.join(bpy.path.abspath(output_dir), "camera_info.txt")

# 创建输出目录
if not os.path.exists(bpy.path.abspath(output_dir)):
    os.makedirs(bpy.path.abspath(output_dir))

# 写入相机信息到文件
with open(output_file, "w") as f:
    f.write(f"Camera Location: {camera_location}\n")
    f.write(f"Camera Rotation: {camera_rotation}\n")
    f.write(f"Camera Focal Length: {camera_focal_length}\n")
    f.write(f"Camera Sensor Width: {camera_sensor_width}\n")
    f.write(f"Camera Sensor Height: {camera_sensor_height}\n")
    f.write(f"Camera Clip Start: {camera_clip_start}\n")
    f.write(f"Camera Clip End: {camera_clip_end}\n")

scene.render.filepath = fp + "/rgb.exr"
bpy.ops.render.render(write_still=True)