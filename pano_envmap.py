import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
from collections import defaultdict
import random
import math

sys.path.append(os.path.dirname(__file__))

argv = sys.argv
argv = argv[argv.index("--") + 1:]

SCENE_Name = argv[0]
OUTDIR = argv[1]

DEBUG = False
RESULTS_PATH = 'outputs'
FORMAT = 'OPEN_EXR'
RENDER_NUM = 30

fp = bpy.path.abspath(f"{OUTDIR}/{SCENE_Name}")
print(fp)
random.seed(114514)

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
        if 0 <= height <= 10:
            height_count[height].append(light)

    # 找出高度相同数量最多的组
    max_count = max(len(v) for v in height_count.values())
    target_height = [h for h, lights in height_count.items() if len(lights) == max_count][0]
    if target_height is None:
        target_height = 4.0
        avg_x = 1.0
        avg_y = 1.0
    else:

        target_lights = height_count[target_height]

        total_x = sum(light.location.x for light in target_lights)
        total_y = sum(light.location.y for light in target_lights)
        avg_x = total_x / len(target_lights)
        avg_y = total_y / len(target_lights)

    return avg_x, avg_y, target_height - 1.0

def get_light_lists():
    # 获取场景中的所有光源
    lights = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
    return lights

def get_a_close_pos_around_lights(room_center):
    lights = get_light_lists()
    
    if not lights:
        print("没有找到光源")
        return
    light = random.choice(lights)
    
    # 获取光源的边界体积
    light_bound_box = light.bound_box
    if light_bound_box:
        # 计算光源的中心位置
        light_center = light.location
        
        # 计算 BVH 半径
        bvh_size = max(mathutils.Vector(light_bound_box[0]) - mathutils.Vector(light_bound_box[6]))
        bvh_radius =  bvh_size  / 2

        direction_vector = mathutils.Vector(room_center) - light_center
        direction_vector.z = 0.0
        direction_vector.normalize()  # Make sure the vector is normalized to unit length

        random_angle = random.uniform(-math.radians(45), math.radians(45))
        
        # 旋转方向向量：绕z轴旋转random_angle
        rotation_matrix = mathutils.Matrix.Rotation(random_angle, 4, 'Z')
        perturbed_direction_vector = rotation_matrix @ direction_vector
        perturbed_direction_vector.normalize()
        
        distance = random.uniform(bvh_radius+0.6, bvh_radius + 1.0)

        x_offset = perturbed_direction_vector.x * distance
        y_offset = perturbed_direction_vector.y * distance
        
        new_loc_x = light_center.x + x_offset
        new_loc_y = light_center.y + y_offset
        new_loc_z = light_center.z
        return new_loc_x , new_loc_y , new_loc_z
        
    
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
            
            # 计算相机位置与物体位置的距离
            distance = (obj_location - cam_loc).length
            
            # 判断相机是否与物体相交
            if distance < (radius + obj_radius):
                return True
    return False

def set_camera_position(cam_obj, cam_x, cam_y, height=3.0):
    
    # 设定一个初始位置
    cam_loc = (cam_x, cam_y, height)
    
    # 设定相机的半径（假设为0.5米）
    cam_radius = 0.5
    
    # 检查相机位置是否发生碰撞
    if is_camera_colliding(cam_loc, radius=cam_radius):
        # 如果碰撞，向上移动相机一段距离，避免与物体发生碰撞
        return True

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
    depth_file_output.file_slots[0].path = "depth"
    depth_file_output.base_path = fp
    # normalization_layer = tree.nodes.new("CompositorNodeNormalize")

    # links.new(render_layers.outputs['Depth'], normalization_layer.inputs[0])
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

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
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
scene.cycles.samples = 512
scene.cycles.use_denoising = True
scene.render.resolution_x = 512
scene.render.resolution_y = 256

cam.cycles.panorama_type = "EQUIRECTANGULAR"

scene.render.image_settings.file_format = FORMAT

# get all exr files in dir
folder_path = "/root/blender_util/envmaps"
exr_files = [f for f in os.listdir(folder_path) if f.endswith('.exr')]
if not exr_files:
    print(f'[Error] No EXR files found in the folder: {folder_path}')

# random choose
selected_exr = random.choice(exr_files)
exr_path = os.path.join(folder_path, selected_exr)

with open(fp + "/texture_exr.txt", 'w') as file:
    file.write(f"{selected_exr}\n")

if exr_path:
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("MyWorld")
        bpy.context.scene.world = world

    world.use_nodes = True

    world_nodes = world.node_tree.nodes

    # get bg node
    background_node = world_nodes.get("Background")
    if not background_node:
        background_node = world_nodes.new(type="ShaderNodeBackground")

    original_envmap = False
    for node in world_nodes:
        if node.type == 'TEX_IMAGE':
            original_envmap = True

    if original_envmap == False:

        # create envmap node
        env_texture_node = world_nodes.new(type="ShaderNodeTexEnvironment")

        env_texture_node.image = bpy.data.images.load(exr_path)
        world.node_tree.links.new(env_texture_node.outputs["Color"], background_node.inputs["Color"])
        background_node.inputs["Strength"].default_value = 1.0

        # display bg
        bpy.context.scene.render.film_transparent = False 

        print("Blender has created shaping tree and new envmap")

for render_indice in range(RENDER_NUM+1):
    depth_file_output.file_slots[0].path = "depth_{}".format(render_indice)
    if render_indice == 0:
        cam_x, cam_y, target_height = get_cam_loc_from_light_sources()
        room_center = (cam_x*1.0, cam_y*1.0, target_height*1.0)
    else:
        cam_x, cam_y, _ = get_a_close_pos_around_lights(room_center)
    # collision_judge = True
    # while collision_judge == True:
    #     collision_judge=set_camera_position(cam_obj, cam_x, cam_y, target_height-1.0)
    #     random_x = random.uniform(-1, 1)
    #     random_y = random.uniform(-1, 1)
    #     random_height = random.uniform(-1, 1)
    #     cam_x += random_x 
    #     cam_y += random_y 
    #     target_height += random_height


    cam_obj.location = (cam_x, cam_y, target_height)
    cam_obj.rotation_euler = (1.3467,0,-1.57079632)
    # cam_obj.rotation_euler = (0,0,0)

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
    output_dir = fp + "/camera_info"
    output_file = os.path.join(bpy.path.abspath(output_dir), "camera_info_{}.txt".format(render_indice))
    print("Saved camera info to: {}".format(output_file))

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

    scene.render.filepath = fp + "/rgb_{}.exr".format(render_indice)
    bpy.ops.render.render(write_still=True)