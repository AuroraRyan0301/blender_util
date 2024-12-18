# TODO: Fix wrong camara extrinsics
#
# By Jet

import os
import bpy
import sys
import copy
import math
import random
import shutil
import tempfile
import bpy_extras
import mathutils
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
import numpy as np
from copy import deepcopy as copy

INTMAX = 999999999
np.random.seed(0)
random.seed(0)

class Files():
    def __init__(self, files: list):
        self.files = files
        self.muted = {}

    def write(self, msg):
        for file in self.files:
            file.write(msg)

    def flush(self):
        for file in self.files:
            file.flush()

    def mute(self, idx):
        raise NotImplementedError
        self.files[1].write('Call mute')
        nullpath = os.devnull
        fd = self.files[idx].fileno()
        self.muted[idx] = os.dup(fd)
        self.files[idx].flush()
        os.close(fd)
        self.files[idx] = open(nullpath, 'w')
        assert self.files[idx].fileno() == fd == 1

    def unmute(self, idx):
        raise NotImplementedError
        self.files[1].write('Call unmute')
        assert self.files[idx].name == os.devnull
        fd = self.files[idx].fileno()
        os.close(fd)
        os.dup(self.muted[idx])
        os.close(self.muted[idx])
        del self.muted[idx]
        self.files[idx] = os.fdopen(fd, 'w')


def urand(low, high):
    return random.random() * (high - low) + low


def parseSceneList(fpath):
    with open(fpath, 'r') as f:
        return f.read().splitlines()


def getArg(args, key: str, type, default=None):
    arg_prefix = '--' + key
    if type == bool:
        return arg_prefix in args

    if arg_prefix in args:
        return type(args[args.index(arg_prefix) + 1])
    return default


# TODO: Better folder lock
def dirLocked(d):
    return os.path.exists(os.path.join(d, 'lock'))


def dirLock(d):
    open(os.path.join(d, 'lock'), 'w').close()


def dirUnlock(d):
    try:
        os.remove(os.path.join(d, 'lock'))
    except:
        print(f'[Warning] Failed to release the lock for "{d}"')


def blenderInit(resolution, numSamples, aoBounces=0, use_nodes=True, no_gpu=False):
    # use cycle
    blenderSetRenderParam(resolution, numSamples, 'CYCLES')
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.diffuse_bounces = 6
    bpy.context.scene.cycles.glossy_bounces = 6
    bpy.context.scene.cycles.transparent_max_bounces = 6
    bpy.context.scene.cycles.transmission_bounces = 6

    bpy.context.scene.cycles.debug_use_spatial_splits = False
    bpy.context.scene.cycles.debug_use_hair_bvh = False
    bpy.context.scene.cycles.tile_size = 200
    bpy.context.scene.render.use_persistent_data = True
    # bpy.context.scene.view_settings.view_transform = 'Filmic Log'
    # bpy.context.scene.cycles.film_exposure = exposure

    # set GPU devices
    cyclePref = bpy.context.preferences.addons['cycles'].preferences
    cyclePref.refresh_devices()
    for dev in cyclePref.devices:
        dev.use = True
        print(f'[Info] Detected device "{dev.name}": {dev.use}')

    use_gpu = False
    use_optix = False

    if not no_gpu:
        for _ in range(2):  # For a strange bug under Ubuntu
            for dev in cyclePref.devices:
                if 'NVIDIA' in dev.name:
                    use_gpu = True
                if 'RTX' in dev.name:
                    use_optix = True

    if use_optix:
        cyclePref.compute_device_type = 'OPTIX'
    elif use_gpu:
        cyclePref.compute_device_type = 'CUDA'
    else:
        cyclePref.compute_device_type = 'NONE'
    print(f'[Info] Render Mode: {cyclePref.compute_device_type}')

    if use_gpu:
        bpy.context.scene.cycles.device = 'GPU'

    # use nodes
    bpy.context.scene.use_nodes = use_nodes

    # use camera culling
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.cycles.use_camera_cull = True
    bpy.context.scene.cycles.camera_cull_margin = 0.1
    for obj in bpy.data.objects:
        obj.cycles.use_camera_cull = True

    # use AO Bounces
    bpy.context.scene.cycles.ao_bounces_render = aoBounces


def blenderInitPano(resolution, numSamples, aoBounces=0, use_nodes=True, no_gpu=False):
    # use cycle
    blenderSetRenderParam(resolution, numSamples, 'CYCLES')
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.diffuse_bounces = 6
    bpy.context.scene.cycles.glossy_bounces = 6
    bpy.context.scene.cycles.transparent_max_bounces = 6
    bpy.context.scene.cycles.transmission_bounces = 6

    bpy.context.scene.cycles.debug_use_spatial_splits = False
    bpy.context.scene.cycles.debug_use_hair_bvh = False
    bpy.context.scene.cycles.tile_size = 128
    bpy.context.scene.render.use_persistent_data = True
    # bpy.context.scene.view_settings.view_transform = 'Filmic Log'
    # bpy.context.scene.cycles.film_exposure = exposure

    # set GPU devices
    cyclePref = bpy.context.preferences.addons['cycles'].preferences
    cyclePref.refresh_devices()
    for dev in cyclePref.devices:
        dev.use = True
        print(f'[Info] Detected device "{dev.name}": {dev.use}')

    use_gpu = False
    use_optix = False

    if not no_gpu:
        for _ in range(2):  # For a strange bug under Ubuntu
            for dev in cyclePref.devices:
                if 'NVIDIA' in dev.name:
                    use_gpu = True
                if 'RTX' in dev.name:
                    use_optix = True

    if use_optix:
        cyclePref.compute_device_type = 'OPTIX'
    elif use_gpu:
        cyclePref.compute_device_type = 'CUDA'
    else:
        cyclePref.compute_device_type = 'NONE'
    print(f'[Info] Render Mode: {cyclePref.compute_device_type}')

    if use_gpu:
        bpy.context.scene.cycles.device = 'GPU'

    # use nodes
    bpy.context.scene.use_nodes = use_nodes

    # use camera culling
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.cycles.use_camera_cull = True
    bpy.context.scene.cycles.camera_cull_margin = 0.1
    for obj in bpy.data.objects:
        obj.cycles.use_camera_cull = True

    # use AO Bounces
    bpy.context.scene.cycles.ao_bounces_render = aoBounces


def blenderInitMultiview(resolution, numSamples, aoBounces=0, use_nodes=True, no_gpu=False):
    # use cycle
    blenderSetRenderParam(resolution, numSamples, 'CYCLES')
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.diffuse_bounces = 6
    bpy.context.scene.cycles.glossy_bounces = 6
    bpy.context.scene.cycles.transparent_max_bounces = 6
    bpy.context.scene.cycles.transmission_bounces = 6

    bpy.context.scene.cycles.debug_use_spatial_splits = False
    bpy.context.scene.cycles.debug_use_hair_bvh = False
    bpy.context.scene.cycles.tile_size = 128
    bpy.context.scene.render.use_persistent_data = True
    # bpy.context.scene.view_settings.view_transform = 'Filmic Log'
    # bpy.context.scene.cycles.film_exposure = exposure

    # set GPU devices
    cyclePref = bpy.context.preferences.addons['cycles'].preferences
    cyclePref.refresh_devices()
    for dev in cyclePref.devices:
        dev.use = True
        print(f'[Info] Detected device "{dev.name}": {dev.use}')

    cyclePref.compute_device_type = 'OPTIX'

    # use nodes
    bpy.context.scene.use_nodes = use_nodes

    # use camera culling
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.cycles.use_camera_cull = True
    bpy.context.scene.cycles.camera_cull_margin = 0.1
    for obj in bpy.data.objects:
        obj.cycles.use_camera_cull = True

    # use AO Bounces
    bpy.context.scene.cycles.ao_bounces_render = aoBounces


def blenderGetRenderParam():
    render_settings = bpy.context.scene.render
    return (
        (render_settings.resolution_x, render_settings.resolution_y),
        bpy.context.scene.cycles.samples,
        render_settings.engine,
    )


def blenderSetRenderParam(resolution=None, numSamples=None, engine='CYCLES'):
    render_settings = bpy.context.scene.render
    render_settings.engine = engine

    if resolution is not None:
        render_settings.resolution_x = resolution[0]
        render_settings.resolution_y = resolution[1]

    if numSamples is not None:
        bpy.context.scene.cycles.samples = numSamples


def setSceneDiffuse():
    for m in bpy.data.materials:
        if 'Principled BSDF' in m.node_tree.nodes:
            for i in [4, 5, 6, 8, 9, 12, 13, 15, 16]:
                m.node_tree.nodes['Principled BSDF'].inputs[i].default_value = 0.0


def setSceneReflective():
    for m in bpy.data.materials:
        if 'Principled BSDF' in m.node_tree.nodes:
            m.node_tree.nodes['Principled BSDF'].inputs[4].default_value = 0.5
            m.node_tree.nodes['Principled BSDF'].inputs[5].default_value = 1.0


def getCameras():
    # Dummy, find all cameras
    return list(filter(lambda x: x.type == 'CAMERA', bpy.data.objects))


def duplicateObject(obj):
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    new_obj.animation_data_clear()
    bpy.context.collection.objects.link(new_obj)
    return new_obj


# Hide all other meshes while rendering
# Returns names of all hidden meshes
def isolateMesh(target):
    # return []
    if isinstance(target, str):
        target = bpy.data.objects[target]

    hidden = []
    for obj in bpy.context.scene.objects:
        if obj is not target and obj.type == 'MESH':
            obj.hide_render = True
            hidden.append(obj.name)

    return hidden


def restoreHidden(objs: list):
    for obj_name in objs:
        bpy.data.objects[obj_name].hide_render = False


# TODO: Should be bottom center
def calculateObjectCenterRadius(obj):
    bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = Vector((0, 0, 0))
    for c in bbox:
        center += Vector(c)
    center /= 8
    tmp = np.array(center - Vector(bbox[0]))
    radius = float(np.sqrt((tmp ** 2).sum()))
    return center, radius


# Using 24mm as our base case
def calculateProperCameraDistance(radius, flength):
    factor = flength / 24.0
    return factor * min(1.8 * radius, 2.7) * urand(0.9, 1.0)


def generateRandomUnitBallPos():
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    return Vector(vec)

def perturbLocation(loc):
    DLOC = 0.1
    loc = copy.deepcopy(loc)

    def nonNegativePerturb(a):
        rand = urand(-DLOC, DLOC)
        if a > 0:
            while a + rand <= 0:
                rand = urand(-DLOC, DLOC)
        return a + rand

    for i in range(3):
        loc[i] = nonNegativePerturb(loc[i])

    return loc


def perturbRotation(rot):
    DROT = math.pi / 36
    rot = copy.deepcopy(rot)
    rot[0] += urand(-DROT, DROT)
    rot[1] += urand(-DROT, DROT)
    rot[2] += urand(-DROT, DROT)
    return rot


def perturbCamera(camera):
    new_camera = duplicateObject(camera)
    new_camera.location = perturbLocation(camera.location)
    new_camera.rotation_euler = perturbRotation(camera.rotation_euler)
    return new_camera


def dumpCameraInfo(obj_camera, dump_dir):
    K = get_calibration_matrix_K_from_blender(obj_camera.data)
    K = np.asarray(K)
    
    Rt = get_3x4_RT_matrix_from_blender(obj_camera)
    Rt = np.asarray(Rt)

    fpath = os.path.join(dump_dir, f"{obj_camera.name}.npz")
    np.savez(fpath, K=K, Rt=Rt)


# TODO: Dump MVSTXT format
def dumpPanoCameraInfo(obj_camera, dump_dir):
    # K = get_calibration_matrix_K_from_blender(obj_camera.data)
    # K = np.asarray(K)
    K = np.eye(3)
    Rt = get_3x4_RT_matrix_from_blender(obj_camera)
    Rt = np.asarray(Rt)
    # Rt = np.asarray(obj_camera.matrix_world)[:3]

    fpath = os.path.join(dump_dir, f"{obj_camera.name}.npz")
    np.savez(fpath, K=K, Rt=Rt)


# Point camera to some certain point
def pointCameraTo(camera, focus: Vector, dist: float = None):
    if camera.location == focus:
        print('[Warning] Degenerated case: Camera at the center of the object!')
        return
    looking_direction = camera.location - focus
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    if dist is not None:
        dvec = camera.location - focus
        dvec.normalize()
        camera.location = focus + dist * dvec


def moveObjectAlongPlane(obj, vec):
    rmat = obj.matrix_world.decompose()[1].to_matrix()
    obj.location = obj.location + rmat @ vec


def prepareCameraSet5(base_camera):
    scene = bpy.context.scene
    scene.render.use_multiview = True
    scene.render.views_format = 'MULTIVIEW'

    for i in range(5):
        view = scene.render.views.new(f'view{i}')
        view.camera_suffix = f'_{i}'
        new_camera = duplicateObject(base_camera)
        new_camera.name = f'Camera_{i}'

    # Remove default two views
    scene.render.views.remove(scene.render.views[0])
    scene.render.views.remove(scene.render.views[0])
    bpy.context.view_layer.update()


def prepareCameraSet(base_camera, number=6):
    scene = bpy.context.scene
    scene.render.use_multiview = True
    scene.render.views_format = 'MULTIVIEW'

    for i in range(number):
        view = scene.render.views.new(f'view{i}')
        view.camera_suffix = f'_{i}'
        new_camera = duplicateObject(base_camera)
        new_camera.name = f'Camera_{i}'

    # Remove default two views
    scene.render.views.remove(scene.render.views[0])
    scene.render.views.remove(scene.render.views[0])
    bpy.context.view_layer.update()


# TODO: Anything else to copy?
def copyCameraInfo(*, src, dst):
    dst.location = src.location.copy()
    dst.rotation_euler = src.rotation_euler.copy()
    dst.data.lens = src.data.lens
    bpy.context.view_layer.update()



def random_rotate_camera(camera):
    random_direction = generateRandomUnitBallPos()
    rot_quat = random_direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def placeCameraSet5(base_camera, baseline, dump_dir, random_view=False):
    for idx, (dx, dy) in enumerate([
        (0, 0), (baseline, 0), (0, baseline), (-baseline, 0), (0, -baseline)
    ]):
        camera = bpy.data.objects[f'Camera_{idx}']
        copyCameraInfo(src=base_camera, dst=camera)
        moveObjectAlongPlane(camera, Vector((dx, dy, 0)))
        if random_view:
            random_rotate_camera(camera)
        bpy.context.view_layer.update()
        dumpCameraInfo(camera, dump_dir)

    bpy.context.scene.camera = bpy.data.objects['Camera_0']


def placePanoCameraSet5(base_camera, baseline, dump_dir):
    for idx, (dx, dy) in enumerate([
        (0, 0), (baseline, 0), (0, baseline), (-baseline, 0), (0, -baseline)
    ]):
        camera = bpy.data.objects[f'Camera_{idx}']
        copyCameraInfo(src=base_camera, dst=camera)
        moveObjectAlongPlane(camera, Vector((dx, dy, 0)))
        bpy.context.view_layer.update()
        dumpPanoCameraInfo(camera, dump_dir)

    bpy.context.scene.camera = bpy.data.objects['Camera_0']

def placePanoCameraSet(base_camera, baseline, dump_dir):
    for idx, (dx, dy) in enumerate([
        (0, 0)
    ]):
        camera = bpy.data.objects[f'Camera_{idx}']
        copyCameraInfo(src=base_camera, dst=camera)
        moveObjectAlongPlane(camera, Vector((dx, dy, 0)))
        bpy.context.view_layer.update()
        dumpPanoCameraInfo(camera, dump_dir)

    bpy.context.scene.camera = bpy.data.objects['Camera_0']



def rotateCameraSet(base_camera, baseline, dump_dir):
    # relative to base camera
    # directions = {
    #     'Front': (0, 0, 0),
    #     'Back': (0, 0, math.pi),
    #     'Left': (0, 0, math.pi/2),
    #     'Right': (0, 0, -math.pi/2),
    #     'Up': (-math.pi/2, 0, 0),
    #     'Down': (math.pi/2, 0, 0),
    # }
    directions = {
        'Front': (0, 0, 0),
        'Back': (0, math.pi, 0),
        'Left': (0, math.pi/2, 0),
        'Right': (0, -math.pi/2, 0),
        'Up': (-math.pi/2, 0, 0),
        'Down': (math.pi/2, 0, 0),
    }
    
    idx = 0
    for name, rotation in directions.items():
        camera = bpy.data.objects[f'Camera_{idx}']
        copyCameraInfo(src=base_camera, dst=camera)
        # camera.rotation_euler = (
        #     base_camera.rotation_euler.x + rotation[0],
        #     base_camera.rotation_euler.y + rotation[1],
        #     base_camera.rotation_euler.z + rotation[2]
        # )
        relative_rotation = mathutils.Euler(rotation, 'XYZ').to_quaternion()
        camera.rotation_euler = (base_camera.rotation_euler.to_quaternion() @ relative_rotation).to_euler()
        bpy.context.view_layer.update()
        dumpCameraInfo(camera, dump_dir)
        idx += 1

    bpy.context.scene.camera = bpy.data.objects['Camera_0']


def getFloor(camera_heuristics=None, z_tol=0.2, lower_pctl=0.50, adaptive=True):
    """
    Find object with largest x-y size among lower_pctl pos.z
    If camera heuristics is given, filter the object list with camera x-y
    Maybe: take object rotation into account
    """
    d = {}
    for obj in bpy.data.objects:
        # discard thick objects
        if (obj.type != 'MESH') or (obj.dimensions.z > z_tol):
            continue
        d[obj.name] = obj.location.z
    k = int(len(d) * lower_pctl)
    lower_k = sorted(d.keys(), key=lambda x: d[x])[:k]

    if camera_heuristics is not None:
        lower_k = list(filter(
            lambda x: camera_heuristics['loc'][2] > d[x] and isOnFloor(camera_heuristics['loc'], x) and isOnFloor_2(camera_heuristics['loc'], x),
            lower_k
        ))

    if len(lower_k) > 0:
        best = sorted(
            lower_k,
            key=lambda x: bpy.data.objects[x].dimensions.x * bpy.data.objects[x].dimensions.y
        )[-1]

        return best
    elif adaptive:
        return getFloor(z_tol=z_tol*2, lower_pctl=lower_pctl*2, adaptive=False)
    else:
        return None


def isOnFloor(loc, floor):
    if isinstance(floor, str):
        floor = bpy.data.objects[floor]

    bb0 = floor.matrix_world @ Vector(floor.bound_box[0])
    bb6 = floor.matrix_world @ Vector(floor.bound_box[6])

    x_min = min(bb0[0], bb6[0])
    x_max = max(bb0[0], bb6[0])
    y_min = min(bb0[1], bb6[1])
    y_max = max(bb0[1], bb6[1])

    return x_min <= loc[0] <= x_max \
       and y_min <= loc[1] <= y_max


def getFloor_2(camera_heuristics=None, z_tol=0.2, lower_pctl=0.50, threshold=1.0, adaptive=True):
    """
    Find object with largest x-y size among lower_pctl pos.z
    If camera heuristics is given, filter the object list with camera x-y
    Maybe: take object rotation into account
    """
    d = {}
    for obj in bpy.data.objects:
        # discard thick objects
        if (obj.type != 'MESH') or (obj.dimensions.z > z_tol) or (obj.location.z > z_tol):
            continue
        d[obj.name] = obj.location.z
    k = int(len(d) * lower_pctl)
    lower_k = sorted(d.keys(), key=lambda x: d[x])[:k]

    if camera_heuristics is not None:
        lower_k = list(filter(
            lambda x: camera_heuristics['loc'][2] > d[x] and isOnFloor_2(camera_heuristics['loc'], x, threshold),
            lower_k
        ))

    if len(lower_k) > 0:
        best = sorted(
            lower_k,
            key=lambda x: bpy.data.objects[x].dimensions.x * bpy.data.objects[x].dimensions.y
        )[-1]

        return best
    elif adaptive:
        return getFloor_2(z_tol=z_tol*2, lower_pctl=lower_pctl*2, threshold=threshold*2, adaptive=False)
    else:
        return None


def isOnFloor_2(loc, floor, threshold=1.0):
    if isinstance(floor, str):
        floor = bpy.data.objects[floor]

    bb0 = floor.matrix_world @ Vector(floor.bound_box[0])
    bb6 = floor.matrix_world @ Vector(floor.bound_box[6])

    x_min = min(bb0[0], bb6[0])
    x_max = max(bb0[0], bb6[0])
    y_min = min(bb0[1], bb6[1])
    y_max = max(bb0[1], bb6[1])

    return x_min - threshold <= loc[0] <= x_max + threshold \
       and y_min - threshold <= loc[1] <= y_max + threshold \
       and x_min - threshold <= floor.location.x <= x_max + threshold \
       and y_min - threshold <= floor.location.y <= y_max + threshold \


def calc_hit(box, scene_bvh_tree):
    vertices = []
    polygons = []
    vertex_offset = 0
    box.data.calc_loop_triangles()
    for tri in box.data.loop_triangles:
        vertices.extend([box.matrix_world @ box.data.vertices[i].co for i in tri.vertices])
        polygons.append((vertex_offset, vertex_offset + 1, vertex_offset + 2))
        vertex_offset += 3
    box_bvh_tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True)

    n_hit = len(scene_bvh_tree.overlap(box_bvh_tree))
    return n_hit


# TODO: use BVHTree.raycast for better efficiency
def findTrajectory(
        threshold,
        camera_heuristics,
        floor=None,
        n_step=25,
        len_step=0.1,
        center_pctl=0.7,
        z_mu=1.5, z_sigma=0.1,
        rx_mu=0.5 * np.pi, rx_sigma=0.0,
        ry_mu=0.0, ry_sigma=0.0,
        max_replacement=2999
):
    # Get all existing objects in the scene except the newly created box
    scene_objects = [obj for obj in bpy.context.scene.objects]

    # Extract vertices and polygons from the scene objects
    vertices = []
    polygons = []
    vertex_offset = 0
    for obj in scene_objects:
        if obj.type == 'MESH':
            obj.data.calc_loop_triangles()
            for tri in obj.data.loop_triangles:
                vertices.extend([obj.matrix_world @ obj.data.vertices[i].co for i in tri.vertices])
                polygons.append((vertex_offset, vertex_offset + 1, vertex_offset + 2))
                vertex_offset += 3

    # Create a BVHTree for the scene objects
    bvh_tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True)

    # Create a new box object
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1)
    mesh = bpy.data.meshes.new("Box")
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()

    box = bpy.data.objects.new("Box", mesh)
    bpy.context.collection.objects.link(box)

    # Set the box dimensions
    lx = (n_step - 1) * len_step + 2 * threshold
    ly = 2 * threshold
    lz = 2 * threshold
    box.dimensions = (lx, ly, lz)

    if floor is None:
        floor = getFloor(camera_heuristics=camera_heuristics)
    if isinstance(floor, str):
        floor = bpy.data.objects[floor]
    floor_bvh = BVHTree.FromObject(floor, bpy.context.evaluated_depsgraph_get())

    # use only the center part
    dx_max = floor.dimensions.x * center_pctl / 2
    dy_max = floor.dimensions.y * center_pctl / 2

    def _refresh_pose():
        dx = urand(low=-dx_max, high=dx_max)
        dy = urand(low=-dy_max, high=dy_max)
        box.location.x = floor.location.x + dx
        box.location.y = floor.location.y + dy
        box.location.z = floor.location.z + z_mu + z_sigma * np.random.randn()

        box.rotation_euler.x = rx_mu + rx_sigma * np.random.randn()
        box.rotation_euler.y = ry_mu + ry_sigma * np.random.randn()
        box.rotation_euler.z = urand(low=-np.pi, high=np.pi)

        bpy.context.view_layer.update()

    # Radius within the camera heuristics
    r = max(floor.dimensions) / 2

    best_case = {'n_hit': INTMAX, 'mat_world': None, 'rot': None}

    i_rep = 0
    while i_rep < max_replacement:
        i_rep += 1
        _refresh_pose()

        cond1 = camera_heuristics['loc'].x - r < box.location.x < camera_heuristics['loc'].x + r \
            and camera_heuristics['loc'].y - r < box.location.y < camera_heuristics['loc'].y + r
        cond2 = isOnFloor(box.matrix_world @ Vector(box.bound_box[0]), floor) \
            and isOnFloor(box.matrix_world @ Vector(box.bound_box[6]), floor)

        vertices = []
        polygons = []
        vertex_offset = 0
        box.data.calc_loop_triangles()
        for tri in box.data.loop_triangles:
            vertices.extend([box.matrix_world @ box.data.vertices[i].co for i in tri.vertices])
            polygons.append((vertex_offset, vertex_offset + 1, vertex_offset + 2))
            vertex_offset += 3
        box_bvh_tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True)

        n_hit = len(bvh_tree.overlap(box_bvh_tree))
        cond3 = n_hit == 0

        # print(f'[DEBUG] {n_hit}, {(cond1, cond2, cond3)}')

        if cond1 and cond2 and n_hit < best_case['n_hit']:
            best_case['n_hit'] = n_hit
            best_case['mat_world'] = copy(box.matrix_world)
            best_case['rot'] = copy(box.rotation_euler)

            if cond3:
                break

    if best_case['n_hit'] == 0:
        print(f'[Info] Found perfect trajectory after {i_rep} trials.')
    elif best_case['n_hit'] < INTMAX:
        print(f'[Info] Found suboptimal trajectory after {i_rep} trials, with {best_case["n_hit"]} hits.')
    else:
        raise RuntimeError('No Feasible Trajectory')

    mat_world = best_case['mat_world']
    rot = best_case['rot']
    rot[2] -= np.pi / 2

    # Generate the list of (location, rotation) tuples
    trajectory = []
    for i in range(n_step):
        local_x = (-lx / 2 + threshold + i * len_step) / lx
        local_vec = Vector((local_x, 0, 0))
        world_loc = mat_world @ local_vec
        trajectory.append((world_loc, rot))

    # Delete the box object
    bpy.data.objects.remove(box, do_unlink=True)

    return trajectory, best_case['n_hit']


def findTrajectorySpiral(
        threshold,
        camera_heuristics,
        floor=None,
        z_mu = 1.5,
        z_sigma = 0.1,
        center_pctl = 0.5,
        scale_steps = 100,
        scale_step_size = 0.05,
        trajectory_ratio = 0.7,
        trajectory_steps = 25,
        max_replacement = 1000,
):
    # Get all existing objects in the scene except the newly created box
    scene_objects = [obj for obj in bpy.context.scene.objects]

    # Extract vertices and polygons from the scene objects
    vertices = []
    polygons = []
    vertex_offset = 0
    for obj in scene_objects:
        if obj.type == 'MESH':
            obj.data.calc_loop_triangles()
            for tri in obj.data.loop_triangles:
                vertices.extend([obj.matrix_world @ obj.data.vertices[i].co for i in tri.vertices])
                polygons.append((vertex_offset, vertex_offset + 1, vertex_offset + 2))
                vertex_offset += 3

    # Create a BVHTree for the scene objects
    bvh_tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True)

    # Create a new box object (cylinder)
    import bmesh
    bm = bmesh.new()
    radius = 2 * threshold
    depth = 1 * threshold
    bmesh.ops.create_cone(
        bm,
        cap_ends = True,
        cap_tris = False,
        segments = 32,
        radius1 = radius,
        radius2 = radius,
        depth = depth,
    )
    mesh = bpy.data.meshes.new("Cyliner")
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()

    box = bpy.data.objects.new("Cyliner", mesh)
    bpy.context.collection.objects.link(box)
    
    # floor = getFloor_2(camera_heuristics=camera_heuristics)
    if isinstance(floor, str):
        floor = bpy.data.objects[floor]
    elif floor is None:
        return None

    box.location.x = floor.location.x
    box.location.y = floor.location.y
    box.location.z = floor.location.z + z_mu

    bpy.context.view_layer.update()

    # use only the center part
    def _refresh_pose(center_pctl):
        dx_max = floor.dimensions.x * center_pctl / 2
        dy_max = floor.dimensions.y * center_pctl / 2
        dx = urand(low=-dx_max, high=dx_max)
        dy = urand(low=-dy_max, high=dy_max)
        box.location.x = floor.location.x + dx
        box.location.y = floor.location.y + dy
        box.location.z = floor.location.z + z_mu + z_sigma * np.random.randn()

        bpy.context.view_layer.update()

    # Radius within the camera heuristics
    r = max(floor.dimensions) / 2

    best_case = {'n_hit': INTMAX, 'mat_world': None, 'rot': None}
    
    i_rep = 0
    while i_rep < max_replacement:
        i_rep += 1
        _refresh_pose(center_pctl)
        
        # location on the floor
        success, hit_location, _, _, _, _ = bpy.context.scene.ray_cast(
            bpy.context.view_layer.depsgraph, 
            origin=Vector((box.location.x, box.location.y, floor.location.z+0.5)), 
            direction=Vector((0, 0, -1))
        )
        # print(success, hit_location)
        if not success:
            continue
        
        if i_rep % 100 == 0:
            # threshold /= 2
            threshold -= 0.05
            center_pctl += 0.1
            if threshold < 0.01:
                break
            radius = 2 * threshold
            depth = 1 * threshold
            box.dimensions = (2*radius, 2*radius, depth)
            bpy.context.view_layer.update()
            print(f'[Info] Threshold reduced to {threshold}.')
        
        # bbox inside the floor
        cond_floor = isOnFloor(box.matrix_world @ Vector(box.bound_box[0]), floor) \
            and isOnFloor(box.matrix_world @ Vector(box.bound_box[6]), floor)

        n_hit = calc_hit(box, bvh_tree)

        if cond_floor and n_hit <= best_case['n_hit']:
            best_case['n_hit'] = n_hit
            # best_case['mat_world'] = copy(box.matrix_world)
            # best_case['rot'] = copy(box.rotation_euler)

            if n_hit == 0:
                # further make sure inside the room
                box.hide_set(True)
                ray_directions = [
                    Vector((1, 0, 0)),
                    Vector((-1, 0, 0)),
                    Vector((0, 1, 0)),
                    Vector((0, -1, 0)),
                    Vector((0, 0, 1)),
                ]
                hit_count = 0
                for ray_d in ray_directions:
                    success, hit_location, _, _, _, _ = bpy.context.scene.ray_cast(
                        bpy.context.view_layer.depsgraph, 
                        origin=box.location, 
                        direction=ray_d,
                    )
                    # print(success, hit_location)
                    if success:
                        hit_count += 1
                box.hide_set(False)
                # print(hit_count)
                if hit_count > 3: # flexible
                    break       

    if best_case['n_hit'] == 0:
        print(f'[Info] Found perfect trajectory after {i_rep} trials.')
    elif best_case['n_hit'] < INTMAX:
        raise RuntimeError(f'[Info] Found suboptimal trajectory after {i_rep} trials, with {best_case["n_hit"]} hits.')
    else:
        raise RuntimeError('No Feasible Trajectory')
    
    # scale the radius along x and y axis util hitting the object
    radius_x = radius_y = radius

    for _ in range(scale_steps):
        radius_y += scale_step_size
        box.location.y += scale_step_size
        box.dimensions = (2*radius_x, 2*radius_y, depth)
        bpy.context.view_layer.update()
        n_hit = calc_hit(box, bvh_tree)
        if n_hit > 0:
            radius_y -= scale_step_size
            box.location.y -= scale_step_size
            box.dimensions = (2*radius_x, 2*radius_y, depth)
            bpy.context.view_layer.update()
            break

    for _ in range(scale_steps):
        radius_y += scale_step_size
        box.location.y -= scale_step_size
        box.dimensions = (2*radius_x, 2*radius_y, depth)
        bpy.context.view_layer.update()
        n_hit = calc_hit(box, bvh_tree)
        if n_hit > 0:
            radius_y -= scale_step_size
            box.location.y += scale_step_size
            box.dimensions = (2*radius_x, 2*radius_y, depth)
            bpy.context.view_layer.update()
            break
    
    radius_y = radius_y / 2
    
    for _ in range(scale_steps):
        radius_x += scale_step_size
        box.location.x += scale_step_size
        box.dimensions = (2*radius_x, 2*radius_y, depth)
        bpy.context.view_layer.update()
        n_hit = calc_hit(box, bvh_tree)
        if n_hit > 0:
            radius_x -= scale_step_size
            box.location.x -= scale_step_size
            box.dimensions = (2*radius_x, 2*radius_y, depth)
            bpy.context.view_layer.update()
            break

    for _ in range(scale_steps):
        radius_x += scale_step_size
        box.location.x -= scale_step_size
        box.dimensions = (2*radius_x, 2*radius_y, depth)
        bpy.context.view_layer.update()
        n_hit = calc_hit(box, bvh_tree)
        if n_hit > 0:
            radius_x -= scale_step_size
            box.location.x += scale_step_size
            box.dimensions = (2*radius_x, 2*radius_y, depth)
            bpy.context.view_layer.update()
            break
    
    radius_x = radius_x / 2
    ratio = radius_y / radius_x + 1e-6
    
    for _ in range(scale_steps):
        radius_x += scale_step_size / 2
        radius_y += scale_step_size * ratio / 2
        box.dimensions = (2*radius_x, 2*radius_y, depth)
        bpy.context.view_layer.update()
        n_hit = calc_hit(box, bvh_tree)
        if n_hit > 0:
            radius_x -= scale_step_size
            radius_y -= scale_step_size
            box.dimensions = (2*radius_x, 2*radius_y, depth)
            bpy.context.view_layer.update()
            break
    
    for _ in range(scale_steps):
        depth += scale_step_size / 5
        box.dimensions = (2*radius_x, 2*radius_y, depth)
        bpy.context.view_layer.update()
        n_hit = calc_hit(box, bvh_tree)
        if n_hit > 0:
            depth -= scale_step_size / 5
            box.dimensions = (2*radius_x, 2*radius_y, depth)
            bpy.context.view_layer.update()
            break

    box_loc = copy(box.location)
    box_mat_world = copy(box.matrix_world)
    # Delete the box object
    bpy.data.objects.remove(box, do_unlink=True)

    # box.hide_set(True)
    ray_directions = [
        Vector((1, 0, 0)),
        Vector((-1, 0, 0)),
        Vector((0, 1, 0)),
        Vector((0, -1, 0)),
        Vector((0, 0, 1)),
        Vector((0, 0, -1)),
    ]
    hit_count = 0
    for ray_d in ray_directions:
        success, hit_location, _, _, _, _ = bpy.context.scene.ray_cast(
            bpy.context.view_layer.depsgraph, 
            origin=box_loc,
            direction=ray_d,
        )
        if success:
            hit_count += 1
    if hit_count < 4:  # flexible
        raise RuntimeError('Trajectory out of bound')
    # box.hide_set(False)
    
    # Generate the list of (location, rotation) tuples
    trajectory = [] # spiral trajectory

    # center
    local_vec = Vector((0,0,0))
    world_loc = box_mat_world @ local_vec
    rot = Vector((1,0,0)).to_track_quat('Z', 'Y').to_euler()
    trajectory.append((world_loc, rot))

    # bm = bmesh.new()
    # bmesh.ops.create_cube(bm, size=0.1)
    # mesh = bpy.data.meshes.new("Cube")
    # bm.to_mesh(mesh)
    # mesh.update()
    # bm.free()
    # cube = bpy.data.objects.new("Cube", mesh)
    # bpy.context.collection.objects.link(cube)
    # cube.location = world_loc

    n_step = trajectory_steps
    for i in range(n_step):
        local_x = (1 - 0.75 * i/(n_step-1)) * np.cos(4 * np.pi * i / (n_step-1)) * radius_x
        local_y = (1 - 0.75 * i/(n_step-1)) * np.sin(4 * np.pi * i / (n_step-1)) * radius_y
        local_z = (1 * i/(n_step-1) - 0.5) * depth
        local_vec = trajectory_ratio * Vector((local_x, local_y, local_z))
        world_loc = box_loc + local_vec
        # always point to the center
        rot = local_vec.to_track_quat('Z', 'Y').to_euler()
        trajectory.append((world_loc, rot))

        # bm = bmesh.new()
        # bmesh.ops.create_cube(bm, size=0.1)
        # mesh = bpy.data.meshes.new(f"Cube_{i}")
        # bm.to_mesh(mesh)
        # mesh.update()
        # bm.free()
        # cube = bpy.data.objects.new(f"Cube_{i}", mesh)
        # bpy.context.collection.objects.link(cube)
        # cube.location = world_loc
        # cube.rotation_euler = rot
    
    for i in range(n_step-1):
        local_x = - np.sin(2 * np.pi * i / (n_step-2)) * radius_x
        local_y = - np.cos(2 * np.pi * i / (n_step-2)) * radius_y
        local_z = - (i/(n_step-2) - 0.5) * depth
        local_vec = trajectory_ratio * Vector((local_x, local_y, local_z))
        world_loc = box_loc + local_vec
        # always point to the center
        rot = local_vec.to_track_quat('Z', 'Y').to_euler()
        trajectory.append((world_loc, rot))

        # bm = bmesh.new()
        # bmesh.ops.create_cube(bm, size=0.1)
        # mesh = bpy.data.meshes.new(f"Cube_{i+n_step}")
        # bm.to_mesh(mesh)
        # mesh.update()
        # bm.free()
        # cube = bpy.data.objects.new(f"Cube_{i+n_step}", mesh)
        # bpy.context.collection.objects.link(cube)
        # cube.location = world_loc
        # cube.rotation_euler = rot

    return trajectory, best_case['n_hit']


# The rendered depth is wrt the pinhole, not the image plane.
def buildMainNodeTree(scene_outdir, save_thumbnails=False, exr_path=None):
    scene = bpy.context.scene
    nodes = scene.node_tree.nodes
    nodes.clear()

    bpy.context.view_layer.use_pass_combined = True
    bpy.context.view_layer.use_pass_z = True
    bpy.context.view_layer.use_pass_normal = True

    view_layer_cycles_setting = bpy.context.scene.cycles
    view_layer_cycles_setting.use_denoising = True
    view_layer_cycles_setting.denoiser = 'OPENIMAGEDENOISE'
    view_layer_cycles_setting.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    view_layer_cycles_setting.denoising_prefilter = 'ACCURATE'

    render_layers = nodes.new("CompositorNodeRLayers")
    file_output = nodes.new("CompositorNodeOutputFile")
    form = file_output.format
    form.file_format = 'OPEN_EXR_MULTILAYER'
    form.exr_codec = 'ZIP'
    form.color_mode = 'RGB'
    # form.views_format = 'MULTIVIEW'    # This will cause bug
    file_output.base_path = os.path.join(scene_outdir, './rendered/')

    file_output.layer_slots.clear()
    file_output.layer_slots.new('rgb')
    file_output.layer_slots.new('depth')
    file_output.layer_slots.new('normal')
    file_output.layer_slots.new('albedo')

    scene.node_tree.links.new(
        render_layers.outputs['Image'],
        file_output.inputs['rgb']
    )

    scene.node_tree.links.new(
        render_layers.outputs['Depth'],
        file_output.inputs['depth']
    )

    scene.node_tree.links.new(
        render_layers.outputs['Normal'],
        file_output.inputs['normal']
    )

    if save_thumbnails:
        img_thumbnail_output = nodes.new("CompositorNodeOutputFile")
        form = img_thumbnail_output.format
        form.file_format = 'JPEG'
        form.color_mode = 'RGB'
        form.compression = 100
        form.quality = 60
        img_thumbnail_output.base_path = os.path.join(scene_outdir, 'thumbnails/image')

        scene.node_tree.links.new(
            render_layers.outputs['Image'],
            img_thumbnail_output.inputs['Image']
        )

        depth_thumbnail_output = nodes.new("CompositorNodeOutputFile")
        form = depth_thumbnail_output.format
        form.file_format = 'JPEG'
        form.color_mode = 'BW'
        form.compression = 100
        form.quality = 60
        depth_thumbnail_output.base_path = os.path.join(scene_outdir, 'thumbnails/depth')

        depth_normalize_node = nodes.new("CompositorNodeNormalize")

        scene.node_tree.links.new(
            render_layers.outputs['Depth'],
            depth_normalize_node.inputs[0]
        )

        scene.node_tree.links.new(
            depth_normalize_node.outputs[0],
            depth_thumbnail_output.inputs['Image']
        )

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
            background_node.inputs["Strength"].default_value = 0.1

            # display bg
            bpy.context.scene.render.film_transparent = False 

            print("Blender has created shaping tree and new envmap")

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio
    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels
    K = Matrix(
        ((s_u, skew, u_0),
         (0, s_v, v_0),
         (0, 0, 1)))
    return K


# # TODO: Fix possible numerical error in this function
# # Returns camera rotation and translation matrices from Blender.
# #
# # There are 3 coordinate systems involved:
# #    1. The World coordinates: "world"
# #       - right-handed
# #    2. The Blender camera coordinates: "bcam"
# #       - x is horizontal
# #       - y is up
# #       - right-handed: negative z look-at direction
# #    3. The desired computer vision camera coordinates: "cv"
# #       - x is horizontal
# #       - y is down (to align to the actual pixel coordinates
# #         used in digital images)
# #       - right-handed: positive z look-at direction
# def get_3x4_RT_matrix_from_blender(cam):
#     # bcam stands for blender camera
#     R_bcam2cv = Matrix(
#         ((1, 0, 0),
#          (0, -1, 0),
#          (0, 0, -1)))

#     # Transpose since the rotation is object rotation,
#     # and we want coordinate rotation
#     # Use matrix_world instead to account for all constraints
#     location, rotation = cam.matrix_world.decompose()[0:2]
#     R_world2bcam = rotation.to_matrix().transposed()

#     # Convert camera location to translation vector used in coordinate changes
#     # Use location from matrix_world to account for constraints:
#     T_world2bcam = -1 * R_world2bcam @ location

#     # Build the coordinate transform matrix from world to computer vision camera
#     R_world2cv = R_bcam2cv @ R_world2bcam
#     T_world2cv = R_bcam2cv @ T_world2bcam

#     # put into 3x4 matrix
#     RT = Matrix((
#         R_world2cv[0][:] + (T_world2cv[0],),
#         R_world2cv[1][:] + (T_world2cv[1],),
#         R_world2cv[2][:] + (T_world2cv[2],)
#     ))
#     return RT

def get_3x4_RT_matrix_from_blender(cam):
    ## save camera RT matrix (C2W)
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = rotation.to_matrix()
    T = np.array(location).reshape(-1, 1)
    RT = np.hstack((R, T))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K * RT, K, RT
