import os
import sys
import logging
from pathlib import Path

import numpy as np
import bpy

sys.path.append(os.path.dirname(bpy.data.filepath))

from blender.src.render import render
from blender.src.tools import delete_objs

import json
import logging
import os
from pathlib import Path
import numpy as np
import json

logger = logging.getLogger(__name__)


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def load_transform(transform_p):
    data = load_json(transform_p)
    frames = data['frames']
    transform = []
    for item in frames:
        transform.append(np.array(item['transform_matrix']))
    c2ws = np.stack(transform, axis=0)
    c2ws[:,:,0] = -c2ws[:,:,0]
    c2ws[:,:,1] = -c2ws[:,:,1]
    ref_w2c = np.linalg.inv(c2ws[:1])
    ref_w2c_repeated = np.repeat(ref_w2c, c2ws.shape[0], axis=0)
    c2ws = np.matmul(ref_w2c_repeated, c2ws)[:, :3, :]
    T_norm = np.linalg.norm(c2ws[:, :3, 3], axis=-1).max()
    scale = T_norm + 1e-5
    c2ws[:, :3, 3] /= scale
 
    c2ws[:,:,0] = -c2ws[:,:,0]
    c2ws[:,:,2] = -c2ws[:,:,2]
    c2ws[:,:3,3] = c2ws[:,:3,3] * 5

    return c2ws

def get_meshes_bounds(mesh_objects):
    all_vertices = []
    for obj in mesh_objects:
        if obj.type == 'MESH':
            all_vertices.extend([obj.matrix_world @ v.co for v in obj.data.vertices])

    all_vertices = np.array(all_vertices)
    min_xyz = np.min(all_vertices, axis=0)
    max_xyz = np.max(all_vertices, axis=0)
    return min_xyz, max_xyz

def get_best_camera_position(mesh_objects, scale_factor=1.0):
    min_xyz, max_xyz = get_meshes_bounds(mesh_objects)
    center = (min_xyz + max_xyz) / 2
    bbox_size = max_xyz - min_xyz

    max_extent = np.max(bbox_size)
    camera_distance = max_extent * scale_factor
    if center[0]<0:
        scale = -1
    else:
        scale = 1
    camera_position = center + np.array([scale*camera_distance, camera_distance, 0])

    return camera_position, center

def look_at_rotation(direction, up=np.array([0, 1, 0])):
    direction = np.array(direction)
    direction /= np.linalg.norm(direction)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    new_up = np.cross(direction, right)
    rotation_matrix = np.array([right, new_up, direction]).T  
    return rotation_matrix

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def look_at_rotation(camera_pos, target_pos, up=np.array([0, 0, 1])):
    direction = normalize(np.array(target_pos) - np.array(camera_pos))
    right = normalize(np.cross(direction, up))
    new_up = np.cross(right, direction)
    return np.array([right, new_up, -direction]).T

def rotation_matrix_to_euler(matrix):
    sy = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(matrix[2, 1], matrix[2, 2])
        y = np.arctan2(-matrix[2, 0], sy)
        z = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x = np.arctan2(-matrix[1, 2], matrix[1, 1])
        y = np.arctan2(-matrix[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def setup_camera(mesh_objects):
    camera = bpy.data.objects.get("Camera")
    if camera is None:
        camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(camera)
    
    camera_position, look_at = get_best_camera_position(mesh_objects)
    
    camera_position[0] = look_at[0]
    camera_position[1] = look_at[1] + 5
    camera_position[2] = look_at[2] + 10
    
    camera.location = camera_position

    rotation_matrix = look_at_rotation(camera_position, look_at)

    euler_angles = rotation_matrix_to_euler(rotation_matrix)
    camera.rotation_euler = euler_angles
    
    bpy.context.scene.camera = camera
    return camera

def set_plane_color(plane_name, color):
    plane = bpy.data.objects.get(plane_name)
    
    if plane:
        material = bpy.data.materials.new(name="PlaneMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (*color, 1)
        
        if plane.data.materials:
            plane.data.materials[0] = material
        else:
            plane.data.materials.append(material)

    else:
        print(f"Plane named {plane_name} not found.")
        
class Renderer:
    def __init__(self):
        self.obj_names = None

    def render_cli(self, input_dir, sample_id, selected_rate, mode, traj_p):
        if self.obj_names is not None:
            delete_objs(self.obj_names)
            delete_objs(["Plane", "myCurve", "Cylinder"])

        input_dir = Path(input_dir)

        char_path = input_dir / "vert_raw" / f"{sample_id}.npy"
        cam_seg_path = input_dir / "cam_segments" / f"{sample_id}.npy"
        char_seg_path = input_dir / "char_segments" / f"{sample_id}.npy"

        cam_segments = np.load(cam_seg_path, allow_pickle=True)
        cam_segments = np.concatenate([[cam_segments[0]], cam_segments])
        cam_segments = np.concatenate([cam_segments]*5,axis=0)
        char_segments = np.load(char_seg_path, allow_pickle=True)
        char_segments = np.concatenate([[char_segments[0]], char_segments])
        char_segments = np.concatenate([char_segments]*5,axis=0)

        traj = load_transform(traj_p)
        traj = traj[:, [0, 2, 1]]
        traj[:, 2] = -traj[:, 2]
        char = np.load(char_path, allow_pickle=True)[()]
        vertices = char["vertices"]
        vertices = vertices[..., [0, 2, 1]]
        vertices[..., 2] = -vertices[..., 2]
        faces = char["faces"]
        faces = faces[..., [0, 2, 1]]

        nframes = traj.shape[0]
        if "video" in mode:
            bpy.context.scene.frame_end = nframes - 1
        num = int(selected_rate * nframes)
        self.obj_names = render(
            traj=traj,
            vertices=vertices,
            faces=faces,
            cam_segments=cam_segments,
            char_segments=char_segments,
            denoising=True,
            oldrender=True,
            res="low",
            canonicalize=True,
            exact_frame=0.5,
            num=num,
            mode=mode,
            init=False,
        )
       
        mesh_objects = [bpy.data.objects[name] for name in self.obj_names]
        setup_camera(mesh_objects)
        scale = get_meshes_bounds(mesh_objects)
        center_position = (scale[0] + scale[1]) / 2  # 计算两个点之间的中心位置
        plane_scale = (scale[1] - scale[0]) / 2 * 1.2  # 计算平面的缩放比例   
        '''set plane_scale'''
        # plane_scale[0] = plane_scale[0]*5
        # plane_scale[1] = plane_scale[1]*1.5
        # plane_scale[0] = plane_scale[0]*2
        center_position[2] = 0
        plane_scale[2] = 1

        '''set plane position'''
        h_pos = 0
        for obj in bpy.data.objects:
            if obj.name == "BigPlane":
                obj.scale = (0.01, 0.01, 0.01)
                obj.location[2] = -0.01 + h_pos
                set_plane_color("BigPlane", (1, 1, 1))
            if obj.name == "SmallPlane":
                obj.scale = plane_scale
                obj.location = center_position
                obj.location[2] = -0.0 + h_pos
            
            '''set color'''
            # if 'cam' in obj.name:
            #     idx = int(obj.name.split('_')[0])
            #     if idx >= 0 and idx <= 30: 
            #         set_plane_color(obj.name, (192/256, 0/256, 0/256))
            #     elif idx < 60:
            #         set_plane_color(obj.name, (250/256, 150/256, 2/256))
            #     elif idx < 90:
            #         set_plane_color(obj.name, (80/256, 158/256, 50/256))
            #     else:
            #         set_plane_color(obj.name, (31/256, 78/256, 121/256))
            

        bpy.context.scene.render.filepath = os.path.splitext(traj_p)[0] + "-rgba.png"

        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":    
    input_dir = 'demo'
    sample_id = '2011_F_EuMeT2wBo_00014_00001'
    selected_rate = 0.2
    mode = 'image'
    
    traj_p = "./vis/case1.json"
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.render.resolution_percentage = 400
    renderer = Renderer()
    renderer.render_cli(input_dir, sample_id, selected_rate, mode, traj_p)
