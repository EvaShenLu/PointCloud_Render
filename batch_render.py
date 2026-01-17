#!/usr/bin/env python3
"""
批量渲染 to_render 文件夹下的所有点云文件
- 使用 example_renderer.py 渲染包含 airplane 或 car 的文件夹
- 使用 chair_example_renderer.py 渲染包含 chair 的文件夹
"""
import numpy as np
import sys
import os
import glob
import re
from plyfile import PlyData
import mitsuba as mi


# ========== 从 example_renderer.py 导入的渲染器 ==========
class XMLTemplatesStandard:
    HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="2,2,2" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="30"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.1"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>
"""
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>

    <bsdf type="diffuse">
        <rgb name="reflectance" value="{},{},{}"/>
    </bsdf>

    </shape>
"""
    TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.2"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="8" y="8" z="1"/>
            <lookat origin="0,0,15" target="0,0,0" up="0,1,0"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="4,4,4"/>
        </emitter>
    </shape>
</scene>
"""


# ========== 从 chair_example_renderer.py 导入的渲染器 ==========
class XMLTemplatesChair:
    HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="2.0,2.0,2.2" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="30"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1440"/>
            <integer name="height" value="1440"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.1"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>
"""
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>

    <bsdf type="diffuse">
        <rgb name="reflectance" value="{},{},{}"/>
    </bsdf>

    </shape>
"""
    TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="{floor_z}"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="8" y="8" z="1"/>
            <lookat origin="0,0,15" target="0,0,0" up="0,1,0"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="4,4,4"/>
        </emitter>
    </shape>
</scene>
"""


class PointCloudRenderer:
    def __init__(self, file_path, output_folder=None, use_chair_renderer=False):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder
        self.use_chair_renderer = use_chair_renderer
        
        # 根据渲染器类型选择模板
        if use_chair_renderer:
            self.XML_HEAD = XMLTemplatesChair.HEAD
            self.XML_BALL_SEGMENT = XMLTemplatesChair.BALL_SEGMENT
            self.XML_TAIL = XMLTemplatesChair.TAIL
        else:
            self.XML_HEAD = XMLTemplatesStandard.HEAD
            self.XML_BALL_SEGMENT = XMLTemplatesStandard.BALL_SEGMENT
            self.XML_TAIL = XMLTemplatesStandard.TAIL

    @staticmethod
    def compute_color(x, y, z, noise_seed=0):
        g = 0.3 
        return np.array([g, g, g])

    @staticmethod
    def standardize_point_cloud(pcl):
        center = np.mean(pcl, axis=0)
        scale = np.amax(pcl - np.amin(pcl, axis=0))
        return ((pcl - center) / scale).astype(np.float32)

    def load_point_cloud(self):
        file_extension = os.path.splitext(self.file_path)[1]
        if file_extension == '.npy':
            return np.load(self.file_path, allow_pickle=True)
        elif file_extension == '.npz':
            return np.load(self.file_path)['pred']
        elif file_extension == '.ply':
            ply_data = PlyData.read(self.file_path)
            return np.column_stack([ply_data['vertex'][t] for t in ('x', 'y', 'z')])
        else:
            raise ValueError('Unsupported file format.')

    def generate_xml_content(self, pcl):
        xml_segments = [self.XML_HEAD]
        pcl_min = np.min(pcl, axis=0)
        pcl_max = np.max(pcl, axis=0)
        pcl_range = pcl_max - pcl_min
        
        # Chair renderer 需要计算 floor_z
        if self.use_chair_renderer:
            floor_z = pcl_min[2] - 0.05
            tail_template = self.XML_TAIL.format(floor_z=floor_z)
        else:
            tail_template = self.XML_TAIL
        
        for idx, point in enumerate(pcl):
            normalized_point = (point - pcl_min) / (pcl_range + 1e-8)
            color = self.compute_color(
                normalized_point[0], normalized_point[1], normalized_point[2], 
                noise_seed=idx)
            xml_segments.append(self.XML_BALL_SEGMENT.format(
                point[0], point[1], point[2], *color))
        xml_segments.append(tail_template)
        return ''.join(xml_segments)

    @staticmethod
    def save_xml_content_to_file(output_file_path, xml_content):
        xml_file_path = f'{output_file_path}.xml'
        with open(xml_file_path, 'w') as f:
            f.write(xml_content)
        return xml_file_path

    @staticmethod
    def init_mitsuba_variant():
        try:
            mi.set_variant('cuda_ad_rgb')
            print('Using CUDA GPU (cuda_ad_rgb)')
            return True
        except:
            try:
                mi.set_variant('cuda_rgb')
                print('Using CUDA GPU (cuda_rgb)')
                return True
            except:
                mi.set_variant('scalar_rgb')
                print('Using CPU (scalar_rgb) - GPU not available')
                return False

    @staticmethod
    def render_scene(xml_file_path):
        scene = mi.load_file(xml_file_path)
        img = mi.render(scene)
        return img

    @staticmethod
    def save_scene(output_file_path, rendered_scene):
        mi.util.write_bitmap(f'{output_file_path}.png', rendered_scene)

    def process(self):
        pcl_data = self.load_point_cloud()
        if len(pcl_data.shape) < 3:
            pcl_data = pcl_data[np.newaxis, :, :]

        total_frames = len(pcl_data)
        for index, pcl in enumerate(pcl_data):
            pcl = self.standardize_point_cloud(pcl)
            pcl = pcl[:, [2, 0, 1]]
            pcl[:, 0] *= -1
            pcl[:, 2] += 0.0125

            output_filename = f'{self.filename}'
            if self.output_folder:
                os.makedirs(self.output_folder, exist_ok=True)
                output_file_path = os.path.join(self.output_folder, output_filename)
            else:
                output_file_path = os.path.join(self.folder, output_filename)
            
            if total_frames > 1:
                print(f'  Frame {index+1}/{total_frames}: Generating XML...', end=' ', flush=True)
            else:
                print(f'  Generating XML...', end=' ', flush=True)
            
            xml_content = self.generate_xml_content(pcl)
            xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
            
            print('Rendering...', end=' ', flush=True)
            rendered_scene = self.render_scene(xml_file_path)
            
            print('Saving...', end=' ', flush=True)
            self.save_scene(output_file_path, rendered_scene)
            
            if os.path.exists(xml_file_path):
                os.remove(xml_file_path)
            
            print('Done!')


def determine_renderer_type(folder_name):
    """根据文件夹名称确定使用哪个渲染器"""
    folder_name_lower = folder_name.lower()
    if 'chair' in folder_name_lower:
        return True  # 使用 chair renderer
    elif 'airplane' in folder_name_lower or 'car' in folder_name_lower:
        return False  # 使用 standard renderer
    else:
        # 默认使用 standard renderer
        return False


def main():
    PointCloudRenderer.init_mitsuba_variant()
    print('=' * 60)
    
    # 使用脚本所在目录作为基础目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_base = os.path.join(base_dir, 'to_render')
    output_base = os.path.join(base_dir, 'render_results')
    
    # 获取所有文件夹（排除 render_indices.txt 等文件）
    folders = [f for f in os.listdir(input_base) 
               if os.path.isdir(os.path.join(input_base, f)) and not f.startswith('.')]
    
    if not folders:
        print(f'No folders found in: {input_base}')
        return
    
    print(f'Found {len(folders)} folder(s) to process')
    print('=' * 60)
    
    total_processed = 0
    total_errors = 0
    
    for folder_idx, folder_name in enumerate(sorted(folders), 1):
        folder_path = os.path.join(input_base, folder_name)
        ply_folder = os.path.join(folder_path, 'ply')
        
        # 确定渲染器类型
        use_chair_renderer = determine_renderer_type(folder_name)
        renderer_type = "Chair Renderer" if use_chair_renderer else "Standard Renderer"
        
        print(f'\n[{folder_idx}/{len(folders)}] Processing folder: {folder_name}')
        print(f'  Renderer type: {renderer_type}')
        print('-' * 60)
        
        # 查找所有 .ply 文件
        if os.path.exists(ply_folder):
            ply_files = sorted(glob.glob(os.path.join(ply_folder, '*.ply')))
        else:
            # 如果没有 ply 子文件夹，直接在文件夹中查找
            ply_files = sorted(glob.glob(os.path.join(folder_path, '*.ply')))
        
        if not ply_files:
            print(f'  Warning: No .ply files found in {folder_path}')
            continue
        
        print(f'  Found {len(ply_files)} .ply file(s)')
        
        # 创建输出文件夹
        output_folder = os.path.join(output_base, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # 处理每个文件
        folder_processed = 0
        folder_errors = 0
        
        for file_idx, ply_file in enumerate(ply_files, 1):
            file_name = os.path.basename(ply_file)
            print(f'\n  [{file_idx}/{len(ply_files)}] Processing: {file_name}')
            try:
                renderer = PointCloudRenderer(ply_file, output_folder=output_folder, 
                                             use_chair_renderer=use_chair_renderer)
                renderer.process()
                folder_processed += 1
                total_processed += 1
            except Exception as e:
                print(f'  ✗ Error processing {file_name}: {str(e)}')
                folder_errors += 1
                total_errors += 1
        
        print(f'\n  Folder summary: {folder_processed} succeeded, {folder_errors} failed')
    
    print('\n' + '=' * 60)
    print(f'Batch processing completed!')
    print(f'  Total processed: {total_processed} files')
    print(f'  Total errors: {total_errors} files')
    print(f'  Output saved to: {output_base}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
