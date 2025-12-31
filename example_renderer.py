import numpy as np
import sys
import os
import glob
from plyfile import PlyData
import mitsuba as mi


class XMLTemplates:
    # XML template for the scene (camera, sampler, surface material, etc.)
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
    # XML template for a single point (ball) in the scene
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.01"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>

    <bsdf type="diffuse">
        <rgb name="reflectance" value="{},{},{}"/>
    </bsdf>

    </shape>
"""
    # XML template for the ground plane and the background plane
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


class PointCloudRenderer:
    XML_HEAD = XMLTemplates.HEAD
    XML_BALL_SEGMENT = XMLTemplates.BALL_SEGMENT
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path, output_folder=None):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder

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
        pcl_center = (pcl_min + pcl_max) / 2.0
        
        for idx, point in enumerate(pcl):
            normalized_point = (point - pcl_min) / (pcl_range + 1e-8)
            color = self.compute_color(
                normalized_point[0], normalized_point[1], normalized_point[2], 
                noise_seed=idx)
            xml_segments.append(self.XML_BALL_SEGMENT.format(
                point[0], point[1], point[2], *color))
        xml_segments.append(self.XML_TAIL)
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


def main(argv):
    PointCloudRenderer.init_mitsuba_variant()
    print('=' * 60)
    
    input_folder = 'val_ply_airplane_pcd'
    output_folder = 'render'
    
    start_idx = 0
    end_idx = 404
    target_files = [f'{i:06d}.ply' for i in range(start_idx, end_idx + 1)]
    
    os.makedirs(output_folder, exist_ok=True)
    
    ply_files = []
    for target_file in target_files:
        file_path = os.path.join(input_folder, target_file)
        if os.path.isfile(file_path):
            ply_files.append(file_path)
        else:
            print(f'Warning: File not found: {file_path}')
    
    if not ply_files:
        print(f'No target files found in folder: {input_folder}')
        print(f'Looking for: {target_files}')
        return
    
    total_files = len(ply_files)
    print(f'Found {total_files} target file(s) in folder: {input_folder}')
    print(f'Output folder: {output_folder}')
    print('=' * 60)
    
    for idx, ply_file in enumerate(ply_files, 1):
        print(f'\n[{idx}/{total_files}] ({idx*100//total_files}%) Processing: {os.path.basename(ply_file)}')
        print('-' * 60)
        try:
            renderer = PointCloudRenderer(ply_file, output_folder=output_folder)
            renderer.process()
            print(f'✓ Successfully processed: {os.path.basename(ply_file)}')
        except Exception as e:
            print(f'✗ Error processing {os.path.basename(ply_file)}: {str(e)}')
    
    print('\n' + '=' * 60)
    print(f'Batch processing completed! Processed {total_files} files.')
    print(f'Output files saved to: {output_folder}/')


if __name__ == '__main__':
    main(sys.argv)
