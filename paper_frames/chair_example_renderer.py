import numpy as np
import sys
import os
import glob
import argparse
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
    # XML template for a single point (ball) in the scene
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="{radius}"/>
        <transform name="toWorld">
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
"""
    # XML template for the ground plane and the background plane
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
            <lookat origin="0,0,14" target="0,0,0" up="0,1,0"/>
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

    def __init__(self, file_path, output_folder=None, radius=0.015):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder
        self.radius = radius

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
        
        # Calculate floor position: slightly below the lowest point
        floor_z = pcl_min[2] - 0.05  # 0.05 units below the lowest point
        
        for idx, point in enumerate(pcl):
            normalized_point = (point - pcl_min) / (pcl_range + 1e-8)
            color = self.compute_color(
                normalized_point[0], normalized_point[1], normalized_point[2], 
                noise_seed=idx)
            xml_segments.append(self.XML_BALL_SEGMENT.format(
                radius=self.radius,
                x=point[0], y=point[1], z=point[2],
                r=color[0], g=color[1], b=color[2]))
        xml_segments.append(self.XML_TAIL.format(floor_z=floor_z))
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


def main():
    parser = argparse.ArgumentParser(description='Batch render PLY point clouds with Mitsuba (chair preset)')
    parser.add_argument('--input', type=str, default='ply',
                        help='Input folder containing .ply files (default: ply)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder for PNGs (default: <input>_rendered)')
    parser.add_argument('--radius', type=float, default=0.015,
                        help='Ball radius (default: 0.015)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Only render specific pts indices, e.g. --indices 3 56 145')
    parser.add_argument('--test', type=int, default=None,
                        help='Quick test: only render first N files')
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output or f'{input_folder.rstrip("/")}_rendered'

    if not os.path.isdir(input_folder):
        print(f'Error: input folder not found: {input_folder}')
        return

    PointCloudRenderer.init_mitsuba_variant()
    print('=' * 60)

    os.makedirs(output_folder, exist_ok=True)

    if args.indices is not None:
        target_files = [f'pts_{i}.ply' for i in args.indices]
        ply_files = []
        for tf in target_files:
            fp = os.path.join(input_folder, tf)
            if os.path.isfile(fp):
                ply_files.append(fp)
            else:
                print(f'Warning: file not found: {fp}')
    else:
        all_ply = [f for f in os.listdir(input_folder) if f.endswith('.ply')]
        all_ply.sort(key=lambda x: int(x.replace('pts_', '').replace('.ply', ''))
                     if x.startswith('pts_') else x)
        ply_files = [os.path.join(input_folder, f) for f in all_ply]

    if args.test is not None:
        ply_files = ply_files[:args.test]

    if not ply_files:
        print(f'No .ply files found in: {input_folder}')
        return

    total_files = len(ply_files)
    print(f'Input:  {input_folder} ({total_files} files to render)')
    print(f'Output: {output_folder}')
    print(f'Radius: {args.radius}')
    print('=' * 60)

    for idx, ply_file in enumerate(ply_files, 1):
        print(f'\n[{idx}/{total_files}] ({idx*100//total_files}%) Processing: {os.path.basename(ply_file)}')
        print('-' * 60)
        try:
            renderer = PointCloudRenderer(ply_file, output_folder=output_folder,
                                          radius=args.radius)
            renderer.process()
            print(f'✓ Successfully processed: {os.path.basename(ply_file)}')
        except Exception as e:
            print(f'✗ Error processing {os.path.basename(ply_file)}: {str(e)}')

    print('\n' + '=' * 60)
    print(f'Batch processing completed! Processed {total_files} files.')
    print(f'Output files saved to: {output_folder}/')


if __name__ == '__main__':
    main()