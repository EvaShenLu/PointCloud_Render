import numpy as np
import sys
import os
from plyfile import PlyData
import mitsuba as mi


class XMLTemplates:
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
    DROPLET_SEGMENT = """
    <shape type="obj">
        <string name="filename" value="{}"/>
        <transform name="toWorld">
            <matrix value="{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}"/>
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


class TrajectoryRenderer:
    XML_HEAD = XMLTemplates.HEAD
    XML_DROPLET_SEGMENT = XMLTemplates.DROPLET_SEGMENT
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path, output_folder=None, droplet_mesh_path=None):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder
        self.droplet_mesh_path = droplet_mesh_path or self._create_droplet_mesh()

    @staticmethod
    def _create_droplet_mesh():
        temp_dir = 'temp_meshes'
        os.makedirs(temp_dir, exist_ok=True)
        mesh_path = os.path.join(temp_dir, 'droplet.obj')
        
        if os.path.exists(mesh_path):
            os.remove(mesh_path)
        
        n_segments = 20
        n_rings = 16
        base_radius = 0.008
        length = 0.035
        
        vertices = []
        faces = []
        
        for i in range(n_rings + 1):
            theta = np.pi * i / n_rings
            for j in range(n_segments):
                phi = 2 * np.pi * j / n_segments
                
                if theta <= np.pi / 3:
                    r = base_radius
                    z_offset = 0
                else:
                    t = (theta - np.pi / 3) / (2 * np.pi / 3)
                    r = base_radius * (1 - t) ** 2
                    z_offset = -length * t * 0.8
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta) + z_offset
                
                vertices.append([x, y, z])
        
        for i in range(n_rings):
            for j in range(n_segments):
                v0 = i * n_segments + j
                v1 = i * n_segments + (j + 1) % n_segments
                v2 = (i + 1) * n_segments + j
                v3 = (i + 1) * n_segments + (j + 1) % n_segments
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])
        
        with open(mesh_path, 'w') as f:
            for v in vertices:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
            for face in faces:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
        
        return os.path.abspath(mesh_path)

    @staticmethod
    def compute_color():
        return np.array([0.3, 0.3, 0.3])

    @staticmethod
    def generate_rotation_matrix_from_velocity(velocity, translation):
        """根据速度向量生成旋转矩阵，使水滴指向速度方向"""
        velocity = np.array(velocity)
        vel_norm = np.linalg.norm(velocity)
        
        if vel_norm < 1e-6:
            # 如果速度为零，使用单位矩阵（不旋转）
            matrix = np.eye(4)
            matrix[:3, 3] = translation
            return matrix.flatten()
        
        # 归一化速度向量（水滴尖端指向速度方向）
        vel_normalized = velocity / vel_norm
        
        # 默认水滴方向是向下（0, 0, -1）
        default_direction = np.array([0, 0, -1])
        
        # 计算旋转轴和角度
        dot_product = np.dot(default_direction, vel_normalized)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # 防止数值误差
        
        if abs(dot_product - 1.0) < 1e-6:
            # 已经对齐，不需要旋转
            matrix = np.eye(4)
            matrix[:3, 3] = translation
            return matrix.flatten()
        elif abs(dot_product + 1.0) < 1e-6:
            # 完全相反，旋转180度
            axis = np.array([1, 0, 0])  # 任意垂直轴
            angle = np.pi
        else:
            # 计算旋转轴（叉积）
            axis = np.cross(default_direction, vel_normalized)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot_product)
        
        # 使用Rodrigues公式计算旋转矩阵
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = translation
        return matrix.flatten()

    @staticmethod
    def generate_random_rotation_matrix(seed, translation):
        """生成随机旋转矩阵（用于向后兼容，当没有速度信息时）"""
        np.random.seed(seed)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0, 2 * np.pi)
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = translation
        return matrix.flatten()

    @staticmethod
    def standardize_point_cloud(pcl):
        # 只标准化位置信息（前3列），保留速度信息
        positions = pcl[:, :3]
        center = np.mean(positions, axis=0)
        scale = np.amax(positions - np.amin(positions, axis=0))
        normalized_positions = ((positions - center) / scale).astype(np.float32)
        
        if pcl.shape[1] == 6:
            # 有速度信息，合并位置和速度
            velocities = pcl[:, 3:6].astype(np.float32)
            return np.column_stack([normalized_positions, velocities])
        else:
            return normalized_positions

    def load_point_cloud(self):
        file_extension = os.path.splitext(self.file_path)[1]
        if file_extension == '.npy':
            data = np.load(self.file_path, allow_pickle=True)
            if data.shape[1] == 6:
                return data  # (N, 6): x, y, z, vx, vy, vz
            else:
                return data
        elif file_extension == '.npz':
            return np.load(self.file_path)['pred']
        elif file_extension == '.ply':
            ply_data = PlyData.read(self.file_path)
            vertex_data = ply_data['vertex']
            
            # 检查是否有速度信息（vx, vy, vz）
            if 'vx' in vertex_data.dtype.names and 'vy' in vertex_data.dtype.names and 'vz' in vertex_data.dtype.names:
                # 有速度信息，返回 (N, 6)
                return np.column_stack([
                    vertex_data['x'], vertex_data['y'], vertex_data['z'],
                    vertex_data['vx'], vertex_data['vy'], vertex_data['vz']
                ])
            # 检查是否用法线(nx, ny, nz)作为速度向量
            elif 'nx' in vertex_data.dtype.names and 'ny' in vertex_data.dtype.names and 'nz' in vertex_data.dtype.names:
                # 用法线作为速度向量，返回 (N, 6)
                return np.column_stack([
                    vertex_data['x'], vertex_data['y'], vertex_data['z'],
                    vertex_data['nx'], vertex_data['ny'], vertex_data['nz']
                ])
            else:
                # 只有位置信息，返回 (N, 3)
                return np.column_stack([vertex_data['x'], vertex_data['y'], vertex_data['z']])
        else:
            raise ValueError('Unsupported file format.')

    def generate_xml_content(self, pcl):
        xml_segments = [self.XML_HEAD]
        color = self.compute_color()
        
        # 检查是否有速度信息
        has_velocity = pcl.shape[1] == 6
        
        for idx, point in enumerate(pcl):
            if has_velocity:
                # 有速度信息：位置和速度
                position = point[:3]
                velocity = point[3:6]
                transform_matrix = self.generate_rotation_matrix_from_velocity(velocity, position)
            else:
                # 没有速度信息：使用随机旋转（向后兼容）
                position = point[:3]
                transform_matrix = self.generate_random_rotation_matrix(idx, position)
            
            xml_segments.append(self.XML_DROPLET_SEGMENT.format(
                self.droplet_mesh_path,
                *transform_matrix,
                color[0], color[1], color[2]
            ))
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
        except:
            try:
                mi.set_variant('cuda_rgb')
                print('Using CUDA GPU (cuda_rgb)')
            except:
                mi.set_variant('scalar_rgb')
                print('Using CPU (scalar_rgb) - GPU not available')

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

    @staticmethod
    def cleanup_temp_meshes():
        import shutil
        temp_dir = 'temp_meshes'
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main(argv):
    TrajectoryRenderer.init_mitsuba_variant()
    print('=' * 60)
    
    input_folder = 'trajectory_ply'
    output_folder = 'render'
    
    target_files = ['frame_0199_b0.ply']
    
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
    
    try:
        for idx, ply_file in enumerate(ply_files, 1):
            print(f'\n[{idx}/{total_files}] ({idx*100//total_files}%) Processing: {os.path.basename(ply_file)}')
            print('-' * 60)
            try:
                renderer = TrajectoryRenderer(ply_file, output_folder=output_folder)
                renderer.process()
                print(f'✓ Successfully processed: {os.path.basename(ply_file)}')
            except Exception as e:
                print(f'✗ Error processing {os.path.basename(ply_file)}: {str(e)}')
    finally:
        TrajectoryRenderer.cleanup_temp_meshes()
    
    print('\n' + '=' * 60)
    print(f'Batch processing completed! Processed {total_files} files.')
    print(f'Output files saved to: {output_folder}/')


if __name__ == '__main__':
    main(sys.argv)

