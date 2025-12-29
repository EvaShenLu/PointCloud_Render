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
            <lookat origin="{},{},{}" target="0,0,-0.05" up="0,0,1"/>
        </transform>
        <float name="fov" value="36"/>
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
    TRAIL_SEGMENT = """
    <shape type="cylinder">
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
            <translate x="0" y="0" z="-0.5"/>
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
    XML_TRAIL_SEGMENT = XMLTemplates.TRAIL_SEGMENT
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
        """根据速度向量生成旋转矩阵，使水滴尖端完全指向速度的反方向"""
        velocity = np.array(velocity, dtype=np.float64)
        vel_norm = np.linalg.norm(velocity)
        
        if vel_norm < 1e-6:
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 3] = translation
            return matrix.flatten()
        
        target_direction = velocity / vel_norm
        default_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # 水滴默认朝向
        
        dot_product = np.clip(np.dot(default_direction, target_direction), -1.0, 1.0)
        axis = np.cross(default_direction, target_direction)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-8:
            if dot_product > 0.999:
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, 3] = translation
                return matrix.flatten()
            else:
                # 完全相反，旋转180度
                temp = np.array([1.0, 0.0, 0.0]) if abs(target_direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                axis = np.cross(target_direction, temp)
                axis_norm = np.linalg.norm(axis)
                axis = axis / axis_norm if axis_norm > 1e-8 else np.array([0.0, 1.0, 0.0])
                angle = np.pi
        else:
            axis = axis / axis_norm
            # 使用arccos，但已经clip到[-1,1]范围，数值稳定
            angle = np.arccos(dot_product)
        
        # Rodrigues公式
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = R
        matrix[:3, 3] = translation
        return matrix.flatten()
    
    def _add_trail_lines(self, xml_segments, position, velocity):
        """为每个点添加尾迹线（浅绿色细线，逐渐变透明）"""
        vel_norm = np.linalg.norm(velocity)
        if vel_norm < 1e-6:
            return  # 速度为零，不添加尾迹
        
        trail_length = min(0.025, vel_norm * 0.08)  # 延长尾迹
        trail_direction = -velocity / vel_norm
        base_color = np.array([0.4, 0.9, 0.6])
        
        # 水滴mesh的中心不在原点，需要调整尾迹起点
        # 水滴mesh从z=0（顶部）到z=-length*0.8（底部尖端），中心大约在z=-0.014
        # 但水滴的transform_matrix的translation是position，对应mesh原点(0,0,0)
        # 水滴实际中心位置需要根据mesh的几何中心计算
        # 简化：假设水滴中心在position（因为mesh大致以原点为中心分布）
        # 尾迹应该从水滴的尖端（水滴mesh的底部）开始
        # 水滴尖端在默认方向(0,0,-1)上，距离中心约length*0.4
        droplet_length = 0.035
        droplet_tip_offset = droplet_length * 0.4  # 水滴尖端到中心的距离
        # 计算水滴尖端位置（沿着速度方向，因为水滴尖端指向速度方向）
        droplet_tip = position + (velocity / vel_norm) * droplet_tip_offset
        
        n_segments = 1  # 单段，减少几何体数量
        for i in range(n_segments):
            t0 = i / n_segments
            t1 = (i + 1) / n_segments
            
            # 尾迹从水滴尖端开始，向速度反方向延伸
            start_pos = droplet_tip + trail_direction * trail_length * t0
            end_pos = droplet_tip + trail_direction * trail_length * t1
            segment_length = trail_length / n_segments
            segment_center = (start_pos + end_pos) / 2
            
            # 使用颜色强度模拟透明度（单段，中等亮度）
            trail_color = base_color * 0.6  # 60%亮度，保持可见但不过亮
            
            z_axis = np.array([0.0, 0.0, 1.0])
            dot = np.clip(np.dot(z_axis, trail_direction), -1.0, 1.0)
            
            if abs(dot - 1.0) < 1e-6:
                R = np.eye(3)
            elif abs(dot + 1.0) < 1e-6:
                temp = np.array([1.0, 0.0, 0.0]) if abs(trail_direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                axis = np.cross(trail_direction, temp)
                axis = axis / np.linalg.norm(axis)
                cos_a, sin_a = np.cos(np.pi), np.sin(np.pi)
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
            else:
                axis = np.cross(z_axis, trail_direction)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(dot)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
            
            radius = 0.0007
            scale_matrix = np.eye(4, dtype=np.float64)
            scale_matrix[0, 0] = scale_matrix[1, 1] = radius
            scale_matrix[2, 2] = segment_length
            
            rot_matrix = np.eye(4, dtype=np.float64)
            rot_matrix[:3, :3] = R
            
            trans_matrix = np.eye(4, dtype=np.float64)
            trans_matrix[:3, 3] = segment_center
            
            transform_matrix = trans_matrix @ rot_matrix @ scale_matrix
            
            xml_segments.append(self.XML_TRAIL_SEGMENT.format(
                *transform_matrix.flatten(),
                trail_color[0], trail_color[1], trail_color[2]
            ))

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
        """标准化点云位置信息，保留速度信息"""
        positions = pcl[:, :3]
        center = np.mean(positions, axis=0)
        scale = np.amax(positions - np.amin(positions, axis=0))
        normalized_positions = ((positions - center) / scale).astype(np.float32)
        
        if pcl.shape[1] == 6:
            velocities = pcl[:, 3:6].astype(np.float32)
            return np.column_stack([normalized_positions, velocities])
        else:
            return normalized_positions

    @staticmethod
    def transform_coordinates(pcl):
        """坐标变换：重新排列位置和速度坐标，统一坐标系"""
        has_velocity = pcl.shape[1] == 6
        if has_velocity:
            pcl_positions = pcl[:, [2, 0, 1]]
            pcl_positions[:, 0] *= -1
            pcl_positions[:, 2] += 0.0125
            
            pcl_velocities = pcl[:, [5, 3, 4]]
            pcl_velocities[:, 0] *= -1
            
            return np.column_stack([pcl_positions, pcl_velocities])
        else:
            pcl = pcl[:, [2, 0, 1]]
            pcl[:, 0] *= -1
            pcl[:, 2] += 0.0125
            return pcl

    def load_point_cloud(self):
        file_extension = os.path.splitext(self.file_path)[1]
        if file_extension == '.npy':
            data = np.load(self.file_path, allow_pickle=True)
            if data.shape[1] == 6:
                print(f'  Loaded data shape: {data.shape} (with velocity)')
                print(f'  Sample velocity: {data[0, 3:6]}')
                return data  # (N, 6): x, y, z, vx, vy, vz
            else:
                print(f'  Loaded data shape: {data.shape} (position only)')
                return data
        elif file_extension == '.npz':
            return np.load(self.file_path)['pred']
        elif file_extension == '.ply':
            ply_data = PlyData.read(self.file_path)
            vertex_data = ply_data['vertex']
            
            # 尝试直接访问属性来检查是否存在
            has_vx = False
            has_nx = False
            try:
                _ = vertex_data['vx']
                has_vx = True
            except (KeyError, ValueError):
                pass
            
            try:
                _ = vertex_data['nx']
                has_nx = True
            except (KeyError, ValueError):
                pass
            
            # 检查是否有速度信息（vx, vy, vz）
            if has_vx:
                try:
                    data = np.column_stack([
                        vertex_data['x'], vertex_data['y'], vertex_data['z'],
                        vertex_data['vx'], vertex_data['vy'], vertex_data['vz']
                    ])
                    print(f'  Loaded PLY with velocity (vx,vy,vz): shape={data.shape}')
                    print(f'  Sample velocity: {data[0, 3:6]}')
                    return data
                except (KeyError, ValueError):
                    pass
            
            # 检查是否用法线(nx, ny, nz)作为速度向量（取反，使水滴朝向飞机内侧）
            if has_nx:
                try:
                    # 法线取反，使水滴朝向飞机内侧
                    data = np.column_stack([
                        vertex_data['x'], vertex_data['y'], vertex_data['z'],
                        vertex_data['nx'], vertex_data['ny'], vertex_data['nz']
                    ])
                    print(f'  Loaded PLY with normal as velocity (nx,ny,nz, inverted): shape={data.shape}')
                    print(f'  Sample velocity (from -normal): {data[0, 3:6]}')
                    return data
                except (KeyError, ValueError):
                    pass
            
            # 只有位置信息，返回 (N, 3)
            data = np.column_stack([vertex_data['x'], vertex_data['y'], vertex_data['z']])
            print(f'  Loaded PLY position only: shape={data.shape}')
            return data
        else:
            raise ValueError('Unsupported file format.')

    @staticmethod
    def compute_camera_position(frame_index, total_frames=200):
        """根据帧数计算相机位置（从远到近）"""
        progress = frame_index / max(total_frames - 1, 1)
        origin_x = 2.8 - 1.0 * progress
        origin_y = 2.8 - 1.0 * progress
        origin_z = 3.0 - 0.8 * progress
        return origin_x, origin_y, origin_z

    def generate_xml_content(self, pcl, frame_index=0, total_frames=200):
        origin_x, origin_y, origin_z = self.compute_camera_position(frame_index, total_frames)
        xml_segments = [self.XML_HEAD.format(origin_x, origin_y, origin_z)]
        color = self.compute_color()
        
        # 检查是否有速度信息
        has_velocity = pcl.shape[1] == 6
        if has_velocity:
            print(f'  Using velocity for orientation: {pcl.shape[0]} points')
        else:
            print(f'  No velocity info, using random rotation: {pcl.shape[0]} points')
        
        for idx, point in enumerate(pcl):
            if has_velocity:
                # 有速度信息：位置和速度
                position = point[:3]
                velocity = point[3:6]
                transform_matrix = self.generate_rotation_matrix_from_velocity(velocity, position)
                
                # 添加尾迹线（沿着速度反方向）
                self._add_trail_lines(xml_segments, position, velocity)
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

    @staticmethod
    def extract_frame_index(filename):
        """从文件名提取帧数"""
        try:
            if 'frame_' in filename:
                frame_str = filename.split('_')[1]
                return int(frame_str)
        except (ValueError, IndexError):
            pass
        return 0

    def process(self):
        """处理单帧点云：标准化、坐标变换、渲染"""
        pcl = self.load_point_cloud()
        if len(pcl.shape) == 3:
            pcl = pcl[0]  # 如果有多帧，只取第一帧
        
        pcl = self.standardize_point_cloud(pcl)
        pcl = self.transform_coordinates(pcl)
        
        output_filename = f'{self.filename}'
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, output_filename)
        else:
            output_file_path = os.path.join(self.folder, output_filename)
        
        print('  Generating XML...', end=' ', flush=True)
        
        frame_index = self.extract_frame_index(self.filename)
        xml_content = self.generate_xml_content(pcl, frame_index=frame_index, total_frames=200)
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
    
    # 测试第199帧
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

