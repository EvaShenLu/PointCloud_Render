import numpy as np
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
            <integer name="sampleCount" value="128"/>
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
        <float name="radius" value="0.01"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
    TRAIL_SEGMENT = """
    <shape type="linearcurve">
        <string name="filename" value="{}"/>
        <bsdf type="roughplastic">
            <rgb name="diffuseReflectance" value="{},{},{}"/>
            <rgb name="specularReflectance" value="{},{},{}"/>
            <float name="alpha" value="0.005"/>
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


class TrajectoryBallRenderer:
    XML_HEAD = XMLTemplates.HEAD
    XML_BALL_SEGMENT = XMLTemplates.BALL_SEGMENT
    XML_TRAIL_SEGMENT = XMLTemplates.TRAIL_SEGMENT
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path, output_folder=None):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder
        self.curve_files = []

    @staticmethod
    def compute_color():
        return np.array([0.3, 0.3, 0.3])

    def _add_velocity_trail(self, xml_segments, position, velocity, point_index=0, frame_index=0):
        """根据速度向量生成尾迹线
        
        Args:
            xml_segments: XML片段列表
            position: 当前点位置
            velocity: 当前点速度向量
            point_index: 点的索引
            frame_index: 当前帧索引
        """
        velocity = np.array(velocity, dtype=np.float64)
        vel_norm = np.linalg.norm(velocity)
        
        # 如果速度太小，不添加尾迹
        if vel_norm < 1e-6:
            return
        
        # 根据帧索引计算尾迹长度缩放因子
        last_motion_frame = 199
        fade_frames = 20
        
        if frame_index <= 19:
            # 0-19帧：尾迹渐渐出现变长（从0到1）
            length_scale = frame_index / 19.0
        else:
            # 20-219帧：尾迹长度不变（保持最长长度，包括200-219帧也保持第199帧的状态）
            length_scale = 1.0
        
        # 如果缩放因子为0或负数，不添加尾迹
        if length_scale <= 0:
            return
        
        # 根据速度大小确定尾迹长度
        # 速度越大，尾迹越长
        base_trail_length = 0.07
        max_trail_length = 0.3
        vel_normalized = min(vel_norm / 10.0, 1.0)  # 归一化速度（假设最大速度约为10）
        trail_length = (base_trail_length + (max_trail_length - base_trail_length) * vel_normalized) * length_scale
        
        # 速度方向（反方向，因为水滴朝向速度反方向）
        vel_direction = -velocity / vel_norm
        
        # 生成尾迹点：从远端到当前位置
        n_trail_points = 20
        trail_points = []
        for i in range(n_trail_points):
            t = (n_trail_points - 1 - i) / (n_trail_points - 1)  # 反转t: 1 -> 0
            trail_point = position + vel_direction * trail_length * t
            trail_points.append(trail_point)
        
        # 现在trail_points[0]是尾迹远端,trail_points[-1]接近position
        # 添加position作为最后一个点,确保完全连接
        trail_points.append(position)
        
        # 创建临时曲线文件
        temp_curves_dir = 'temp_curves'
        os.makedirs(temp_curves_dir, exist_ok=True)
        curve_filename = f'trail_{point_index}_{id(self)}.txt'
        curve_filepath = os.path.join(temp_curves_dir, curve_filename)
        self.curve_files.append(curve_filepath)
        
        # 细线半径
        radius = 0.0007
        
        # 验证点是否有效
        valid_points = []
        for point in trail_points:
            point = np.asarray(point)
            if len(point.shape) == 1 and point.shape[0] == 3:
                if np.all(np.isfinite(point)) and not np.any(np.isnan(point)):
                    valid_points.append(point)
        
        if len(valid_points) < 2:
            return
        
        # 写入曲线文件（Mitsuba格式：每行 x y z radius）
        with open(curve_filepath, 'w') as f:
            for point in valid_points:
                f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {radius:.6f}\n')
        
        # 尾迹颜色和材质（保持不变）
        trail_color = np.array([0.2, 1.0, 0.4])
        abs_curve_path = os.path.abspath(curve_filepath).replace('\\', '/')
        specular_color = trail_color * 1.5
        specular_color = np.clip(specular_color, 0.0, 1.0)
        
        xml_segments.append(self.XML_TRAIL_SEGMENT.format(
            abs_curve_path,
            trail_color[0], trail_color[1], trail_color[2],
            specular_color[0], specular_color[1], specular_color[2]
        ))

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
                return data
            else:
                print(f'  Loaded data shape: {data.shape} (position only)')
                return data
        elif file_extension == '.npz':
            return np.load(self.file_path)['pred']
        elif file_extension == '.ply':
            ply_data = PlyData.read(self.file_path)
            vertex_data = ply_data['vertex']
            
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
            
            if has_vx:
                try:
                    data = np.column_stack([
                        vertex_data['x'], vertex_data['y'], vertex_data['z'],
                        vertex_data['vx'], vertex_data['vy'], vertex_data['vz']
                    ])
                    print(f'  Loaded PLY with velocity (vx,vy,vz): shape={data.shape}')
                    return data
                except (KeyError, ValueError):
                    pass
            
            if has_nx:
                try:
                    data = np.column_stack([
                        vertex_data['x'], vertex_data['y'], vertex_data['z'],
                        vertex_data['nx'], vertex_data['ny'], vertex_data['nz']
                    ])
                    print(f'  Loaded PLY with normal as velocity (nx,ny,nz): shape={data.shape}')
                    return data
                except (KeyError, ValueError):
                    pass
            
            data = np.column_stack([vertex_data['x'], vertex_data['y'], vertex_data['z']])
            print(f'  Loaded PLY position only: shape={data.shape}')
            return data
        else:
            raise ValueError('Unsupported file format.')

    @staticmethod
    def compute_camera_position(frame_index, total_frames=220):
        """根据帧数计算相机位置
        0-199帧：从(2.8, 2.8, 3.0)拉近到(2, 2, 2)
        200-219帧：从(2, 2, 2)拉近到(1.8, 1.8, 1.8)
        """
        last_motion_frame = 199
        fade_frames = 20
        
        if frame_index <= last_motion_frame:
            # 0-199帧：从起始位置到(2, 2, 2)
            start_pos = (2.8, 2.8, 3.0)
            end_pos = (1.8, 1.8, 1.8)
            progress = frame_index / max(last_motion_frame, 1)
            origin_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            origin_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            origin_z = start_pos[2] + (end_pos[2] - start_pos[2]) * progress
        else:
            # 200-219帧：从(2, 2, 2)到(1.8, 1.8, 1.8)
            start_pos = (1.8, 1.8, 1.8)
            end_pos = (1.6, 1.6, 1.6)
            fade_progress = (frame_index - last_motion_frame) / max(fade_frames, 1)
            origin_x = start_pos[0] + (end_pos[0] - start_pos[0]) * fade_progress
            origin_y = start_pos[1] + (end_pos[1] - start_pos[1]) * fade_progress
            origin_z = start_pos[2] + (end_pos[2] - start_pos[2]) * fade_progress
        
        return origin_x, origin_y, origin_z

    def generate_xml_content(self, pcl, frame_index=0, total_frames=220):
        origin_x, origin_y, origin_z = self.compute_camera_position(frame_index, total_frames)
        xml_segments = [self.XML_HEAD.format(origin_x, origin_y, origin_z)]
        color = self.compute_color()
        
        has_velocity = pcl.shape[1] == 6
        if not has_velocity:
            print('  Warning: No velocity info, trails will not be rendered')
        
        for idx, point in enumerate(pcl):
            position = point[:3]
            
            if has_velocity:
                velocity = point[3:6]
                # 根据速度添加尾迹
                self._add_velocity_trail(xml_segments, position, velocity, point_index=idx, frame_index=frame_index)
            
            # 使用小球替代水滴mesh
            xml_segments.append(self.XML_BALL_SEGMENT.format(
                position[0], position[1], position[2],
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

    def process(self, frame_index=0, total_frames=220):
        """处理单帧点云：标准化、坐标变换、渲染"""
        self.curve_files = []
        
        pcl = self.load_point_cloud()
        if len(pcl.shape) == 3:
            pcl = pcl[0]
        
        pcl = self.standardize_point_cloud(pcl)
        pcl = self.transform_coordinates(pcl)
        
        output_filename = f'frame_{frame_index:04d}_b0' if frame_index > 199 else self.filename
        
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, output_filename)
        else:
            output_file_path = os.path.join(self.folder, output_filename)
        
        print('  Generating XML...', end=' ', flush=True)
        xml_content = self.generate_xml_content(pcl, frame_index=frame_index, total_frames=total_frames)
        xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
        
        print('Rendering...', end=' ', flush=True)
        rendered_scene = self.render_scene(xml_file_path)
        
        print('Saving...', end=' ', flush=True)
        self.save_scene(output_file_path, rendered_scene)
        
        if os.path.exists(xml_file_path):
            os.remove(xml_file_path)
        
        self.cleanup_temp_curves()
        print('Done!')

    def cleanup_temp_curves(self):
        """清理临时曲线文件"""
        for curve_file in self.curve_files:
            try:
                if os.path.exists(curve_file):
                    os.remove(curve_file)
            except:
                pass
        self.curve_files = []
    
    @staticmethod
    def cleanup_temp_curves_dir():
        """清理临时曲线目录"""
        import shutil
        temp_dir = 'temp_curves'
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    TrajectoryBallRenderer.init_mitsuba_variant()
    print('=' * 60)
    
    input_folder = 'trajectory_ply'
    output_folder = 'render'
    
    last_motion_frame = 199
    fade_frames = 20
    total_frames = last_motion_frame + fade_frames + 1
    
    # 渲染全部帧（0-219）
    start_frame = 0
    end_frame = 219
    frame_numbers = list(range(start_frame, end_frame + 1))
    target_files = []
    for num in frame_numbers:
        if num <= last_motion_frame:
            target_files.append(f'frame_{num:04d}_b0.ply')
        else:
            target_files.append(f'frame_0199_b0.ply')
    
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
        return
    
    total_files = len(ply_files)
    print(f'Found {total_files} target file(s) in folder: {input_folder}')
    print(f'Output folder: {output_folder}')
    print('=' * 60)
    
    try:
        for idx, ply_file in enumerate(ply_files):
            print(f'\n[{idx+1}/{total_files}] ({(idx+1)*100//total_files}%) Processing: {os.path.basename(ply_file)}')
            print('-' * 60)
            try:
                frame_index = frame_numbers[idx]  # 使用实际的帧号
                renderer = TrajectoryBallRenderer(ply_file, output_folder=output_folder)
                renderer.process(frame_index, total_frames)
                print(f'✓ Successfully processed: {os.path.basename(ply_file)}')
            except Exception as e:
                print(f'✗ Error processing {os.path.basename(ply_file)}: {str(e)}')
    finally:
        TrajectoryBallRenderer.cleanup_temp_curves_dir()
    
    print('\n' + '=' * 60)
    print(f'Batch processing completed! Processed {total_files} files.')
    print(f'Output files saved to: {output_folder}/')


if __name__ == '__main__':
    main()

