import numpy as np
import os
from plyfile import PlyData
import mitsuba as mi

try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print('Warning: scipy not available, trail rendering disabled')


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
        self.curve_files = []  # 存储临时曲线文件路径，用于清理

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

    def _add_trail_lines(self, xml_segments, position, velocity, history_positions=None, point_index=0):
        """为每个点添加尾迹线，将过去10帧的真实位置拟合成曲线，使用Mitsuba的linearcurve
        
        Args:
            xml_segments: XML片段列表
            position: 当前点位置（当前帧的真实位置）
            velocity: 当前点速度
            history_positions: 历史位置列表，每个元素是(3,)数组，从最远到最近
            point_index: 点的索引
        """
        if history_positions is None or len(history_positions) == 0:
            return
        
        # 尾迹长度（帧数），例如20帧
        trail_length_frames = 20
        
        # 限制历史帧数为trail_length_frames
        max_history = min(trail_length_frames, len(history_positions))
        used_history = history_positions[-max_history:]  # 取最近N帧
        
        # 如果历史点太少，不添加尾迹
        if len(used_history) < 2:
            return
        
        # 构建用于拟合的点列表（历史位置，从最远到最近）
        # 注意：不包括当前帧的position，因为我们要拟合的是历史轨迹
        fit_points = used_history  # 保持时间从旧 → 新
        
        # 将点转换为numpy数组
        points_array = np.array(fit_points)  # (n_points, 3)
        
        # 使用Catmull-Rom样条插值拟合光滑曲线
        if len(fit_points) >= 2:
            try:
                # Catmull-Rom插值函数
                def catmull_rom_segment(p0, p1, p2, p3, t):
                    """Catmull-Rom样条段插值
                    Args:
                        p0, p1, p2, p3: 四个控制点
                        t: 参数 [0, 1]，t=0时返回p1，t=1时返回p2
                    """
                    t2 = t * t
                    t3 = t2 * t
                    return 0.5 * (
                        (2 * p1) +
                        (-p0 + p2) * t +
                        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                        (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                    )
                
                # 计算每个点沿曲线的累积距离作为参数（弦长参数化）
                distances = np.zeros(len(fit_points))
                for i in range(1, len(fit_points)):
                    distances[i] = distances[i-1] + np.linalg.norm(points_array[i] - points_array[i-1])
                
                # 归一化到[0,1]
                if distances[-1] > 1e-6:
                    distances = distances / distances[-1]
                else:
                    # 如果距离为0，使用均匀参数化
                    distances = np.linspace(0, 1, len(fit_points))
                
                # 对每个段进行Catmull-Rom插值
                n_samples = 20  # 总采样点数
                smooth_points = []
                
                if len(fit_points) == 2:
                    # 只有2个点，使用线性插值
                    for i in range(n_samples):
                        t = i / (n_samples - 1)
                        point = (1 - t) * points_array[0] + t * points_array[1]
                        smooth_points.append(point)
                else:
                    # 3个或更多点，使用Catmull-Rom
                    # 为每个段生成采样点
                    n_segments = len(fit_points) - 1
                    samples_per_segment = max(2, n_samples // n_segments)
                    
                    for seg_idx in range(n_segments):
                        # 获取四个控制点（处理边界）
                        if seg_idx == 0:
                            # 第一段：在开头添加虚拟点（使用p0-p1的延长）
                            p0 = points_array[0] - (points_array[1] - points_array[0])
                            p1, p2, p3 = points_array[0], points_array[1], points_array[min(2, len(fit_points)-1)]
                        elif seg_idx == n_segments - 1:
                            # 最后一段：在结尾添加虚拟点（使用p2-p3的延长）
                            p0 = points_array[max(seg_idx-1, 0)]
                            p1, p2 = points_array[seg_idx], points_array[seg_idx+1]
                            p3 = points_array[seg_idx+1] + (points_array[seg_idx+1] - points_array[seg_idx])
                        else:
                            # 中间段：使用正常的四个点
                            p0, p1, p2, p3 = points_array[seg_idx-1], points_array[seg_idx], points_array[seg_idx+1], points_array[min(seg_idx+2, len(fit_points)-1)]
                        
                        # 对当前段进行采样
                        for i in range(samples_per_segment):
                            t = i / (samples_per_segment - 1) if samples_per_segment > 1 else 0
                            point = catmull_rom_segment(p0, p1, p2, p3, t)
                            smooth_points.append(point)
                    
                    # 确保总采样点数正确
                    if len(smooth_points) > n_samples:
                        # 均匀采样
                        indices = np.linspace(0, len(smooth_points)-1, n_samples).astype(int)
                        smooth_points = [smooth_points[i] for i in indices]
                    elif len(smooth_points) < n_samples:
                        # 补充采样
                        while len(smooth_points) < n_samples:
                            smooth_points.append(smooth_points[-1])
                
                smooth_points = np.array(smooth_points)
            except Exception as e:
                # 如果插值失败，使用原始点
                smooth_points = points_array
        else:
            # 如果点太少，使用原始点
            smooth_points = points_array
        
        # 尾迹点列表：时间方向从旧 → 新，最后是当前位置
        # smooth_points已经是按时间顺序（从旧到新）的采样点
        # 转换为列表以便使用append
        if isinstance(smooth_points, np.ndarray):
            trail_points_list = smooth_points.tolist()
        else:
            trail_points_list = list(smooth_points)
        trail_points_list.append(position)
        
        # 创建临时曲线文件
        temp_curves_dir = 'temp_curves'
        os.makedirs(temp_curves_dir, exist_ok=True)
        curve_filename = f'trail_{point_index}_{id(self)}.txt'
        curve_filepath = os.path.join(temp_curves_dir, curve_filename)
        self.curve_files.append(curve_filepath)
        
        # 细线半径
        radius = 0.0007
        
        # 验证点是否有效（检查NaN和Inf）
        valid_points = []
        for point in trail_points_list:
            point = np.asarray(point)
            if len(point.shape) == 1 and point.shape[0] == 3:
                if np.all(np.isfinite(point)) and not np.any(np.isnan(point)):
                    valid_points.append(point)
        
        # 如果有效点太少，不添加尾迹
        if len(valid_points) < 2:
            return
        
        # 写入曲线文件（Mitsuba格式：每行 x y z radius）
        # 确保至少有两个不同的点
        if len(valid_points) < 2:
            return
        
        # 检查点之间的距离，如果太近可能有问题
        # 确保曲线是开放的，不会形成闭环
        min_distance = 1e-5  # 增大最小距离阈值
        filtered_points = [valid_points[0]]
        for i in range(1, len(valid_points)):
            dist = np.linalg.norm(valid_points[i] - filtered_points[-1])
            if dist > min_distance:
                filtered_points.append(valid_points[i])
        
        # 确保首尾点不重复（避免形成封闭曲线）
        if len(filtered_points) >= 2:
            first_point = filtered_points[0]
            last_point = filtered_points[-1]
            if np.linalg.norm(first_point - last_point) < min_distance:
                # 如果首尾太接近，移除最后一个点，确保曲线是开放的
                filtered_points = filtered_points[:-1]
        
        if len(filtered_points) < 2:
            return
        
        with open(curve_filepath, 'w') as f:
            for point in filtered_points:
                f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {radius:.6f}\n')
        
        # 亮绿色streamline效果：使用高亮度的reflectance值来模拟发光
        # 使用bsdf而不是emitter，避免Mitsuba的emitter兼容性问题
        # 降低R和B通道让绿色更深，保持G通道高亮度
        trail_color = np.array([0.2, 1.0, 0.4])
        
        # 使用absolute path，并转换为正斜杠（Mitsuba XML需要）
        abs_curve_path = os.path.abspath(curve_filepath).replace('\\', '/')
        # 使用更亮的specularReflectance来增强发光效果
        specular_color = trail_color * 1.5  # 增强镜面反射亮度
        specular_color = np.clip(specular_color, 0.0, 1.0)  # 限制在[0,1]范围内
        
        xml_segments.append(self.XML_TRAIL_SEGMENT.format(
            abs_curve_path,
            trail_color[0], trail_color[1], trail_color[2],  # diffuseReflectance
            specular_color[0], specular_color[1], specular_color[2]  # specularReflectance (更亮)
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
        # 增加拉近程度：从2.8拉近到0.8（移动2.0单位），高度从3.0拉近到1.0（移动2.0单位）
        origin_x = 2.8 - 2.0 * progress
        origin_y = 2.8 - 2.0 * progress
        origin_z = 3.0 - 2.0 * progress
        return origin_x, origin_y, origin_z

    def generate_xml_content(self, pcl, frame_index=0, total_frames=220, history_pcls=None):
        origin_x, origin_y, origin_z = self.compute_camera_position(frame_index, total_frames)
        xml_segments = [self.XML_HEAD.format(origin_x, origin_y, origin_z)]
        color = self.compute_color()
        
        # 检查是否有速度信息
        has_velocity = pcl.shape[1] == 6
        if has_velocity:
            print(f'  Using velocity for orientation: {pcl.shape[0]} points')
        else:
            print(f'  No velocity info, using random rotation: {pcl.shape[0]} points')
        
        # 准备历史位置数据：为每个点准备其历史位置列表（最近N帧）
        point_histories = None
        if history_pcls is not None and len(history_pcls) > 0:
            point_histories = []
            # 简单的索引匹配（假设点的顺序一致）
            for point_idx in range(pcl.shape[0]):
                point_history = []
                for hist_pcl in history_pcls:
                    if point_idx < hist_pcl.shape[0]:
                        hist_position = hist_pcl[point_idx, :3]
                        point_history.append(hist_position)
                point_histories.append(point_history)
        
        for idx, point in enumerate(pcl):
            if has_velocity:
                # 有速度信息：位置和速度
                position = point[:3]
                velocity = point[3:6]
                transform_matrix = self.generate_rotation_matrix_from_velocity(velocity, position)
                
                # 添加尾迹线（如果有历史数据）
                history_positions = point_histories[idx] if point_histories is not None else None
                if history_positions is not None and len(history_positions) > 0:
                    self._add_trail_lines(xml_segments, position, velocity, history_positions, point_index=idx)
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

    def process(self, frame_index=0, history_pcls=None, total_frames=220):
        """处理单帧点云：标准化、坐标变换、渲染
        
        Args:
            frame_index: 当前帧索引
            history_pcls: 历史帧的点云数据列表（已标准化和坐标变换），每个元素是(N, 3)或(N, 6)的数组
            total_frames: 总帧数
        """
        # 清理之前的曲线文件
        self.curve_files = []
        
        pcl = self.load_point_cloud()
        if len(pcl.shape) == 3:
            pcl = pcl[0]  # 如果有多帧，只取第一帧
        
        pcl = self.standardize_point_cloud(pcl)
        pcl = self.transform_coordinates(pcl)
        
        output_filename = f'frame_{frame_index:04d}_b0' if frame_index > 199 else self.filename
        
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, output_filename)
        else:
            output_file_path = os.path.join(self.folder, output_filename)
        
        print('  Generating XML...', end=' ', flush=True)
        xml_content = self.generate_xml_content(pcl, frame_index=frame_index, total_frames=total_frames, history_pcls=history_pcls)
        xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
        
        print('Rendering...', end=' ', flush=True)
        rendered_scene = self.render_scene(xml_file_path)
        
        print('Saving...', end=' ', flush=True)
        self.save_scene(output_file_path, rendered_scene)
        
        if os.path.exists(xml_file_path):
            os.remove(xml_file_path)
        
        # 清理临时曲线文件
        self.cleanup_temp_curves()
        
        print('Done!')

    @staticmethod
    def cleanup_temp_meshes():
        import shutil
        temp_dir = 'temp_meshes'
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
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
    TrajectoryRenderer.init_mitsuba_variant()
    print('=' * 60)
    
    input_folder = 'trajectory_ply'
    output_folder = 'render'
    
    last_motion_frame = 199
    fade_frames = 20
    total_frames = last_motion_frame + fade_frames + 1
    
    # 渲染全部帧
    frame_numbers = list(range(0, total_frames))  # 0, 1, 2, ..., 219
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
        print(f'Looking for: {target_files}')
        return
    
    total_files = len(ply_files)
    print(f'Found {total_files} target file(s) in folder: {input_folder}')
    print(f'Output folder: {output_folder}')
    print('=' * 60)
    
    # 一次性加载所有帧的数据（用于历史轨迹）
    print('\nLoading all frame data...')
    all_frame_data = []
    
    first_valid_file = next((f for f in ply_files if os.path.exists(f)), None)
    if first_valid_file is None:
        print('No valid files found')
        return
    
    temp_renderer = TrajectoryRenderer(first_valid_file, output_folder=output_folder)
    
    for ply_file in ply_files:
        try:
            temp_renderer.file_path = ply_file
            pcl = temp_renderer.load_point_cloud()
            if len(pcl.shape) == 3:
                pcl = pcl[0]
            pcl = temp_renderer.standardize_point_cloud(pcl)
            pcl = temp_renderer.transform_coordinates(pcl)
            all_frame_data.append(pcl)
        except Exception as e:
            print(f'Warning: Failed to load {os.path.basename(ply_file)}: {e}')
            all_frame_data.append(None)
    
    print(f'Loaded {len([d for d in all_frame_data if d is not None])} frames successfully')
    print('=' * 60)
    
    try:
        shared_renderer = TrajectoryRenderer(first_valid_file, output_folder=output_folder)
        
        for idx, ply_file in enumerate(ply_files, 1):
            print(f'\n[{idx}/{total_files}] ({idx*100//total_files}%) Processing: {os.path.basename(ply_file)}')
            print('-' * 60)
            try:
                frame_index = idx - 1
                
                # 准备历史数据（只保留最近20帧历史轨迹）
                max_history_frames = 20
                history_pcls = []
                if frame_index > 0:
                    start_frame = max(0, frame_index - max_history_frames)
                    for hist_idx in range(start_frame, frame_index):
                        if hist_idx < len(all_frame_data) and all_frame_data[hist_idx] is not None:
                            history_pcls.append(all_frame_data[hist_idx])
                
                shared_renderer.file_path = ply_file
                shared_renderer.folder, full_filename = os.path.split(ply_file)
                shared_renderer.folder = shared_renderer.folder or '.'
                shared_renderer.filename, _ = os.path.splitext(full_filename)
                
                shared_renderer.process(frame_index, history_pcls, total_frames)
                print(f'✓ Successfully processed: {os.path.basename(ply_file)}')
            except Exception as e:
                print(f'✗ Error processing {os.path.basename(ply_file)}: {str(e)}')
    finally:
        TrajectoryRenderer.cleanup_temp_meshes()
        TrajectoryRenderer.cleanup_temp_curves_dir()
    
    print('\n' + '=' * 60)
    print(f'Batch processing completed! Processed {total_files} files.')
    print(f'Output files saved to: {output_folder}/')


if __name__ == '__main__':
    main()

