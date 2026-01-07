import numpy as np
import os
import sys
from plyfile import PlyData, PlyElement
import mitsuba as mi


class NormalColorRenderer:
    """
    高效的点云渲染器，使用Mitsuba Python API和instancing优化
    将法向量映射为RGB颜色
    """
    
    def __init__(self, file_path, output_folder=None, 
                 particle_radius=None, 
                 image_width=1920, image_height=1080,
                 samples=64):
        """
        初始化渲染器
        
        Args:
            file_path: PLY文件路径
            output_folder: 输出文件夹
            particle_radius: 粒子半径（None表示自适应）
            image_width: 图像宽度
            image_height: 图像高度
            samples: 采样数
        """
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder
        self.particle_radius = particle_radius
        self.image_width = image_width
        self.image_height = image_height
        self.samples = samples
        
    @staticmethod
    def init_mitsuba_variant():
        """初始化Mitsuba variant，使用GPU"""
        try:
            mi.set_variant('cuda_ad_rgb')
            print('Using CUDA GPU (cuda_ad_rgb)')
        except:
            try:
                mi.set_variant('cuda_rgb')
                print('Using CUDA GPU (cuda_rgb)')
            except Exception as e:
                raise RuntimeError(f'Failed to initialize GPU. CUDA not available: {e}')
    
    
    @staticmethod
    def _read_ply_header_fast(file_path):
        """
        快速读取PLY文件header，返回顶点数量和header大小
        
        Returns:
            num_vertices: 顶点数量
            header_size: header字节大小
        """
        with open(file_path, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline()
                header_lines.append(line)
                if b'end_header' in line:
                    break
            header_size = sum(len(line) for line in header_lines)
            
            # 从header中提取顶点数量
            num_vertices = None
            for line in header_lines:
                if b'element vertex' in line:
                    parts = line.decode('ascii', errors='ignore').split()
                    if len(parts) >= 3:
                        num_vertices = int(parts[2])
                        break
            
            if num_vertices is None:
                raise ValueError('Could not find vertex count in PLY header')
            
            return num_vertices, header_size
    
    @staticmethod
    def _read_ply_binary_fast(file_path, num_vertices, header_size):
        """
        使用np.fromfile快速读取PLY二进制数据（跳过Python解析开销）
        
        Args:
            file_path: PLY文件路径
            num_vertices: 顶点数量
            header_size: header大小（字节）
            
        Returns:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组
            batch_indices: (N,) 批次索引数组或None
        """
        # 定义顶点数据结构：x(4) + y(4) + z(4) + nx(4) + ny(4) + nz(4) + batch_idx(4) = 28字节
        dtype = np.dtype([
            ('x', '<f4'),      # little-endian float32
            ('y', '<f4'),
            ('z', '<f4'),
            ('nx', '<f4'),
            ('ny', '<f4'),
            ('nz', '<f4'),
            ('batch_idx', '<i4')  # little-endian int32
        ])
        
        # 直接从文件读取二进制数据（跳过header）
        with open(file_path, 'rb') as f:
            f.seek(header_size)  # 跳过header
            data = np.fromfile(f, dtype=dtype, count=num_vertices)
        
        # 提取位置、法向量和批次索引
        positions = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
        normals = np.column_stack([data['nx'], data['ny'], data['nz']]).astype(np.float32)
        batch_indices = data['batch_idx'].astype(np.int32)
        
        return positions, normals, batch_indices
    
    @staticmethod
    def _read_colored_ply_binary_fast(file_path, num_vertices, header_size):
        """
        快速读取带颜色的PLY文件（x, y, z, red, green, blue）
        
        Args:
            file_path: PLY文件路径
            num_vertices: 顶点数量
            header_size: header大小（字节）
            
        Returns:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组，范围[0, 1]
        """
        # 定义顶点数据结构：x(4) + y(4) + z(4) + red(1) + green(1) + blue(1) = 15字节
        dtype = np.dtype([
            ('x', '<f4'),      # little-endian float32
            ('y', '<f4'),
            ('z', '<f4'),
            ('red', 'u1'),    # uint8
            ('green', 'u1'),
            ('blue', 'u1')
        ])
        
        # 直接从文件读取二进制数据（跳过header）
        with open(file_path, 'rb') as f:
            f.seek(header_size)  # 跳过header
            data = np.fromfile(f, dtype=dtype, count=num_vertices)
        
        # 提取位置和颜色
        positions = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
        colors = np.column_stack([
            data['red'] / 255.0,
            data['green'] / 255.0,
            data['blue'] / 255.0
        ]).astype(np.float32)
        
        return positions, colors
    
    def load_point_cloud(self):
        """
        加载PLY文件，返回位置、法向量和批次索引
        使用优化的二进制读取方法，跳过Python解析开销
        """
        file_extension = os.path.splitext(self.file_path)[1]
        
        if file_extension == '.ply':
            # 快速读取：先读header获取信息
            num_vertices, header_size = self._read_ply_header_fast(self.file_path)
            
            # 使用np.fromfile直接读取二进制数据（比PlyData.read快得多）
            positions, normals, batch_indices = self._read_ply_binary_fast(
                self.file_path, num_vertices, header_size
            )
            
            return positions, normals, batch_indices
        else:
            raise ValueError(f'Unsupported file format: {file_extension}')
    
    @staticmethod
    def normal_to_rgb(normals):
        """
        将法向量映射到RGB颜色（优化版本）
        
        Args:
            normals: (N, 3) 法向量数组
            
        Returns:
            rgb: (N, 3) RGB颜色数组，范围[0, 1]
        """
        # 归一化法向量（使用更高效的向量化操作）
        # 使用平方和开方，避免重复计算
        norms = np.sqrt(np.sum(normals ** 2, axis=1, keepdims=True))
        normalized = normals / (norms + 1e-8)
        
        # 映射到[0, 1]范围: (normal + 1) / 2
        rgb = (normalized + 1.0) * 0.5
        
        # 确保值在[0, 1]范围内（clip比条件判断快）
        return np.clip(rgb, 0.0, 1.0)
    
    @staticmethod
    def compute_adaptive_radius(positions):
        """
        根据点云计算自适应粒子半径（优化：使用更小的半径以提升渲染速度）
        
        Args:
            positions: (N, 3) 位置数组
            
        Returns:
            radius: 自适应半径
        """
        bbox_min = np.min(positions, axis=0)
        bbox_max = np.max(positions, axis=0)
        bbox_size = np.max(bbox_max - bbox_min)
        num_points = len(positions)
        
        # 根据点云密度计算半径
        # 使用立方根来估算合适的粒子大小
        # 减小系数以使用更小的半径（提升渲染速度）
        radius = bbox_size / (num_points ** 0.33) * 0.12  # 从0.15降到0.12
        
        # 限制在合理范围内（稍微减小上限以提升速度）
        radius = max(0.003, min(radius, 0.015))  # 从0.02降到0.015
        
        return radius
    
    @staticmethod
    def standardize_point_cloud(positions):
        """
        标准化点云：居中并缩放到单位范围（优化版本）
        
        Args:
            positions: (N, 3) 位置数组
            
        Returns:
            standardized: 标准化后的位置
            center: 原始中心
            scale: 原始缩放
        """
        # 使用更高效的计算方式
        center = np.mean(positions, axis=0, dtype=np.float32)
        positions_centered = positions - center
        scale = np.max(np.abs(positions_centered))
        
        # 避免除法，使用乘法（更快）
        if scale > 1e-8:
            inv_scale = 1.0 / scale
            standardized = positions_centered * inv_scale
        else:
            standardized = positions_centered
        
        return standardized, center, scale
    
    def _create_base_scene_dict(self, center, bbox_size, camera_origin):
        """
        创建基础场景配置（相机、积分器等）
        
        Args:
            center: 点云中心点 (3,)
            bbox_size: 边界框大小
            camera_origin: 相机位置 (3,)
            
        Returns:
            scene_dict: 基础场景字典
        """
        return {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': 2  # 使用较低的深度以提升渲染速度
            },
            'sensor': {
                'type': 'perspective',
                'fov': 30,
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=camera_origin.tolist(),
                    target=center.tolist(),
                    up=[0, 0, 1]
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': self.image_width,
                    'height': self.image_height,
                    'rfilter': {'type': 'box'}  # 使用box filter，比gaussian快
                },
                'sampler': {
                    'type': 'independent',  # Mitsuba 3标准采样器
                    'sample_count': self.samples
                }
            }
        }
    
    def _add_environment(self, scene_dict, bbox_min, bbox_max, center, bbox_size):
        """
        添加地板和背景光源
        
        Args:
            scene_dict: 场景字典（会被修改）
            bbox_min: 边界框最小值 (3,)
            bbox_max: 边界框最大值 (3,)
            center: 点云中心点 (3,)
            bbox_size: 边界框大小
        """
        # 创建地面（增大尺寸以避免露出黑色背景）
        floor_z = bbox_min[2] - bbox_size * 0.1
        scene_dict['floor'] = {
            'type': 'rectangle',
            'to_world': (mi.ScalarTransform4f.scale([bbox_size * 5, bbox_size * 5, 1.0])
                        @ mi.ScalarTransform4f.translate([0, 0, floor_z])),
            'bsdf': {
                'type': 'roughplastic',
                'distribution': 'ggx',
                'alpha': 0.1,
                'int_ior': 1.46,
                'diffuse_reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
            }
        }
        
        # 创建背景光源（降低亮度）
        scene_dict['background'] = {
            'type': 'rectangle',
            'to_world': (mi.ScalarTransform4f.scale([bbox_size * 3, bbox_size * 3, 1.0])
                        @ mi.ScalarTransform4f.look_at(
                            origin=[0, 0, center[2] + bbox_size * 2],
                            target=center.tolist(),
                            up=[0, 1, 0]
                        )),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [1.8, 1.8, 1.8]}
            }
        }
    
    def _create_colored_ply(self, positions, colors):
        """
        创建带颜色的PLY文件（直接二进制写入，避免Python序列化开销）
        
        Args:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组，范围[0, 1]
            
        Returns:
            ply_file_path: 临时PLY文件路径
        """
        num_points = len(positions)
        
        # 将颜色从[0,1]转换为[0,255]的uint8
        colors_uint8 = (colors * 255).astype(np.uint8)
        
        # 创建临时文件
        import tempfile
        tmp_ply = tempfile.NamedTemporaryFile(mode='wb', suffix='.ply', delete=False)
        tmp_ply_path = tmp_ply.name
        
        # 写入ASCII header
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        tmp_ply.write(header.encode('ascii'))
        
        # 直接写入二进制数据（使用结构化数组，避免循环）
        # 定义dtype：x(4) + y(4) + z(4) + red(1) + green(1) + blue(1) = 15字节
        dtype = np.dtype([
            ('x', '<f4'),      # little-endian float32
            ('y', '<f4'),
            ('z', '<f4'),
            ('red', 'u1'),    # uint8
            ('green', 'u1'),
            ('blue', 'u1')
        ])
        
        # 创建结构化数组
        vertex_data = np.empty(num_points, dtype=dtype)
        vertex_data['x'] = positions[:, 0].astype(np.float32)
        vertex_data['y'] = positions[:, 1].astype(np.float32)
        vertex_data['z'] = positions[:, 2].astype(np.float32)
        vertex_data['red'] = colors_uint8[:, 0]
        vertex_data['green'] = colors_uint8[:, 1]
        vertex_data['blue'] = colors_uint8[:, 2]
        
        # 直接写入二进制数据（非常快）
        vertex_data.tofile(tmp_ply)
        
        tmp_ply.close()
        
        return tmp_ply_path
    
    def _build_scene_xml_from_ply(self, ply_file_path, radius, bbox_min, bbox_max, center, bbox_size, 
                                   camera_origin, add_environment):
        """
        从PLY文件生成XML场景（优化的批量字符串生成，避免Python循环开销）
        
        Args:
            ply_file_path: PLY文件路径
            radius: 粒子半径
            bbox_min: 边界框最小值
            bbox_max: 边界框最大值
            center: 点云中心
            bbox_size: 边界框大小
            camera_origin: 相机位置
            add_environment: 是否添加地板和背景光源
            
        Returns:
            xml_string: XML场景字符串
        """
        # 快速读取PLY文件（使用二进制读取，避免PlyData开销）
        num_vertices, header_size = self._read_ply_header_fast(ply_file_path)
        positions, colors = self._read_colored_ply_binary_fast(ply_file_path, num_vertices, header_size)
        num_points = len(positions)
        
        # 构建XML头部
        xml_parts = ['<scene version="0.6.0">']
        
        # 积分器 (使用path积分器，较低的max_depth以提升渲染速度)
        xml_parts.append('    <integrator type="path">')
        xml_parts.append('        <integer name="maxDepth" value="2"/>')
        xml_parts.append('    </integrator>')
        
        # 传感器（相机）
        xml_parts.append('    <sensor type="perspective">')
        xml_parts.append(f'        <float name="fov" value="30"/>')
        xml_parts.append('        <transform name="toWorld">')
        xml_parts.append(f'            <lookat origin="{camera_origin[0]:.6f},{camera_origin[1]:.6f},{camera_origin[2]:.6f}" '
                        f'target="{center[0]:.6f},{center[1]:.6f},{center[2]:.6f}" up="0,0,1"/>')
        xml_parts.append('        </transform>')
        xml_parts.append('        <film type="hdrfilm">')
        xml_parts.append(f'            <integer name="width" value="{self.image_width}"/>')
        xml_parts.append(f'            <integer name="height" value="{self.image_height}"/>')
        # 使用box filter（比gaussian快，虽然质量略低但速度更快）
        xml_parts.append('            <rfilter type="box"/>')
        xml_parts.append('        </film>')
        # 使用independent采样器（Mitsuba 3标准）
        xml_parts.append('        <sampler type="independent">')
        xml_parts.append(f'            <integer name="sampleCount" value="{self.samples}"/>')
        xml_parts.append('        </sampler>')
        xml_parts.append('    </sensor>')
        
        # 使用merge shape合并所有粒子
        xml_parts.append('    <shape type="merge">')
        
        # 优化的批量字符串生成（使用列表推导式和模板，避免循环中的字符串拼接）
        print(f'    Creating {num_points:,} particle shapes (optimized batch generation)...', end=' ', flush=True)
        
        # 使用字符串模板和列表推导式，比循环中的字符串拼接快得多
        particle_template = '''        <shape type="sphere">
            <float name="radius" value="{radius}"/>
            <transform name="toWorld">
                <translate x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{r:.6f},{g:.6f},{b:.6f}"/>
            </bsdf>
        </shape>'''
        
        radius_str = f'{radius:.6f}'
        
        # 批量生成所有粒子的XML（列表推导式比循环快，且内存效率更高）
        # 虽然还是需要遍历，但列表推导式比显式循环+字符串拼接快得多
        particle_xmls = [
            particle_template.format(
                radius=radius_str,
                x=pos[0], y=pos[1], z=pos[2],
                r=color[0], g=color[1], b=color[2]
            )
            for pos, color in zip(positions, colors)
        ]
        
        xml_parts.extend(particle_xmls)
        xml_parts.append('    </shape>')
        print('Done')
        
        # 添加环境（地板和背景光源）
        if add_environment:
            floor_z = bbox_min[2] - bbox_size * 0.1
            xml_parts.append('    <shape type="rectangle">')
            xml_parts.append(f'        <transform name="toWorld">')
            xml_parts.append(f'            <scale x="{bbox_size * 5:.6f}" y="{bbox_size * 5:.6f}" z="1.0"/>')
            xml_parts.append(f'            <translate x="0" y="0" z="{floor_z:.6f}"/>')
            xml_parts.append(f'        </transform>')
            xml_parts.append('        <bsdf type="roughplastic">')
            xml_parts.append('            <string name="distribution" value="ggx"/>')
            xml_parts.append('            <float name="alpha" value="0.1"/>')
            xml_parts.append('            <float name="intIOR" value="1.46"/>')
            xml_parts.append('            <rgb name="diffuseReflectance" value="1.0,1.0,1.0"/>')
            xml_parts.append('        </bsdf>')
            xml_parts.append('    </shape>')
            
            bg_z = center[2] + bbox_size * 2
            xml_parts.append('    <shape type="rectangle">')
            xml_parts.append(f'        <transform name="toWorld">')
            xml_parts.append(f'            <scale x="{bbox_size * 3:.6f}" y="{bbox_size * 3:.6f}" z="1.0"/>')
            xml_parts.append(f'            <lookat origin="0,0,{bg_z:.6f}" target="{center[0]:.6f},{center[1]:.6f},{center[2]:.6f}" up="0,1,0"/>')
            xml_parts.append(f'        </transform>')
            xml_parts.append('        <emitter type="area">')
            xml_parts.append('            <rgb name="radiance" value="2.0,2.0,2.0"/>')
            xml_parts.append('        </emitter>')
            xml_parts.append('    </shape>')
        
        xml_parts.append('</scene>')
        print('Done')
        
        return '\n'.join(xml_parts)
    
    def build_scene(self, positions, colors, radius, bbox_min=None, bbox_max=None, center=None, bbox_size=None, add_environment=True):
        """
        使用XML字符串生成场景（高性能版本），避免Python字典构建的性能瓶颈
        
        Args:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组
            radius: 粒子半径
            bbox_min: 边界框最小值（可选，用于分批渲染时保持场景一致）
            bbox_max: 边界框最大值（可选，用于分批渲染时保持场景一致）
            center: 点云中心（可选，用于分批渲染时保持场景一致）
            bbox_size: 边界框大小（可选，用于分批渲染时保持场景一致）
            add_environment: 是否添加地板和背景光源（默认True）
            
        Returns:
            scene: Mitsuba场景对象
        """
        # 计算点云边界框以设置合适的相机位置（优化：一次性计算）
        if bbox_min is None or bbox_max is None:
            bbox_min = np.min(positions, axis=0)
            bbox_max = np.max(positions, axis=0)
        if center is None:
            center = (bbox_min + bbox_max) * 0.5  # 使用乘法代替除法
        if bbox_size is None:
            bbox_size = np.max(bbox_max - bbox_min)
        
        # 相机位置：在点云上方和侧面（增加距离）
        camera_distance = bbox_size * 3.5
        camera_origin = center + np.array([camera_distance * 0.7, 
                                           camera_distance * 0.7, 
                                           camera_distance * 0.5])
        
        # 新方法：创建带颜色的PLY文件，然后从PLY生成XML
        # 这样可以跳过大量的字符串格式化操作
        print('    Creating colored PLY file...', end=' ', flush=True)
        tmp_ply_path = self._create_colored_ply(positions, colors)
        print('Done')
        
        # 从PLY文件生成简化的XML
        xml_string = self._build_scene_xml_from_ply(tmp_ply_path, radius, bbox_min, bbox_max, 
                                                    center, bbox_size, camera_origin, add_environment)
        
        # 写入临时XML文件并加载
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_file:
            tmp_file.write(xml_string)
            tmp_xml_path = tmp_file.name
        
        try:
            print('    Loading scene into Mitsuba...', end=' ', flush=True)
            scene = mi.load_file(tmp_xml_path)
            print('Done')
            return scene
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_xml_path)
                os.unlink(tmp_ply_path)
            except:
                pass
    
    def render(self, positions, normals, batch_indices=None, use_batch_rendering=False):
        """
        渲染点云
        
        使用merge shape优化，一次性渲染所有粒子，性能比独立shape快>100倍。
        如果指定use_batch_rendering=True且batch_indices不为None，则按batch_idx分批渲染。
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组
            batch_indices: (N,) 批次索引数组（可选）
            use_batch_rendering: 是否使用分批渲染（基于batch_idx）
            
        Returns:
            image: 渲染的图像
        """
        # 标准化点云
        print('  Standardizing point cloud...', end=' ', flush=True)
        positions_std, center, scale = self.standardize_point_cloud(positions)
        print('Done')
        
        # 法向量到RGB映射
        print('  Converting normals to RGB colors...', end=' ', flush=True)
        colors = self.normal_to_rgb(normals)
        print('Done')
        
        # 计算粒子半径
        print('  Computing particle radius...', end=' ', flush=True)
        if self.particle_radius is None:
            radius = self.compute_adaptive_radius(positions_std)
        else:
            radius = self.particle_radius
        print('Done')
        
        # 计算全局边界框（用于保持场景一致性，优化：一次性计算）
        bbox_min = np.min(positions_std, axis=0)
        bbox_max = np.max(positions_std, axis=0)
        bbox_center = (bbox_min + bbox_max) * 0.5  # 使用乘法代替除法
        bbox_size = np.max(bbox_max - bbox_min)
        
        # 如果使用分批渲染且batch_indices可用
        if use_batch_rendering and batch_indices is not None:
            # 检查是否有single_batch或max_batches参数（通过实例变量传递）
            single_batch_id = getattr(self, '_single_batch_id', None)
            max_batches = getattr(self, '_max_batches', None)
            return self._render_batched_by_batch_idx(
                positions_std, colors, radius, batch_indices,
                bbox_min, bbox_max, bbox_center, bbox_size,
                single_batch_id=single_batch_id,
                max_batches=max_batches
            )
        
        # 一次性渲染所有粒子
        # 注意：使用merge shape优化后，一次性渲染已经非常高效（比独立shape快>100倍）
        scene = self.build_scene(positions_std, colors, radius, 
                                bbox_min, bbox_max, bbox_center, bbox_size,
                                add_environment=True)
        
        # 渲染场景，使用进度回调显示进度
        print('  Rendering scene (this may take a while)...', flush=True)
        
        # 定义进度回调函数
        last_progress_percent = -1
        def progress_callback(progress):
            """渲染进度回调函数"""
            nonlocal last_progress_percent
            progress_percent = int(progress * 100)
            # 只在进度变化时更新，避免过于频繁的打印
            if progress_percent != last_progress_percent:
                print(f'\r  Rendering progress: {progress_percent}%', end='', flush=True)
                last_progress_percent = progress_percent
        
        # 尝试使用进度回调渲染
        try:
            image = mi.render(scene, progress=progress_callback)
            print('\r  Rendering progress: 100% - Done', flush=True)
        except TypeError:
            # 如果当前版本的Mitsuba不支持progress参数，回退到普通渲染
            print('  Rendering...', end=' ', flush=True)
            image = mi.render(scene)
            print('Done')
        
        return image
    
    def _render_batched_by_batch_idx(self, positions, colors, radius, batch_indices,
                                     bbox_min, bbox_max, center, bbox_size,
                                     single_batch_id=None, max_batches=None):
        """
        按batch_idx收集粒子并一次性渲染
        
        Args:
            positions: (N, 3) 标准化后的位置数组
            colors: (N, 3) RGB颜色数组
            radius: 粒子半径
            batch_indices: (N,) 批次索引数组
            bbox_min: 全局边界框最小值
            bbox_max: 全局边界框最大值
            center: 全局中心点
            bbox_size: 全局边界框大小
            single_batch_id: 如果指定，只渲染这个batch ID（用于测试）
            max_batches: 如果指定，最多渲染这么多batch（用于测试）
            
        Returns:
            image: 渲染的图像
        """
        # 获取唯一的批次ID并排序
        unique_batches = np.unique(batch_indices)
        unique_batches = np.sort(unique_batches)
        total_batches = len(unique_batches)
        
        # 确定要渲染的batch列表
        if single_batch_id is not None:
            if single_batch_id not in unique_batches:
                raise ValueError(f'Batch ID {single_batch_id} not found. Available batch IDs: {unique_batches[0]} to {unique_batches[-1]}')
            selected_batches = [single_batch_id]
            num_batches = 1
            print(f'  Collecting particles from batch {single_batch_id}...')
        elif max_batches is not None:
            # 限制渲染的batch数量
            selected_batches = unique_batches[:max_batches]
            num_batches = len(selected_batches)
            print(f'  Collecting particles from {num_batches} batches (out of {total_batches} total)...')
        else:
            selected_batches = unique_batches
            num_batches = total_batches
            print(f'  Collecting particles from all {num_batches} batches...')
        
        # 收集所有指定batch的粒子
        batch_mask = np.isin(batch_indices, selected_batches)
        selected_positions = positions[batch_mask]
        selected_colors = colors[batch_mask]
        num_particles = len(selected_positions)
        
        print(f'  Total particles to render: {num_particles:,}')
        
        # 一次性构建场景并渲染
        scene = self.build_scene(selected_positions, selected_colors, radius,
                                bbox_min, bbox_max, center, bbox_size,
                                add_environment=True)
        
        # 渲染场景
        print('  Rendering scene (this may take a while)...', flush=True)
        
        # 定义进度回调函数
        last_progress_percent = -1
        def progress_callback(progress):
            """渲染进度回调函数"""
            nonlocal last_progress_percent
            progress_percent = int(progress * 100)
            # 只在进度变化时更新，避免过于频繁的打印
            if progress_percent != last_progress_percent:
                print(f'\r  Rendering progress: {progress_percent}%', end='', flush=True)
                last_progress_percent = progress_percent
        
        # 尝试使用进度回调渲染
        try:
            image = mi.render(scene, progress=progress_callback)
            print('\r  Rendering progress: 100% - Done', flush=True)
        except TypeError:
            # 如果当前版本的Mitsuba不支持progress参数，回退到普通渲染
            print('  Rendering...', end=' ', flush=True)
            image = mi.render(scene)
            print('Done')
        
        return image
    
    
    def save_image(self, output_file_path, image):
        """
        保存渲染图像
        
        Args:
            output_file_path: 输出文件路径（不含扩展名）
            image: Mitsuba渲染的图像
        """
        mi.util.write_bitmap(f'{output_file_path}.png', image)
    
    def process(self, use_batch_rendering=False, single_batch_id=None):
        """
        处理单帧点云
        
        Args:
            use_batch_rendering: 是否使用基于batch_idx的分批渲染
            single_batch_id: 如果指定，只渲染这个batch ID（用于测试）
        """
        # 如果指定了single_batch_id，设置实例变量
        if single_batch_id is not None:
            self._single_batch_id = single_batch_id
        
        # 加载点云
        positions, normals, batch_indices = self.load_point_cloud()
        
        # 渲染
        image = self.render(positions, normals, batch_indices, use_batch_rendering)
        
        # 保存
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, self.filename)
        else:
            output_file_path = os.path.join(self.folder, self.filename)
        
        self.save_image(output_file_path, image)
        
        return output_file_path


def batch_render(input_folder='trajectory_ply', 
                 output_folder='render_output',
                 pattern='frame_*.ply',
                 start_frame=None,
                 end_frame=None,
                 image_width=1920,
                 image_height=1080,
                 samples=64,
                 use_batch_rendering=False,
                 single_batch_id=None,
                 max_batches=None):
    """
    批量渲染PLY文件
    
    Args:
        input_folder: 输入文件夹
        output_folder: 输出文件夹
        pattern: 文件匹配模式
        start_frame: 起始帧号（None表示从第一个开始）
        end_frame: 结束帧号（None表示到最后一个）
        image_width: 图像宽度
        image_height: 图像高度
        samples: 采样数
        use_batch_rendering: 是否使用基于batch_idx的分批渲染
        single_batch_id: 如果指定，只渲染这个batch ID（用于测试，需要use_batch_rendering=True）
        max_batches: 如果指定，最多渲染这么多batch（用于测试，需要use_batch_rendering=True）
    """
    import glob
    
    # 初始化Mitsuba GPU
    NormalColorRenderer.init_mitsuba_variant()
    
    # 查找所有PLY文件
    ply_files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    
    if not ply_files:
        print(f'No files found matching pattern: {os.path.join(input_folder, pattern)}')
        return
    
    # 过滤帧范围
    if start_frame is not None or end_frame is not None:
        filtered_files = []
        for f in ply_files:
            # 从文件名提取帧号
            basename = os.path.basename(f)
            try:
                # 假设文件名格式为 frame_XXXX.ply
                frame_num = int(basename.split('_')[1].split('.')[0])
                if start_frame is not None and frame_num < start_frame:
                    continue
                if end_frame is not None and frame_num > end_frame:
                    continue
                filtered_files.append(f)
            except:
                # 如果无法解析帧号，包含该文件
                filtered_files.append(f)
        ply_files = filtered_files
    
    total_files = len(ply_files)
    print('=' * 60)
    print(f'Found {total_files} file(s) to render')
    print(f'Input folder: {input_folder}')
    print(f'Output folder: {output_folder}')
    print(f'Image size: {image_width}x{image_height}')
    print(f'Samples per pixel: {samples}')
    print('=' * 60)
    
    os.makedirs(output_folder, exist_ok=True)
    
    successful = 0
    failed = 0
    
    # 单进程GPU渲染
    for idx, ply_file in enumerate(ply_files, 1):
            basename = os.path.basename(ply_file)
            
            print(f'\n[{idx}/{total_files}] ({idx*100//total_files}%) Processing: {basename}')
            print('-' * 60)
            
            try:
                renderer = NormalColorRenderer(
                    ply_file,
                    output_folder=output_folder,
                    image_width=image_width,
                    image_height=image_height,
                    samples=samples
                )
                
                # 加载点云
                print('  Loading point cloud...', end=' ', flush=True)
                positions, normals, batch_indices = renderer.load_point_cloud()
                print(f'Done ({len(positions):,} points)')
                
                # 处理和渲染
                renderer._single_batch_id = single_batch_id if use_batch_rendering else None
                renderer._max_batches = max_batches if use_batch_rendering else None
                image = renderer.render(positions, normals, batch_indices, use_batch_rendering)
                
                # 保存图像
                print('  Saving image...', end=' ', flush=True)
                output_path = os.path.join(output_folder, renderer.filename)
                renderer.save_image(output_path, image)
                print(f'Done -> {os.path.basename(output_path)}.png')
                
                # 清理内存
                del renderer, positions, normals, image
                import gc
                gc.collect()
                
                successful += 1
                print(f'  ✓ Successfully processed: {basename}')
                
            except Exception as e:
                failed += 1
                print(f'  ✗ Error processing {basename}: {str(e)}')
                import traceback
                traceback.print_exc()
    
    print('\n' + '=' * 60)
    print(f'Batch processing completed!')
    print(f'  Successful: {successful}/{total_files}')
    print(f'  Failed: {failed}/{total_files}')
    print(f'Output files saved to: {output_folder}/')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Render point clouds with normal-based coloring')
    parser.add_argument('--input', type=str, default='trajectory_ply',
                        help='Input folder containing PLY files')
    parser.add_argument('--output', type=str, default='render_output',
                        help='Output folder for rendered images')
    parser.add_argument('--pattern', type=str, default='frame_*.ply',
                        help='File pattern to match')
    parser.add_argument('--start', type=int, default=None,
                        help='Start frame number')
    parser.add_argument('--end', type=int, default=None,
                        help='End frame number')
    parser.add_argument('--width', type=int, default=1920,
                        help='Image width')
    parser.add_argument('--height', type=int, default=1080,
                        help='Image height')
    parser.add_argument('--samples', type=int, default=64,
                        help='Samples per pixel (default: 64 for faster rendering)')
    parser.add_argument('--use-batch-rendering', action='store_true',
                        help='Use batch_idx-based batch rendering (256 batches, 2048 particles each)')
    parser.add_argument('--single-batch', type=int, default=None,
                        help='Render only a single batch ID (0-255) for testing (requires --use-batch-rendering)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to render (requires --use-batch-rendering)')
    args = parser.parse_args()
    
    # 验证参数
    if args.single_batch is not None and not args.use_batch_rendering:
        print('Warning: --single-batch requires --use-batch-rendering. Ignoring --single-batch.')
        args.single_batch = None
    
    if args.max_batches is not None and not args.use_batch_rendering:
        print('Warning: --max-batches requires --use-batch-rendering. Ignoring --max-batches.')
        args.max_batches = None
    
    if args.single_batch is not None and args.max_batches is not None:
        print('Warning: --single-batch and --max-batches cannot be used together. Using --single-batch.')
        args.max_batches = None
    
    batch_render(
        input_folder=args.input,
        output_folder=args.output,
        pattern=args.pattern,
        start_frame=args.start,
        end_frame=args.end,
        image_width=args.width,
        image_height=args.height,
        samples=args.samples,
        use_batch_rendering=args.use_batch_rendering,
        single_batch_id=args.single_batch,
        max_batches=args.max_batches
    )

