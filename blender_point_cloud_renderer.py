import numpy as np
import os

# 尝试导入bpy（Blender Python API）
try:
    import bpy
    from mathutils import Vector
    BPY_AVAILABLE = True
except ImportError:
    BPY_AVAILABLE = False
    print("Warning: bpy not available. This script must be run within Blender or via 'blender --python' command.")


class BlenderPointCloudRenderer:
    """
    Blender Python API 
    """
    
    def __init__(self, file_path, output_folder=None, 
                 particle_radius=None, 
                 image_width=1920, image_height=1080,
                 samples=64, max_points=None,
                 particle_color=(0.3, 0.3, 0.3)):
        """
        初始化渲染器
        
        Args:
            file_path: PLY文件路径
            output_folder: 输出文件夹
            particle_radius: 粒子半径（None表示自适应）
            image_width: 图像宽度
            image_height: 图像高度
            samples: 采样数
            max_points: 最大点数（None表示不限制，用于LOD降采样）
            particle_color: 粒子颜色 (R, G, B)，范围[0, 1]，默认灰色(0.3, 0.3, 0.3)
        """
        if not BPY_AVAILABLE:
            raise RuntimeError("bpy is not available. Please run this script within Blender or via 'blender --python' command.")
        
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder
        self.particle_radius = particle_radius
        self.image_width = image_width
        self.image_height = image_height
        self.samples = samples
        self.max_points = max_points  # LOD降采样目标点数
        self.particle_color = particle_color  # 粒子颜色
    
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
        使用np.fromfile快速读取PLY二进制数据
        
        Args:
            file_path: PLY文件路径
            num_vertices: 顶点数量
            header_size: header大小（字节）
            
        Returns:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组（不使用，保持接口一致）
            batch_indices: (N,) 批次索引数组
        """
        dtype = np.dtype([
            ('x', '<f4'),      # little-endian float32
            ('y', '<f4'),
            ('z', '<f4'),
            ('nx', '<f4'),
            ('ny', '<f4'),
            ('nz', '<f4'),
            ('batch_idx', '<i4')
        ])
        
        # 读取二进制数据
        with open(file_path, 'rb') as f:
            f.seek(header_size)  # 跳过header
            data = np.fromfile(f, dtype=dtype, count=num_vertices)
        
        # 提取位置、法向量和批次索引
        positions = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
        normals = np.column_stack([data['nx'], data['ny'], data['nz']]).astype(np.float32)
        batch_indices = data['batch_idx'].astype(np.int32)
        
        return positions, normals, batch_indices
    
    def load_point_cloud(self):
        """加载PLY文件，返回位置、法向量和批次索引"""
        file_extension = os.path.splitext(self.file_path)[1]
        
        if file_extension == '.ply':
            num_vertices, header_size = self._read_ply_header_fast(self.file_path)
            positions, normals, batch_indices = self._read_ply_binary_fast(
                self.file_path, num_vertices, header_size
            )
            
            return positions, normals, batch_indices
        else:
            raise ValueError(f'Unsupported file format: {file_extension}')
    
    @staticmethod
    def downsample_points(positions, normals, target_points=50000):
        """
        降采样点云以提升渲染性能（LOD）
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组（不使用，保持接口一致）
            target_points: 目标点数
            
        Returns:
            downsampled_positions: 降采样后的位置
            downsampled_normals: 降采样后的法向量
        """
        if len(positions) <= target_points:
            return positions, normals
        
        # 均匀采样
        step = len(positions) // target_points
        indices = np.arange(0, len(positions), step)
        
        return positions[indices], normals[indices]
    
    @staticmethod
    def compute_adaptive_radius(positions):
        """
        根据点云计算自适应粒子半径
        
        Args:
            positions: (N, 3) 位置数组
            
        Returns:
            radius: 自适应半径
        """
        bbox_min = np.min(positions, axis=0)
        bbox_max = np.max(positions, axis=0)
        bbox_size = np.max(bbox_max - bbox_min)
        num_points = len(positions)
        
        radius = bbox_size / (num_points ** 0.33) * 0.30
        radius = max(0.005, min(radius, 0.035))
        
        return radius
    
    @staticmethod
    def standardize_point_cloud(positions):
        """
        标准化点云：居中并缩放到单位范围
        
        Args:
            positions: (N, 3) 位置数组
            
        Returns:
            standardized: 标准化后的位置
            center: 原始中心
            scale: 原始缩放
        """
        center = np.mean(positions, axis=0, dtype=np.float32)
        positions_centered = positions - center
        scale = np.max(np.abs(positions_centered))
        
        if scale > 1e-8:
            inv_scale = 1.0 / scale
            standardized = positions_centered * inv_scale
        else:
            standardized = positions_centered
        
        return standardized, center, scale
    
    def init_blender_scene(self):
        """
        初始化Blender场景：清理默认场景，设置渲染引擎和基本参数
        """
        # 清理默认场景
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # 设置渲染引擎为Cycles
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        
        # 设置渲染分辨率
        scene.render.resolution_x = self.image_width
        scene.render.resolution_y = self.image_height
        scene.render.resolution_percentage = 100
        
        # 设置Cycles渲染参数
        scene.cycles.samples = max(self.samples, 64)
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.03
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, 'denoiser'):
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        scene.cycles.max_bounces = 2
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 1
        scene.cycles.transmission_bounces = 0
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        scene.cycles.use_fast_gi = True
        scene.cycles.fast_gi_method = 'REPLACE'
        scene.cycles.use_persistent_data = False
        scene.cycles.preview_samples = 16
        if hasattr(scene.cycles, 'sampling_pattern'):
            try:
                scene.cycles.sampling_pattern = 'TABULATED_SOBOL'
            except (ValueError, TypeError):
                pass
        
        # 使用GPU加速（如果可用）
        try:
            prefs = bpy.context.preferences
            cycles_prefs = prefs.addons['cycles'].preferences
            devices = cycles_prefs.get_devices()
            
            if devices is None:
                try:
                    scene.cycles.device = 'GPU'
                    scene.cycles.tile_size = 256
                    print('    Using GPU (device type set to GPU)')
                except:
                    scene.cycles.device = 'CPU'
                    scene.cycles.tile_size = 512
                    print('    Using CPU (could not set GPU device type)')
            else:
                gpu_devices = []
                for device in devices:
                    if device and device.type in ('CUDA', 'OPENCL', 'OPTIX', 'HIP', 'METAL', 'ONEAPI'):
                        if not device.use:
                            device.use = True
                        gpu_devices.append(device)
                
                has_gpu = len(gpu_devices) > 0
                
                if has_gpu:
                    scene.cycles.device = 'GPU'
                    scene.cycles.tile_size = 256
                    device_names = [f"{d.name} ({d.type})" for d in gpu_devices if d.use]
                    print(f'    Using GPU: {", ".join(device_names)}')
                else:
                    try:
                        scene.cycles.device = 'GPU'
                        scene.cycles.tile_size = 256
                        print('    Using GPU (device list empty, but GPU mode enabled)')
                    except:
                        scene.cycles.device = 'CPU'
                        scene.cycles.tile_size = 512
                        print('    Using CPU (no GPU devices found)')
        except Exception as e:
            try:
                scene.cycles.device = 'GPU'
                scene.cycles.tile_size = 256
                print(f'    Using GPU (fallback method, error: {e})')
            except:
                scene.cycles.device = 'CPU'
                scene.cycles.tile_size = 512
                print(f'    Using CPU (GPU configuration failed: {e})')
        
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
        
        scene.render.use_motion_blur = False
        
        world = scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg is None:
            bg = world.node_tree.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs["Strength"].default_value = 20.0
        
        return scene
    
    def setup_camera(self, center, bbox_size, camera_position=None):
        """
        设置相机位置和朝向
        
        Args:
            center: 点云中心点 (3,)
            bbox_size: 边界框大小
            camera_position: 相机位置 (3,)，如果为None则根据bbox_size计算
        """
        if camera_position is None:
            camera_distance = bbox_size * 5.5
            camera_origin = center + np.array([
                camera_distance * 0.7,
                camera_distance * 0.7,
                camera_distance * 0.5
            ])
        else:
            camera_origin = camera_position
        
        center_vec = Vector(center)
        camera_origin_vec = Vector(camera_origin)
        
        if 'Camera' not in bpy.data.objects:
            bpy.ops.object.camera_add()
        camera = bpy.data.objects['Camera']
        camera.location = camera_origin_vec
        
        direction = center_vec - camera_origin_vec
        if direction.length > 0:
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
        else:
            camera.rotation_euler = (np.radians(90), 0, 0)
        
        camera.data.type = 'PERSP'
        camera.data.angle = np.radians(30)
        
        bpy.context.scene.camera = camera
        return camera
    
    def add_environment(self, bbox_min, bbox_max, center, bbox_size):
        """
        添加地板和背景光源
        
        Args:
            bbox_min: 边界框最小值 (3,)
            bbox_max: 边界框最大值 (3,)
            center: 点云中心点 (3,)
            bbox_size: 边界框大小
        """
        floor_z = float(bbox_min[2] - bbox_size * 0.1)
        bpy.ops.mesh.primitive_plane_add(
            size=bbox_size * 12,
            location=(float(center[0]), float(center[1]), floor_z)
        )
        floor = bpy.context.active_object
        floor.name = "BackgroundFloor"
        
        floor_mat = bpy.data.materials.new(name="BackgroundFloorMaterial")
        floor_mat.use_nodes = True
        bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.9
            bsdf.inputs["Metallic"].default_value = 0.0
        floor.data.materials.append(floor_mat)
        
        bpy.ops.object.light_add(type='AREA', location=(
            float(center[0]), 
            float(center[1]), 
            float(center[2] + bbox_size * 2)
        ))
        top_light = bpy.context.active_object
        top_light.name = "TopLight"
        top_light.data.energy = 300.0
        top_light.data.size = bbox_size * 1.5
        top_light.rotation_euler = (0, 0, 0)
    
    def create_point_cloud_simple(self, positions, normals, radius):
        """
        创建纯点云网格，使用Geometry Nodes实例化
        所有粒子使用相同的颜色
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组
            radius: 粒子半径
        """
        num_points = len(positions)
        
        # 创建网格对象
        mesh = bpy.data.meshes.new(name="PointCloud")
        obj = bpy.data.objects.new("PointCloud", mesh)
        bpy.context.collection.objects.link(obj)
        
        # 添加顶点
        mesh.vertices.add(num_points)
        mesh.vertices.foreach_set("co", positions.flatten())
        
        mesh.update()
        
        mat = bpy.data.materials.new(name="PointCloudMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        
        if bsdf and "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (*self.particle_color, 1.0)
        
        if bsdf:
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 0.3
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (*self.particle_color, 1.0)
            elif "Emission" in bsdf.inputs:
                bsdf.inputs["Emission"].default_value = (*self.particle_color, 1.0)
        
        obj.data.materials.append(mat)
        
        # 创建Geometry Nodes修改器
        if bpy.app.version >= (3, 0, 0):
            try:
                node_group_name = f"PointInstances_{id(obj)}"
                node_group = bpy.data.node_groups.new(name=node_group_name, type='GeometryNodeTree')
                node_group.inputs.new('NodeSocketGeometry', 'Geometry')
                node_group.outputs.new('NodeSocketGeometry', 'Geometry')
                
                input_node = node_group.nodes.new('NodeGroupInput')
                output_node = node_group.nodes.new('NodeGroupOutput')
                
                instance_node = node_group.nodes.new('GeometryNodeInstanceOnPoints')
                sphere_node = node_group.nodes.new('GeometryNodeMeshIcoSphere')
                
                sphere_node.inputs['Radius'].default_value = radius
                sphere_node.inputs['Subdivisions'].default_value = 1
                
                node_group.links.new(input_node.outputs['Geometry'], instance_node.inputs['Points'])
                node_group.links.new(sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
                
                realize_node = node_group.nodes.new('GeometryNodeRealizeInstances')
                node_group.links.new(instance_node.outputs['Instances'], realize_node.inputs['Geometry'])
                
                set_mat_node = node_group.nodes.new('GeometryNodeSetMaterial')
                set_mat_node.inputs['Material'].default_value = mat
                node_group.links.new(realize_node.outputs['Geometry'], set_mat_node.inputs['Geometry'])
                
                node_group.links.new(set_mat_node.outputs['Geometry'], output_node.inputs['Geometry'])
                
                mod = obj.modifiers.new(name="PointInstances", type='NODES')
                mod.node_group = node_group
            except Exception as e:
                print(f"Warning: Could not create Geometry Nodes modifier: {e}")
                import traceback
                traceback.print_exc()
        
        return obj
    
    def build_scene(self, positions, normals, radius, bbox_min=None, bbox_max=None, 
                    center=None, bbox_size=None, add_environment=True, camera_position=None):
        """
        构建Blender场景
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组
            radius: 粒子半径
            bbox_min: 边界框最小值（可选）
            bbox_max: 边界框最大值（可选）
            center: 点云中心（可选）
            bbox_size: 边界框大小（可选）
            add_environment: 是否添加地板和背景光源
            camera_position: 相机位置 (3,)，如果为None则根据bbox_size计算
        """
        if bbox_min is None or bbox_max is None:
            bbox_min = np.min(positions, axis=0)
            bbox_max = np.max(positions, axis=0)
        if center is None:
            center = (bbox_min + bbox_max) * 0.5
        if bbox_size is None:
            bbox_size = np.max(bbox_max - bbox_min)
        
        scene = self.init_blender_scene()
        self.setup_camera(center, bbox_size, camera_position)
        
        print(f'    Creating point cloud with {len(positions):,} points...', end=' ', flush=True)
        self.create_point_cloud_simple(positions, normals, radius)
        print('Done')
        
        if add_environment:
            self.add_environment(bbox_min, bbox_max, center, bbox_size)
        
        return scene
    
    def render(self, positions, normals, batch_indices=None, use_batch_rendering=False,
               single_batch_id=None, max_batches=None, camera_position=None):
        """
        渲染点云
        所有粒子使用相同的颜色
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组（不使用，但保持接口一致）
            batch_indices: (N,) 批次索引数组（可选）
            use_batch_rendering: 是否使用分批渲染（基于batch_idx）
            single_batch_id
            max_batches
            camera_position: 相机位置 (3,)，如果为None则根据bbox_size计算
            
        Returns:
            image_path: 渲染图像的路径
        """
        if self.max_points is not None and len(positions) > self.max_points:
            print(f'  Downsampling from {len(positions):,} to {self.max_points:,} points...', end=' ', flush=True)
            positions, normals = self.downsample_points(positions, normals, self.max_points)
            print('Done')
        
        print('  Standardizing point cloud...', end=' ', flush=True)
        positions_std, center, scale = self.standardize_point_cloud(positions)
        positions_std[:, 2] = -positions_std[:, 2]  # 上下翻转
        positions_std[:, 0] = -positions_std[:, 0]  # Z轴旋转180度
        positions_std[:, 1] = -positions_std[:, 1]
        print('Done')
        
        print('  Computing particle radius...', end=' ', flush=True)
        if self.particle_radius is None:
            radius = self.compute_adaptive_radius(positions_std)
        else:
            radius = self.particle_radius
        print(f'Done (radius={radius:.6f})')
        
        bbox_min = np.min(positions_std, axis=0)
        bbox_max = np.max(positions_std, axis=0)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bbox_size = np.max(bbox_max - bbox_min)
        
        if use_batch_rendering and batch_indices is not None:
            return self._render_batched_by_batch_idx(
                positions_std, normals, radius, batch_indices,
                bbox_min, bbox_max, bbox_center, bbox_size,
                single_batch_id=single_batch_id,
                max_batches=max_batches,
                camera_position=camera_position
            )
        
        print('  Building Blender scene...', flush=True)
        self.build_scene(positions_std, normals, radius, 
                        bbox_min, bbox_max, bbox_center, bbox_size,
                        add_environment=True, camera_position=camera_position)
        
        print('  Rendering scene...', end=' ', flush=True)
        
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, self.filename)
        else:
            output_file_path = os.path.join(self.folder, self.filename)
        
        scene = bpy.context.scene
        scene.render.filepath = output_file_path
        bpy.ops.render.render(write_still=True)
        
        print(f'Done -> {os.path.basename(output_file_path)}.png')
        
        return output_file_path
    
    def _render_batched_by_batch_idx(self, positions, normals, radius, batch_indices,
                                     bbox_min, bbox_max, center, bbox_size,
                                     single_batch_id=None, max_batches=None, camera_position=None):
        """
        按batch_idx收集粒子并一次性渲染
        所有粒子使用相同的颜色
        
        Args:
            positions: (N, 3) 标准化后的位置数组
            normals: (N, 3) 法向量数组（不使用，但保持接口一致）
            radius: 粒子半径
            batch_indices: (N,) 批次索引数组
            bbox_min: 全局边界框最小值
            bbox_max: 全局边界框最大值
            center: 全局中心点
            bbox_size: 全局边界框大小
            single_batch_id: 如果指定，只渲染这个batch ID（用于测试）
            max_batches: 如果指定，最多渲染这么多batch（用于测试）
            camera_position: 相机位置 (3,)，如果为None则根据bbox_size计算
            
        Returns:
            image_path: 渲染图像的路径
        """
        unique_batches = np.unique(batch_indices)
        unique_batches = np.sort(unique_batches)
        total_batches = len(unique_batches)
        
        if single_batch_id is not None:
            if single_batch_id not in unique_batches:
                raise ValueError(f'Batch ID {single_batch_id} not found. Available batch IDs: {unique_batches[0]} to {unique_batches[-1]}')
            selected_batches = [single_batch_id]
            num_batches = 1
            print(f'  Collecting particles from batch {single_batch_id}...')
        elif max_batches is not None:
            selected_batches = unique_batches[:max_batches]
            num_batches = len(selected_batches)
            print(f'  Collecting particles from {num_batches} batches (out of {total_batches} total)...')
        else:
            selected_batches = unique_batches
            num_batches = total_batches
            print(f'  Collecting particles from all {num_batches} batches...')
        
        batch_mask = np.isin(batch_indices, selected_batches)
        selected_positions = positions[batch_mask]
        selected_normals = normals[batch_mask]
        num_particles = len(selected_positions)
        
        print(f'  Total particles to render: {num_particles:,}')
        
        print('  Building Blender scene...', flush=True)
        self.build_scene(selected_positions, selected_normals, radius,
                        bbox_min, bbox_max, center, bbox_size,
                        add_environment=True, camera_position=camera_position)
        
        print('  Rendering scene...', end=' ', flush=True)
        
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, self.filename)
        else:
            output_file_path = os.path.join(self.folder, self.filename)
        
        scene = bpy.context.scene
        scene.render.filepath = output_file_path
        bpy.ops.render.render(write_still=True)
        
        print(f'Done -> {os.path.basename(output_file_path)}.png')
        
        return output_file_path
    
    def process(self, use_batch_rendering=False, single_batch_id=None, max_batches=None):
        """
        处理单帧点云
        
        Args:
            use_batch_rendering: 是否使用基于batch_idx的分批渲染
            single_batch_id: 如果指定，只渲染这个batch ID（用于测试）
            max_batches: 如果指定，最多渲染这么多batch（用于测试）
        """
        positions, normals, batch_indices = self.load_point_cloud()
        image_path = self.render(positions, normals, batch_indices, 
                                use_batch_rendering, single_batch_id, max_batches)
        return image_path


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
                 max_batches=None,
                 max_points=None,
                 particle_color=(0.3, 0.3, 0.3)):
    """
    批量渲染PLY文件（需要在Blender环境中运行）
    
    Args:
        input_folder: 输入文件夹
        output_folder: 输出文件夹
        pattern: 文件匹配模式
        start_frame: 起始帧号（None表示从第一个开始）
        end_frame: 结束帧号（None表示到最后一个）
        image_width: 图像宽度
        image_height: 图像高度
        samples: 采样数（Cycles渲染引擎）
        use_batch_rendering: 是否使用基于batch_idx的分批渲染
        single_batch_id: 如果指定，只渲染这个batch ID（用于测试，需要use_batch_rendering=True）
        max_batches: 如果指定，最多渲染这么多batch（用于测试，需要use_batch_rendering=True）
        max_points: 最大点数（LOD降采样）
        particle_color: 粒子颜色 (R, G, B)，范围[0, 1]，默认灰色(0.3, 0.3, 0.3)
    """
    import glob
    
    if not BPY_AVAILABLE:
        raise RuntimeError(
            "bpy is not available. Please run this script within Blender:\n"
            f"  blender --background --python {__file__} -- --input {input_folder} --output {output_folder}"
        )
    
    # 查找所有PLY文件
    ply_files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    
    if not ply_files:
        print(f'No files found matching pattern: {os.path.join(input_folder, pattern)}')
        return
    
    if start_frame is not None or end_frame is not None:
        filtered_files = []
        for f in ply_files:
            basename = os.path.basename(f)
            try:
                frame_num = int(basename.split('_')[1].split('.')[0])
                if start_frame is not None and frame_num < start_frame:
                    continue
                if end_frame is not None and frame_num > end_frame:
                    continue
                filtered_files.append(f)
            except:
                filtered_files.append(f)
        ply_files = filtered_files
    
    total_files = len(ply_files)
    print('=' * 60)
    print(f'Found {total_files} file(s) to render')
    print(f'Input folder: {input_folder}')
    print(f'Output folder: {output_folder}')
    print(f'Image size: {image_width}x{image_height}')
    print(f'Samples per pixel: {samples}')
    print(f'Render engine: Cycles')
    print(f'Particle color: RGB{particle_color}')
    if use_batch_rendering:
        if single_batch_id is not None:
            print(f'Batch rendering: Single batch ID {single_batch_id}')
        elif max_batches is not None:
            print(f'Batch rendering: Max {max_batches} batches')
        else:
            print('Batch rendering: All batches')
    print('=' * 60)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 预计算相机位置：使用第0帧和第299帧
    camera_positions = {}
    frame_0_file = None
    frame_299_file = None
    
    # 查找第0帧和第299帧的文件
    for ply_file in ply_files:
        basename = os.path.basename(ply_file)
        try:
            frame_num = int(basename.split('_')[1].split('.')[0])
            if frame_num == 0:
                frame_0_file = ply_file
            elif frame_num == 299:
                frame_299_file = ply_file
        except:
            pass
    
    # 计算第0帧和第299帧的相机位置
    if frame_0_file and frame_299_file:
        print('=' * 60)
        print('Precomputing camera positions from frame 0 and 299...')
        print('=' * 60)
        
        for frame_file, frame_id in [(frame_0_file, 0), (frame_299_file, 299)]:
            temp_renderer = BlenderPointCloudRenderer(
                frame_file,
                output_folder=None,
                image_width=image_width,
                image_height=image_height,
                samples=samples,
                max_points=max_points,
                particle_color=particle_color
            )
            positions, normals, _ = temp_renderer.load_point_cloud()
            
            if temp_renderer.max_points is not None and len(positions) > temp_renderer.max_points:
                positions, normals = temp_renderer.downsample_points(positions, normals, temp_renderer.max_points)
            
            positions_std, center, scale = temp_renderer.standardize_point_cloud(positions)
            positions_std[:, 2] = -positions_std[:, 2]
            positions_std[:, 0] = -positions_std[:, 0]
            positions_std[:, 1] = -positions_std[:, 1]
            
            bbox_min = np.min(positions_std, axis=0)
            bbox_max = np.max(positions_std, axis=0)
            bbox_center = (bbox_min + bbox_max) * 0.5
            bbox_size = np.max(bbox_max - bbox_min)
            
            # 第0帧使用更远的距离，第299帧使用正常距离
            if frame_id == 0:
                camera_distance = bbox_size * 5.5  # 第0帧拉远
            else:
                camera_distance = bbox_size * 3.5  # 第299帧正常距离
            
            camera_pos = bbox_center + np.array([
                camera_distance * 0.7,
                camera_distance * 0.7,
                camera_distance * 0.5
            ])
            camera_positions[frame_id] = camera_pos
            print(f'  Frame {frame_id}: camera position = ({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f})')
        
        print('=' * 60)
    
    successful = 0
    failed = 0
    
    for idx, ply_file in enumerate(ply_files, 1):
        basename = os.path.basename(ply_file)
        
        # 从文件名提取帧号
        try:
            frame_num = int(basename.split('_')[1].split('.')[0])
        except:
            frame_num = None
        
        print(f'\n[{idx}/{total_files}] ({idx*100//total_files}%) Processing: {basename}')
        print('-' * 60)
        
        try:
            renderer = BlenderPointCloudRenderer(
                ply_file,
                output_folder=output_folder,
                image_width=image_width,
                image_height=image_height,
                samples=samples,
                max_points=max_points,
                particle_color=particle_color
            )
            
            print('  Loading point cloud...', end=' ', flush=True)
            positions, normals, batch_indices = renderer.load_point_cloud()
            print(f'Done ({len(positions):,} points)')
            
            # 计算当前帧的相机位置（如果在0-299帧之间进行插值）
            camera_position = None
            if frame_num is not None and 0 in camera_positions and 299 in camera_positions:
                if frame_num == 0:
                    camera_position = camera_positions[0]
                elif frame_num == 299:
                    camera_position = camera_positions[299]
                elif 0 < frame_num < 299:
                    progress = frame_num / 299.0
                    start_pos = camera_positions[0]
                    end_pos = camera_positions[299]
                    camera_position = start_pos + (end_pos - start_pos) * progress
                    print(f'  Using interpolated camera position (progress: {progress:.3f})')
                else:
                    camera_position = camera_positions[299]  # 299帧之后使用第299帧的位置
            
            image_path = renderer.render(positions, normals, batch_indices, 
                                        use_batch_rendering, single_batch_id, max_batches,
                                        camera_position=camera_position)
            
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            
            for block in bpy.data.meshes:
                if block.users == 0:
                    bpy.data.meshes.remove(block)
            for block in bpy.data.materials:
                if block.users == 0:
                    bpy.data.materials.remove(block)
            for block in bpy.data.node_groups:
                if block.users == 0:
                    bpy.data.node_groups.remove(block)
            
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
    
    parser = argparse.ArgumentParser(description='Render point clouds with uniform coloring using Blender')
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
                        help='Samples per pixel (default: 64)')
    parser.add_argument('--use-batch-rendering', action='store_true',
                        help='Use batch_idx-based batch rendering (256 batches, 2048 particles each)')
    parser.add_argument('--single-batch', type=int, default=None,
                        help='Render only a single batch ID (0-255) for testing (requires --use-batch-rendering)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to render (requires --use-batch-rendering)')
    parser.add_argument('--max-points', type=int, default=None,
                        help='Maximum number of points to render (LOD downsampling, None=no limit)')
    parser.add_argument('--color', type=float, nargs=3, default=[0.3, 0.3, 0.3],
                        metavar=('R', 'G', 'B'),
                        help='Particle color RGB values (0.0-1.0, default: 0.3 0.3 0.3 for gray)')
    args = parser.parse_args()
    
    if args.single_batch is not None and not args.use_batch_rendering:
        print('Warning: --single-batch requires --use-batch-rendering. Ignoring --single-batch.')
        args.single_batch = None
    
    if args.max_batches is not None and not args.use_batch_rendering:
        print('Warning: --max-batches requires --use-batch-rendering. Ignoring --max-batches.')
        args.max_batches = None
    
    if args.single_batch is not None and args.max_batches is not None:
        print('Warning: --single-batch and --max-batches cannot be used together. Using --single-batch.')
        args.max_batches = None
    
    particle_color = tuple(np.clip(args.color, 0.0, 1.0))
    if particle_color != tuple(args.color):
        print(f'Warning: Color values clipped to [0.0, 1.0]: {particle_color}')
    
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
        max_batches=args.max_batches,
        max_points=args.max_points,
        particle_color=particle_color
    )

