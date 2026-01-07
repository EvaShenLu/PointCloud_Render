import numpy as np
import os
import sys

# 尝试导入bpy（Blender Python API）
try:
    import bpy
    from mathutils import Vector
    BPY_AVAILABLE = True
except ImportError:
    BPY_AVAILABLE = False
    print("Warning: bpy not available. This script must be run within Blender or via 'blender --python' command.")


class BlenderNormalRenderer:
    """
    使用Blender Python API的点云渲染器
    将法向量映射为RGB颜色，使用Blender的高效实例化功能渲染
    """
    
    def __init__(self, file_path, output_folder=None, 
                 particle_radius=None, 
                 image_width=1920, image_height=1080,
                 samples=64, engine='cycles', max_points=None):
        """
        初始化渲染器
        
        Args:
            file_path: PLY文件路径
            output_folder: 输出文件夹
            particle_radius: 粒子半径（None表示自适应）
            image_width: 图像宽度
            image_height: 图像高度
            samples: 采样数（Cycles渲染引擎）
            engine: 渲染引擎 ('cycles' 或 'eevee')
            max_points: 最大点数（None表示不限制，用于LOD降采样）
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
        self.engine = engine.lower()
        self.max_points = max_points  # LOD降采样目标点数
        
        if self.engine not in ['cycles', 'eevee']:
            raise ValueError(f"Unsupported engine: {engine}. Must be 'cycles' or 'eevee'")
    
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
            normals: (N, 3) 法向量数组
            batch_indices: (N,) 批次索引数组或None
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
        """
        加载PLY文件，返回位置、法向量和批次索引
        """
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
    def normal_to_rgb(normals):
        """
        将法向量映射到RGB颜色
        
        Args:
            normals: (N, 3) 法向量数组
            
        Returns:
            rgb: (N, 3) RGB颜色数组，范围[0, 1]
        """
        # 归一化法向量
        norms = np.sqrt(np.sum(normals ** 2, axis=1, keepdims=True))
        normalized = normals / (norms + 1e-8)
        
        # 映射到[0, 1]范围
        rgb = (normalized + 1.0) * 0.5
        return np.clip(rgb, 0.0, 1.0)
    
    @staticmethod
    def downsample_points(positions, normals, target_points=50000):
        """
        降采样点云以提升渲染性能（LOD）
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组
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
        
        radius = bbox_size / (num_points ** 0.33) * 0.20  # 进一步增加系数以放大粒子
        radius = max(0.005, min(radius, 0.025))  # 进一步增加最小和最大半径限制
        
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
        
        # 设置渲染引擎
        scene = bpy.context.scene
        scene.render.engine = self.engine.upper()
        
        # 调整曝光以提升整体亮度（Cycles和Eevee都支持）
        if hasattr(scene.view_settings, 'exposure'):
            scene.view_settings.exposure = 0.5  # 增加曝光值
        if hasattr(scene.view_settings, 'gamma'):
            scene.view_settings.gamma = 1.2  # 稍微提升gamma
        
        # 设置渲染分辨率
        scene.render.resolution_x = self.image_width
        scene.render.resolution_y = self.image_height
        scene.render.resolution_percentage = 100
        
        # 设置渲染采样数（仅Cycles）
        if self.engine == 'cycles':
            # 使用传入的采样数，但设置合理的最小值
            scene.cycles.samples = max(self.samples, 32)  # 降低最小采样数以提升速度
            
            # 使用自适应采样（平衡质量和速度）
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.05  # 提高阈值以加快渲染
            
            # 启用降噪以减少噪点（允许较低采样数）
            scene.cycles.use_denoising = True
            # 使用OpenImageDenoise（如果可用）
            if hasattr(scene.cycles, 'denoiser'):
                scene.cycles.denoiser = 'OPENIMAGEDENOISE'
            
            # 简化光线追踪：对于点云，光线弹射1次足够
            scene.cycles.max_bounces = 1
            scene.cycles.diffuse_bounces = 0
            scene.cycles.glossy_bounces = 0
            scene.cycles.transparent_max_bounces = 1
            scene.cycles.transmission_bounces = 0
            
            # 关闭Caustics（焦散）- 点云不需要，可以提升性能
            scene.cycles.caustics_reflective = False
            scene.cycles.caustics_refractive = False
            
            # 启用快速GI近似
            scene.cycles.use_fast_gi = True
            scene.cycles.fast_gi_method = 'REPLACE'
            
            # 使用Persistent Data（如果显存足够，保持静态BVH数据）
            scene.cycles.use_persistent_data = True
            
            # 优化tile大小（GPU建议256，CPU建议512）
            # 根据设备类型自动调整
            try:
                prefs = bpy.context.preferences
                cycles_prefs = prefs.addons['cycles'].preferences
                devices = cycles_prefs.get_devices()
                has_gpu = any(device.type == 'CUDA' or device.type == 'OPENCL' or device.type == 'OPTIX' 
                             for device in devices if device.use)
                scene.cycles.tile_size = 256 if has_gpu else 512  # GPU用256，CPU用512
            except:
                scene.cycles.tile_size = 256
            
            # 简化视口预览
            scene.cycles.preview_samples = 16
            
            # 优化采样模式（使用更快的采样器）
            if hasattr(scene.cycles, 'sampling_pattern'):
                try:
                    # 尝试使用TABULATED_SOBOL（通常更快）
                    scene.cycles.sampling_pattern = 'TABULATED_SOBOL'
                except (ValueError, TypeError):
                    # 如果不可用，使用默认值
                    pass
            
            # 使用GPU加速（如果可用）
            try:
                prefs = bpy.context.preferences
                cycles_prefs = prefs.addons['cycles'].preferences
                devices = cycles_prefs.get_devices()
                has_gpu = any(device.type == 'CUDA' or device.type == 'OPENCL' or device.type == 'OPTIX' 
                             for device in devices if device.use)
                scene.cycles.device = 'GPU' if has_gpu else 'CPU'
            except:
                scene.cycles.device = 'CPU'
        
        # 设置输出格式
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
        
        # 禁用不必要的功能以提升性能
        scene.render.use_motion_blur = False  # 关闭运动模糊
        
        # 设置世界背景
        world = scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg is None:
            bg = world.node_tree.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # 黑色背景
        bg.inputs["Strength"].default_value = 0.0
        
        # 添加环境光（Cycles）
        if self.engine == 'cycles':
            bg.inputs["Strength"].default_value = 1.5  # 增加环境光强度以提升整体亮度
        
        return scene
    
    def setup_camera(self, center, bbox_size):
        """
        设置相机位置和朝向
        
        Args:
            center: 点云中心点 (3,)
            bbox_size: 边界框大小
        """
        # 计算相机位置：在点云上方和侧面
        camera_distance = bbox_size * 3.5
        camera_origin = center + np.array([
            camera_distance * 0.7, 
            camera_distance * 0.7, 
            camera_distance * 0.5
        ])
        
        # 转换为Blender Vector
        center_vec = Vector(center)
        camera_origin_vec = Vector(camera_origin)
        
        # 创建或获取相机
        if 'Camera' not in bpy.data.objects:
            bpy.ops.object.camera_add()
        camera = bpy.data.objects['Camera']
        
        # 设置相机位置
        camera.location = camera_origin_vec
        
        # 设置相机朝向
        direction = center_vec - camera_origin_vec
        if direction.length > 0:
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
        else:
            camera.rotation_euler = (np.radians(90), 0, 0)
        
        # 设置相机参数
        camera.data.type = 'PERSP'
        camera.data.angle = np.radians(30)
        
        # 设置活动相机
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
        # 创建白色背景板（地板）
        floor_z = float(bbox_min[2] - bbox_size * 0.1)
        bpy.ops.mesh.primitive_plane_add(
            size=bbox_size * 12,  # 增大背景板尺寸
            location=(float(center[0]), float(center[1]), floor_z)
        )
        floor = bpy.context.active_object
        floor.name = "BackgroundFloor"
        
        # 创建白色背景板材质
        floor_mat = bpy.data.materials.new(name="BackgroundFloorMaterial")
        floor_mat.use_nodes = True
        bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # 纯白色
            bsdf.inputs["Roughness"].default_value = 0.9
            bsdf.inputs["Metallic"].default_value = 0.0
        floor.data.materials.append(floor_mat)
        
        # 添加顶部光源（环境光已在init_blender_scene中设置）
        bpy.ops.object.light_add(type='AREA', location=(
            float(center[0]), 
            float(center[1]), 
            float(center[2] + bbox_size * 2)
        ))
        top_light = bpy.context.active_object
        top_light.name = "TopLight"
        top_light.data.energy = 150.0  # 增加顶部光源亮度
        top_light.data.size = bbox_size * 1.5
        top_light.rotation_euler = (0, 0, 0)
    
    def create_point_cloud_simple(self, positions, normals, radius):
        """
        创建带颜色的点云网格，使用Geometry Nodes实例化
        每个粒子的RGB颜色直接从PLY文件中读取的normal信息转换而来
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组（从PLY文件读取）
            radius: 粒子半径
        """
        num_points = len(positions)
        assert len(normals) == num_points, "Positions and normals must have the same length"
        
        # 创建网格对象
        mesh = bpy.data.meshes.new(name="PointCloud")
        obj = bpy.data.objects.new("PointCloud", mesh)
        bpy.context.collection.objects.link(obj)
        
        # 添加顶点
        mesh.vertices.add(num_points)
        mesh.vertices.foreach_set("co", positions.flatten())
        
        # 将每个粒子的normal转换为RGB颜色
        # 确保每个粒子的颜色直接来自其对应的normal信息
        colors = self.normal_to_rgb(normals)
        
        # 创建颜色属性（Geometry Nodes使用属性而不是顶点颜色）
        # 属性名称"Col"将用于材质中的ShaderNodeAttribute节点
        color_attr_name = "Col"
        if color_attr_name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[color_attr_name])
        
        color_attr = mesh.attributes.new(name=color_attr_name, type='FLOAT_COLOR', domain='POINT')
        
        # 为每个粒子设置颜色：使用foreach_set优化性能（10x-50x提升）
        # 确保颜色值在[0, 1]范围内，并转换为RGBA格式
        colors_clipped = np.clip(colors, 0.0, 1.0)
        # 创建RGBA数组：将RGB转换为(R, G, B, A)格式，A固定为1.0
        rgba_array = np.column_stack([colors_clipped, np.ones(num_points)]).astype(np.float32).flatten()
        # 使用foreach_set一次性设置所有颜色（比for循环快10-50倍）
        color_attr.data.foreach_set('color', rgba_array)
        
        mesh.update()
        
        # 创建材质
        mat = bpy.data.materials.new(name="PointCloudMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        
        # 创建Attribute节点读取颜色属性
        # 关键：由于使用了Realize Instances，实例已转换为真实几何体
        # 因此需要明确设置为GEOMETRY模式，从几何体点域读取属性
        try:
            color_attr_node = mat.node_tree.nodes.new("ShaderNodeAttribute")
            color_attr_node.attribute_name = "Col"
            # 强制设置为 GEOMETRY，因为我们在节点里已经 Realize 了
            if hasattr(color_attr_node, 'attribute_type'):
                color_attr_node.attribute_type = 'GEOMETRY'
            color_output = color_attr_node.outputs["Color"]
            print(f'    Created Attribute node to read "Col" attribute (GEOMETRY mode)')
        except Exception as e:
            print(f'    Warning: Could not create Attribute node: {e}')
            color_attr_node = mat.node_tree.nodes.new("ShaderNodeRGB")
            color_attr_node.outputs[0].default_value = (1.0, 0.0, 0.0, 1.0)
            color_output = color_attr_node.outputs["Color"]
        
        # 连接颜色到BSDF
        if bsdf and "Base Color" in bsdf.inputs:
            mat.node_tree.links.new(color_output, bsdf.inputs["Base Color"])
        
        # 设置自发光（适中的自发光强度）
        if bsdf:
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 0.3  # 稍微降低自发光强度
            if "Emission Color" in bsdf.inputs:
                mat.node_tree.links.new(color_output, bsdf.inputs["Emission Color"])
            elif "Emission" in bsdf.inputs:
                mat.node_tree.links.new(color_output, bsdf.inputs["Emission"])
        
        obj.data.materials.append(mat)
        
        # 创建Geometry Nodes修改器
        # 关键：需要显式传递颜色属性到实例
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
                sphere_node.inputs['Subdivisions'].default_value = 1  # 降低细分以提升渲染速度
                
                # 连接：输入几何体 -> InstanceOnPoints
                node_group.links.new(input_node.outputs['Geometry'], instance_node.inputs['Points'])
                node_group.links.new(sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
                
                # --- [简化流程开始] ---
                
                # 1. Realize Instances (实现实例)
                # 这一步会自动把点上的 "Col" 属性传递给生成的球体顶点
                realize_node = node_group.nodes.new('GeometryNodeRealizeInstances')
                node_group.links.new(instance_node.outputs['Instances'], realize_node.inputs['Geometry'])
                
                # 2. Set Material (设置材质) - 这一步至关重要！
                # Realize Instances 后材质经常丢失，必须在这里显式指定
                set_mat_node = node_group.nodes.new('GeometryNodeSetMaterial')
                set_mat_node.inputs['Material'].default_value = mat
                node_group.links.new(realize_node.outputs['Geometry'], set_mat_node.inputs['Geometry'])
                
                # 3. Output (输出)
                node_group.links.new(set_mat_node.outputs['Geometry'], output_node.inputs['Geometry'])
                
                # --- [简化流程结束] ---
                
                mod = obj.modifiers.new(name="PointInstances", type='NODES')
                mod.node_group = node_group
                print(f'    Geometry Nodes modifier created (Simplified: Instance -> Realize -> SetMaterial)')
            except Exception as e:
                print(f"Warning: Could not create Geometry Nodes modifier: {e}")
                import traceback
                traceback.print_exc()
        
        return obj
    
    def build_scene(self, positions, normals, radius, bbox_min=None, bbox_max=None, 
                    center=None, bbox_size=None, add_environment=True):
        """
        构建Blender场景
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组（从PLY文件读取，将转换为RGB颜色）
            radius: 粒子半径
            bbox_min: 边界框最小值（可选）
            bbox_max: 边界框最大值（可选）
            center: 点云中心（可选）
            bbox_size: 边界框大小（可选）
            add_environment: 是否添加地板和背景光源
        """
        # 计算点云边界框
        if bbox_min is None or bbox_max is None:
            bbox_min = np.min(positions, axis=0)
            bbox_max = np.max(positions, axis=0)
        if center is None:
            center = (bbox_min + bbox_max) * 0.5
        if bbox_size is None:
            bbox_size = np.max(bbox_max - bbox_min)
        
        # 初始化场景
        scene = self.init_blender_scene()
        
        # 设置相机
        self.setup_camera(center, bbox_size)
        
        # 创建点云（每个粒子的颜色将从其normal信息转换而来）
        print(f'    Creating point cloud with {len(positions):,} points...', end=' ', flush=True)
        self.create_point_cloud_simple(positions, normals, radius)
        print('Done')
        
        # 添加环境（地板和背景光源）
        if add_environment:
            self.add_environment(bbox_min, bbox_max, center, bbox_size)
        
        return scene
    
    def render(self, positions, normals, batch_indices=None, use_batch_rendering=False,
               single_batch_id=None, max_batches=None):
        """
        渲染点云
        每个粒子的RGB颜色将从PLY文件中读取的normal信息转换而来
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组（从PLY文件读取，将转换为RGB颜色）
            batch_indices: (N,) 批次索引数组（可选）
            use_batch_rendering: 是否使用分批渲染（基于batch_idx）
            single_batch_id: 如果指定，只渲染这个batch ID（用于测试）
            max_batches: 如果指定，最多渲染这么多batch（用于测试）
            
        Returns:
            image_path: 渲染图像的路径
        """
        # 降采样点云（如果设置了max_points）
        if self.max_points is not None and len(positions) > self.max_points:
            print(f'  Downsampling from {len(positions):,} to {self.max_points:,} points...', end=' ', flush=True)
            positions, normals = self.downsample_points(positions, normals, self.max_points)
            print('Done')
        
        # 标准化点云位置
        print('  Standardizing point cloud...', end=' ', flush=True)
        positions_std, center, scale = self.standardize_point_cloud(positions)
        print('Done')
        
        # 注意：normals不需要标准化，因为它们只是方向信息
        # 颜色转换将在create_point_cloud_simple中完成，确保每个粒子的颜色
        # 直接来自其对应的normal信息
        
        # 计算粒子半径
        print('  Computing particle radius...', end=' ', flush=True)
        if self.particle_radius is None:
            radius = self.compute_adaptive_radius(positions_std)
        else:
            radius = self.particle_radius
        print(f'Done (radius={radius:.6f})')
        
        # 计算全局边界框（用于保持场景一致性）
        bbox_min = np.min(positions_std, axis=0)
        bbox_max = np.max(positions_std, axis=0)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bbox_size = np.max(bbox_max - bbox_min)
        
        # 如果使用分批渲染且batch_indices可用
        if use_batch_rendering and batch_indices is not None:
            return self._render_batched_by_batch_idx(
                positions_std, normals, radius, batch_indices,
                bbox_min, bbox_max, bbox_center, bbox_size,
                single_batch_id=single_batch_id,
                max_batches=max_batches
            )
        
        # 一次性渲染所有粒子
        # normals将直接传递给build_scene，在create_point_cloud_simple中转换为颜色
        print('  Building Blender scene...', flush=True)
        self.build_scene(positions_std, normals, radius, 
                        bbox_min, bbox_max, bbox_center, bbox_size,
                        add_environment=True)
        
        # 渲染场景
        print('  Rendering scene...', end=' ', flush=True)
        
        # 设置输出路径
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, self.filename)
        else:
            output_file_path = os.path.join(self.folder, self.filename)
        
        # 设置渲染输出路径
        scene = bpy.context.scene
        scene.render.filepath = output_file_path
        
        # 执行渲染
        bpy.ops.render.render(write_still=True)
        
        print(f'Done -> {os.path.basename(output_file_path)}.png')
        
        return output_file_path
    
    def _render_batched_by_batch_idx(self, positions, normals, radius, batch_indices,
                                     bbox_min, bbox_max, center, bbox_size,
                                     single_batch_id=None, max_batches=None):
        """
        按batch_idx收集粒子并一次性渲染
        每个粒子的RGB颜色将从其对应的normal信息转换而来
        
        Args:
            positions: (N, 3) 标准化后的位置数组
            normals: (N, 3) 法向量数组（从PLY文件读取，将转换为RGB颜色）
            radius: 粒子半径
            batch_indices: (N,) 批次索引数组
            bbox_min: 全局边界框最小值
            bbox_max: 全局边界框最大值
            center: 全局中心点
            bbox_size: 全局边界框大小
            single_batch_id: 如果指定，只渲染这个batch ID（用于测试）
            max_batches: 如果指定，最多渲染这么多batch（用于测试）
            
        Returns:
            image_path: 渲染图像的路径
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
        
        # 收集所有指定batch的粒子（保持positions和normals的对应关系）
        batch_mask = np.isin(batch_indices, selected_batches)
        selected_positions = positions[batch_mask]
        selected_normals = normals[batch_mask]  # 保持与positions的对应关系
        num_particles = len(selected_positions)
        
        print(f'  Total particles to render: {num_particles:,}')
        
        # 一次性构建场景并渲染
        # normals将直接传递给build_scene，在create_point_cloud_simple中转换为颜色
        print('  Building Blender scene...', flush=True)
        self.build_scene(selected_positions, selected_normals, radius,
                        bbox_min, bbox_max, center, bbox_size,
                        add_environment=True)
        
        # 渲染场景
        print('  Rendering scene...', end=' ', flush=True)
        
        # 设置输出路径
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, self.filename)
        else:
            output_file_path = os.path.join(self.folder, self.filename)
        
        # 设置渲染输出路径
        scene = bpy.context.scene
        scene.render.filepath = output_file_path
        
        # 执行渲染
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
        # 加载点云
        positions, normals, batch_indices = self.load_point_cloud()
        
        # 渲染
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
                 engine='cycles',
                 blender_path=None,
                 use_batch_rendering=False,
                 single_batch_id=None,
                 max_batches=None,
                 max_points=None):
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
        engine: 渲染引擎 ('cycles' 或 'eevee')
        blender_path: Blender可执行文件路径（如果通过命令行调用）
        use_batch_rendering: 是否使用基于batch_idx的分批渲染
        single_batch_id: 如果指定，只渲染这个batch ID（用于测试，需要use_batch_rendering=True）
        max_batches: 如果指定，最多渲染这么多batch（用于测试，需要use_batch_rendering=True）
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
    print(f'Render engine: {engine}')
    if use_batch_rendering:
        if single_batch_id is not None:
            print(f'Batch rendering: Single batch ID {single_batch_id}')
        elif max_batches is not None:
            print(f'Batch rendering: Max {max_batches} batches')
        else:
            print('Batch rendering: All batches')
    print('=' * 60)
    
    os.makedirs(output_folder, exist_ok=True)
    
    successful = 0
    failed = 0
    
    # 批量渲染
    for idx, ply_file in enumerate(ply_files, 1):
        basename = os.path.basename(ply_file)
        
        print(f'\n[{idx}/{total_files}] ({idx*100//total_files}%) Processing: {basename}')
        print('-' * 60)
        
        try:
            renderer = BlenderNormalRenderer(
                ply_file,
                output_folder=output_folder,
                image_width=image_width,
                image_height=image_height,
                samples=samples,
                engine=engine,
                max_points=max_points
            )
            
            # 加载点云
            print('  Loading point cloud...', end=' ', flush=True)
            positions, normals, batch_indices = renderer.load_point_cloud()
            print(f'Done ({len(positions):,} points)')
            
            # 处理和渲染
            image_path = renderer.render(positions, normals, batch_indices, 
                                        use_batch_rendering, single_batch_id, max_batches)
            
            # 清理场景（为下一个文件准备）
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            
            # 清理未使用的数据块
            for block in bpy.data.meshes:
                if block.users == 0:
                    bpy.data.meshes.remove(block)
            for block in bpy.data.materials:
                if block.users == 0:
                    bpy.data.materials.remove(block)
            
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
    
    parser = argparse.ArgumentParser(description='Render point clouds with normal-based coloring using Blender')
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
                        help='Samples per pixel (Cycles engine, default: 64)')
    parser.add_argument('--engine', type=str, default='cycles',
                        choices=['cycles', 'eevee'],
                        help='Render engine: cycles or eevee (default: cycles)')
    parser.add_argument('--use-batch-rendering', action='store_true',
                        help='Use batch_idx-based batch rendering (256 batches, 2048 particles each)')
    parser.add_argument('--single-batch', type=int, default=None,
                        help='Render only a single batch ID (0-255) for testing (requires --use-batch-rendering)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to render (requires --use-batch-rendering)')
    parser.add_argument('--max-points', type=int, default=None,
                        help='Maximum number of points to render (LOD downsampling, None=no limit)')
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
        engine=args.engine,
        use_batch_rendering=args.use_batch_rendering,
        single_batch_id=args.single_batch,
        max_batches=args.max_batches,
        max_points=args.max_points
    )

