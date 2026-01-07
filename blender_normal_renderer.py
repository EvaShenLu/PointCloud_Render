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
                 samples=64, engine='cycles'):
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
        
        # 设置渲染分辨率
        scene.render.resolution_x = self.image_width
        scene.render.resolution_y = self.image_height
        scene.render.resolution_percentage = 100
        
        # 设置渲染采样数（仅Cycles）
        if self.engine == 'cycles':
            scene.cycles.samples = self.samples
            scene.cycles.use_denoising = True  # 启用降噪以提升质量
            # 使用GPU加速（如果可用）
            try:
                # 检查是否有GPU设备可用
                prefs = bpy.context.preferences
                cycles_prefs = prefs.addons['cycles'].preferences
                devices = cycles_prefs.get_devices()
                has_gpu = any(device.type == 'CUDA' or device.type == 'OPENCL' or device.type == 'OPTIX' 
                             for device in devices if device.use)
                scene.cycles.device = 'GPU' if has_gpu else 'CPU'
            except:
                # 如果无法检测GPU，默认使用CPU
                scene.cycles.device = 'CPU'
        
        # 设置输出格式
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '16'
        
        # 设置世界背景为黑色
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
        
        # 设置相机朝向（看向点云中心）
        direction = center_vec - camera_origin_vec
        if direction.length > 0:
            # 使用look_at方法设置相机朝向
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
        else:
            # 如果方向为零，使用默认朝向
            camera.rotation_euler = (np.radians(90), 0, 0)
        
        # 设置相机参数
        camera.data.type = 'PERSP'
        camera.data.angle = np.radians(30)  # FOV 30度
        
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
        # 创建地板
        floor_z = float(bbox_min[2] - bbox_size * 0.1)
        bpy.ops.mesh.primitive_plane_add(
            size=bbox_size * 5,
            location=(float(center[0]), float(center[1]), floor_z)
        )
        floor = bpy.context.active_object
        floor.name = "Floor"
        
        # 创建地板材质
        floor_mat = bpy.data.materials.new(name="FloorMaterial")
        floor_mat.use_nodes = True
        bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.9
        floor.data.materials.append(floor_mat)
        
        # 创建背景光源（矩形面光源）
        bg_z = float(center[2] + bbox_size * 2)
        bpy.ops.mesh.primitive_plane_add(
            size=bbox_size * 3,
            location=(float(center[0]), float(center[1]), bg_z)
        )
        bg_light = bpy.context.active_object
        bg_light.name = "BackgroundLight"
        
        # 旋转光源使其面向点云
        bg_light.rotation_euler = (np.radians(90), 0, 0)
        
        # 添加发光材质
        light_mat = bpy.data.materials.new(name="BackgroundLightMaterial")
        light_mat.use_nodes = True
        emission = light_mat.node_tree.nodes.get("Emission")
        if emission is None:
            emission = light_mat.node_tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 1.8
        
        # 连接到输出
        output = light_mat.node_tree.nodes.get("Material Output")
        if output:
            light_mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        
        bg_light.data.materials.append(light_mat)
        
        # 设置为发光对象（Eevee需要）
        if self.engine == 'eevee':
            bg_light.data.use_auto_smooth = False
    
    def create_point_cloud_mesh(self, positions, colors, radius):
        """
        使用Geometry Nodes创建点云（高效实例化方法）
        
        Args:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组，范围[0, 1]
            radius: 粒子半径
        """
        num_points = len(positions)
        
        # 创建基础球体作为实例模板
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius)
        sphere_template = bpy.context.active_object
        sphere_template.name = "ParticleTemplate"
        
        # 创建点云对象（空网格）
        bpy.ops.object.empty_add(type='PLAIN_AXES')
        point_cloud = bpy.context.active_object
        point_cloud.name = "PointCloud"
        
        # 删除空对象，创建新的网格对象
        bpy.ops.object.delete()
        bpy.ops.mesh.primitive_cube_add()
        point_cloud = bpy.context.active_object
        point_cloud.name = "PointCloud"
        
        # 清除默认立方体，创建点云网格
        mesh = point_cloud.data
        mesh.clear_geometry()
        
        # 添加顶点（点云位置）
        mesh.vertices.add(num_points)
        mesh.vertices.foreach_set("co", positions.flatten())
        
        # 更新网格
        mesh.update()
        
        # 使用Geometry Nodes进行实例化
        # 为点云对象添加Geometry Nodes修改器
        if bpy.app.version >= (3, 0, 0):  # Blender 3.0+ 支持Geometry Nodes
            # 创建Geometry Nodes修改器
            mod = point_cloud.modifiers.new(name="PointCloudInstances", type='NODES')
            
            # 创建节点组
            node_group = bpy.data.node_groups.new(name="PointCloudInstances", type='GeometryNodeTree')
            mod.node_group = node_group
            
            # 添加输入输出节点
            input_node = node_group.nodes.new('NodeGroupInput')
            output_node = node_group.nodes.new('NodeGroupOutput')
            
            # 添加实例化节点
            instance_node = node_group.nodes.new('GeometryNodeInstanceOnPoints')
            
            # 添加球体节点
            sphere_node = node_group.nodes.new('GeometryNodeMeshIcoSphere')
            sphere_node.inputs['Radius'].default_value = radius
            sphere_node.inputs['Subdivisions'].default_value = 2
            
            # 连接节点
            node_group.links.new(input_node.outputs['Geometry'], instance_node.inputs['Points'])
            node_group.links.new(sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
            node_group.links.new(instance_node.outputs['Instances'], output_node.inputs['Geometry'])
        else:
            # Blender 2.93及以下版本：使用粒子系统
            # 删除当前对象，改用粒子系统方法
            bpy.ops.object.delete()
            self._create_point_cloud_with_particles(positions, colors, radius)
            return
        
        # 创建材质并应用颜色
        self._apply_colors_to_mesh(point_cloud, colors, num_points)
        
        # 隐藏模板球体
        sphere_template.hide_viewport = True
        sphere_template.hide_render = True
    
    def _create_point_cloud_with_particles(self, positions, colors, radius):
        """
        使用粒子系统创建点云（Blender 2.93及以下版本的备选方法）
        """
        # 创建基础球体
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius)
        sphere = bpy.context.active_object
        sphere.name = "PointCloud"
        
        # 创建材质
        mat = bpy.data.materials.new(name="PointCloudMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # 使用顶点颜色（如果支持）
        # 这里简化处理：使用平均颜色
        avg_color = np.mean(colors, axis=0)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*avg_color, 1.0)
        
        sphere.data.materials.append(mat)
        
        # 使用数组修改器复制球体（性能较差，但兼容旧版本）
        # 注意：对于大量点云，这种方法会很慢
        print(f"Warning: Using fallback method for Blender < 3.0. Performance may be slow for large point clouds.")
    
    def _apply_colors_to_mesh(self, obj, colors, num_points):
        """
        为网格应用颜色（使用顶点颜色或材质）
        
        Args:
            obj: Blender对象
            colors: (N, 3) RGB颜色数组
            num_points: 点的数量
        """
        # 创建材质
        mat = bpy.data.materials.new(name="PointCloudMaterial")
        mat.use_nodes = True
        
        # 获取或创建Principled BSDF节点
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        
        # 尝试使用顶点颜色
        # 为网格添加顶点颜色层
        mesh = obj.data
        if not mesh.vertex_colors:
            color_layer = mesh.vertex_colors.new(name="Col")
        else:
            color_layer = mesh.vertex_colors[0]
        
        # 设置顶点颜色
        for i, color in enumerate(colors):
            if i < len(color_layer.data):
                color_layer.data[i].color = (*color, 1.0)
        
        # 在材质中使用顶点颜色
        vertex_color_node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        vertex_color_node.layer_name = "Col"
        mat.node_tree.links.new(vertex_color_node.outputs["Color"], bsdf.inputs["Base Color"])
        
        # 应用材质
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
    
    def create_point_cloud_simple(self, positions, colors, radius):
        """
        简化方法：直接创建带颜色的点云网格（不使用Geometry Nodes）
        对于大量点云，这种方法可能更稳定
        
        Args:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组，范围[0, 1]
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
        
        # 添加顶点颜色
        color_layer = mesh.vertex_colors.new(name="Col")
        for i, color in enumerate(colors):
            if i < len(color_layer.data):
                color_layer.data[i].color = (*color, 1.0)
        
        # 更新网格
        mesh.update()
        
        # 创建材质
        mat = bpy.data.materials.new(name="PointCloudMaterial")
        mat.use_nodes = True
        
        # 使用顶点颜色
        vertex_color_node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        vertex_color_node.layer_name = "Col"
        
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            mat.node_tree.links.new(vertex_color_node.outputs["Color"], bsdf.inputs["Base Color"])
            # 设置材质为自发光（使点更明显）
            bsdf.inputs["Emission Strength"].default_value = 0.5
            bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        
        obj.data.materials.append(mat)
        
        # 使用Geometry Nodes将点转换为球体实例
        # 这需要Blender 3.0+
        if bpy.app.version >= (3, 0, 0):
            try:
                # 检查是否已有同名的节点组
                if "PointInstances" in bpy.data.node_groups:
                    node_group = bpy.data.node_groups["PointInstances"]
                else:
                    node_group = bpy.data.node_groups.new(name="PointInstances", type='GeometryNodeTree')
                
                # 清空现有节点（如果存在）
                node_group.nodes.clear()
                
                # 创建输入输出节点
                input_node = node_group.nodes.new('NodeGroupInput')
                output_node = node_group.nodes.new('NodeGroupOutput')
                
                # 添加输入插槽
                if 'Geometry' not in [socket.name for socket in input_node.outputs]:
                    input_node.outputs.new('NodeSocketGeometry', 'Geometry')
                if 'Geometry' not in [socket.name for socket in output_node.inputs]:
                    output_node.inputs.new('NodeSocketGeometry', 'Geometry')
                
                # 创建实例化节点
                instance_node = node_group.nodes.new('GeometryNodeInstanceOnPoints')
                sphere_node = node_group.nodes.new('GeometryNodeMeshIcoSphere')
                
                sphere_node.inputs['Radius'].default_value = radius
                sphere_node.inputs['Subdivisions'].default_value = 2
                
                # 连接节点
                node_group.links.new(input_node.outputs['Geometry'], instance_node.inputs['Points'])
                node_group.links.new(sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
                node_group.links.new(instance_node.outputs['Instances'], output_node.inputs['Geometry'])
                
                # 添加修改器
                mod = obj.modifiers.new(name="PointInstances", type='NODES')
                mod.node_group = node_group
                
            except Exception as e:
                print(f"Warning: Could not create Geometry Nodes modifier: {e}")
                print("Falling back to simple point rendering (points may not be visible as spheres)")
                import traceback
                traceback.print_exc()
        
        return obj
    
    def build_scene(self, positions, colors, radius, bbox_min=None, bbox_max=None, 
                    center=None, bbox_size=None, add_environment=True):
        """
        构建Blender场景
        
        Args:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组
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
        
        # 创建点云
        print(f'    Creating point cloud with {len(positions):,} points...', end=' ', flush=True)
        self.create_point_cloud_simple(positions, colors, radius)
        print('Done')
        
        # 添加环境（地板和背景光源）
        if add_environment:
            self.add_environment(bbox_min, bbox_max, center, bbox_size)
        
        return scene
    
    def render(self, positions, normals, batch_indices=None, use_batch_rendering=False):
        """
        渲染点云
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组
            batch_indices: (N,) 批次索引数组（可选，当前版本未使用）
            use_batch_rendering: 是否使用分批渲染（当前版本未实现）
            
        Returns:
            image_path: 渲染图像的路径
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
        print(f'Done (radius={radius:.6f})')
        
        # 构建场景
        print('  Building Blender scene...', flush=True)
        self.build_scene(positions_std, colors, radius, add_environment=True)
        
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
    
    def process(self):
        """
        处理单帧点云
        """
        # 加载点云
        positions, normals, batch_indices = self.load_point_cloud()
        
        # 渲染
        image_path = self.render(positions, normals, batch_indices)
        
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
                 blender_path=None):
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
                engine=engine
            )
            
            # 加载点云
            print('  Loading point cloud...', end=' ', flush=True)
            positions, normals, batch_indices = renderer.load_point_cloud()
            print(f'Done ({len(positions):,} points)')
            
            # 处理和渲染
            image_path = renderer.render(positions, normals, batch_indices)
            
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
    args = parser.parse_args()
    
    batch_render(
        input_folder=args.input,
        output_folder=args.output,
        pattern=args.pattern,
        start_frame=args.start,
        end_frame=args.end,
        image_width=args.width,
        image_height=args.height,
        samples=args.samples,
        engine=args.engine
    )

