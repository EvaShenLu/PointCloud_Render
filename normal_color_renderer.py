import numpy as np
import os
import sys
from plyfile import PlyData
import mitsuba as mi


class NormalColorRenderer:
    """
    高效的点云渲染器，使用Mitsuba Python API和instancing优化
    将法向量映射为RGB颜色
    """
    
    def __init__(self, file_path, output_folder=None, 
                 particle_radius=None, 
                 image_width=1920, image_height=1080,
                 samples=256):
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
        """初始化Mitsuba variant，优先使用GPU"""
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
    
    def load_point_cloud(self):
        """加载PLY文件，返回位置、法向量和批次索引"""
        file_extension = os.path.splitext(self.file_path)[1]
        
        if file_extension == '.ply':
            ply_data = PlyData.read(self.file_path)
            vertex_data = ply_data['vertex']
            
            # 读取位置
            positions = np.column_stack([
                vertex_data['x'],
                vertex_data['y'],
                vertex_data['z']
            ]).astype(np.float32)
            
            # 读取法向量 - 使用try-except检查字段是否存在
            try:
                normals = np.column_stack([
                    vertex_data['nx'],
                    vertex_data['ny'],
                    vertex_data['nz']
                ]).astype(np.float32)
            except (KeyError, ValueError) as e:
                raise ValueError(f'PLY file does not contain normal vectors (nx, ny, nz): {e}')
            
            # 读取批次索引
            try:
                batch_indices = vertex_data['batch_idx'].astype(np.int32)
            except (KeyError, ValueError):
                # 如果没有batch_idx字段，返回None
                batch_indices = None
            
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
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normalized = normals / (norms + 1e-8)
        
        # 映射到[0, 1]范围: (normal + 1) / 2
        rgb = (normalized + 1.0) / 2.0
        
        # 确保值在[0, 1]范围内
        rgb = np.clip(rgb, 0.0, 1.0)
        
        return rgb
    
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
        
        # 根据点云密度计算半径
        # 使用立方根来估算合适的粒子大小
        point_density = num_points / (bbox_size ** 3 + 1e-8)
        radius = bbox_size / (num_points ** 0.33) * 0.15
        
        # 限制在合理范围内
        radius = max(0.005, min(radius, 0.02))
        
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
        center = np.mean(positions, axis=0)
        positions_centered = positions - center
        scale = np.max(np.abs(positions_centered))
        if scale > 1e-8:
            standardized = positions_centered / scale
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
                'max_depth': -1
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
                    'rfilter': {'type': 'gaussian'}
                },
                'sampler': {
                    'type': 'independent',
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
        # 创建地面
        floor_z = bbox_min[2] - bbox_size * 0.1
        scene_dict['floor'] = {
            'type': 'rectangle',
            'to_world': (mi.ScalarTransform4f.scale([bbox_size * 2, bbox_size * 2, 1.0])
                        @ mi.ScalarTransform4f.translate([0, 0, floor_z])),
            'bsdf': {
                'type': 'roughplastic',
                'distribution': 'ggx',
                'alpha': 0.1,
                'int_ior': 1.46,
                'diffuse_reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
            }
        }
        
        # 创建背景光源
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
                'radiance': {'type': 'rgb', 'value': [4.0, 4.0, 4.0]}
            }
        }
    
    def build_scene(self, positions, colors, radius, bbox_min=None, bbox_max=None, center=None, bbox_size=None):
        """
        使用Mitsuba Python API构建场景，优化大量粒子的渲染
        
        Args:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组
            radius: 粒子半径
            bbox_min: 边界框最小值（可选，用于分批渲染时保持场景一致）
            bbox_max: 边界框最大值（可选，用于分批渲染时保持场景一致）
            center: 点云中心（可选，用于分批渲染时保持场景一致）
            bbox_size: 边界框大小（可选，用于分批渲染时保持场景一致）
            
        Returns:
            scene: Mitsuba场景对象
        """
        # 计算点云边界框以设置合适的相机位置
        if bbox_min is None:
            bbox_min = np.min(positions, axis=0)
        if bbox_max is None:
            bbox_max = np.max(positions, axis=0)
        if center is None:
            center = (bbox_min + bbox_max) / 2.0
        if bbox_size is None:
            bbox_size = np.max(bbox_max - bbox_min)
        
        # 相机位置：在点云上方和侧面
        camera_distance = bbox_size * 2.5
        camera_origin = center + np.array([camera_distance * 0.7, 
                                           camera_distance * 0.7, 
                                           camera_distance * 0.5])
        
        # 创建基础场景配置
        scene_dict = self._create_base_scene_dict(center, bbox_size, camera_origin)
        
        # 使用merge shape优化：将所有粒子合并到一个merge shape中
        # 根据Mitsuba3 issue #1017，对于简单几何体，merge比独立shape快>100倍
        # 
        # Merge shape的正确用法（参考GitHub issue #1017）：
        # - 创建一个字典，type设为'merge'
        # - 将子shape作为键值对添加到merge字典中（键名可以是任意字符串）
        # - 每个子shape必须有自己的type、transform和bsdf
        # - 将merge shape添加到场景字典中作为顶级shape
        num_points = len(positions)
        
        # 创建merge shape来合并所有粒子
        # 格式：{'type': 'merge', 'particle_0': {...}, 'particle_1': {...}, ...}
        merge_shape = {'type': 'merge'}
        
        # 创建粒子shapes
        print(f'    Creating {num_points:,} particle shapes...', end=' ', flush=True)
        for idx, (pos, color) in enumerate(zip(positions, colors)):
            merge_shape[f'particle_{idx}'] = {
                'type': 'sphere',
                'radius': radius,
                'to_world': mi.ScalarTransform4f.translate(pos.tolist()),
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': color.tolist()
                    }
                }
            }
            # 每10%显示一次进度
            if (idx + 1) % (num_points // 10 + 1) == 0 or (idx + 1) == num_points:
                progress = int((idx + 1) * 100 / num_points)
                print(f'{progress}%', end=' ', flush=True)
        print('Done')
        
        # 将merge shape添加到场景字典
        scene_dict['particles'] = merge_shape
        
        # 添加地板和背景光源
        self._add_environment(scene_dict, bbox_min, bbox_max, center, bbox_size)
        
        # 加载场景
        print('    Loading scene into Mitsuba...', end=' ', flush=True)
        scene = mi.load_dict(scene_dict)
        print('Done')
        return scene
    
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
        
        # 计算全局边界框（用于保持场景一致性）
        bbox_min = np.min(positions_std, axis=0)
        bbox_max = np.max(positions_std, axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_size = np.max(bbox_max - bbox_min)
        
        # 如果使用分批渲染且batch_indices可用
        if use_batch_rendering and batch_indices is not None:
            # 检查是否有single_batch参数（通过实例变量传递）
            single_batch_id = getattr(self, '_single_batch_id', None)
            return self._render_batched_by_batch_idx(
                positions_std, colors, radius, batch_indices,
                bbox_min, bbox_max, bbox_center, bbox_size,
                single_batch_id=single_batch_id
            )
        
        # 一次性渲染所有粒子
        # 注意：使用merge shape优化后，一次性渲染已经非常高效（比独立shape快>100倍）
        scene = self.build_scene(positions_std, colors, radius, 
                                bbox_min, bbox_max, bbox_center, bbox_size)
        
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
                                     single_batch_id=None):
        """
        按batch_idx分批渲染点云
        
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
            
        Returns:
            image: 合并后的渲染图像
        """
        # 获取唯一的批次ID并排序
        unique_batches = np.unique(batch_indices)
        unique_batches = np.sort(unique_batches)
        
        # 如果指定了single_batch_id，只渲染该batch
        if single_batch_id is not None:
            if single_batch_id not in unique_batches:
                raise ValueError(f'Batch ID {single_batch_id} not found. Available batch IDs: {unique_batches[0]} to {unique_batches[-1]}')
            unique_batches = [single_batch_id]
            print(f'  Rendering single batch {single_batch_id} (test mode)...')
        else:
            num_batches = len(unique_batches)
            print(f'  Rendering {num_batches} batches (batch size: {len(positions) // num_batches} particles per batch)...')
        
        # 初始化累积图像（使用alpha混合）
        accumulated_image = None
        
        # 按批次渲染
        for batch_idx, batch_id in enumerate(unique_batches):
            # 获取当前批次的粒子索引
            batch_mask = batch_indices == batch_id
            batch_positions = positions[batch_mask]
            batch_colors = colors[batch_mask]
            num_batch_particles = len(batch_positions)
            
            print(f'    Batch {batch_id} ({batch_idx + 1}/{num_batches}): {num_batch_particles:,} particles...', 
                  end=' ', flush=True)
            
            # 为当前批次构建场景（使用全局边界框保持一致性）
            scene = self.build_scene(batch_positions, batch_colors, radius,
                                    bbox_min, bbox_max, center, bbox_size)
            
            # 渲染当前批次
            batch_image = mi.render(scene)
            
            # 转换为numpy数组以便处理
            batch_image_np = mi.util.convert_to_bitmap(batch_image)
            
            # 合并图像：使用alpha混合或直接叠加
            # 对于点云渲染，我们使用简单的叠加（因为粒子是半透明的）
            if accumulated_image is None:
                accumulated_image = batch_image_np.copy()
            else:
                # 叠加：新图像覆盖旧图像（点云粒子会自然混合）
                # 使用最大值混合，保留最亮的像素
                accumulated_image = np.maximum(accumulated_image, batch_image_np)
            
            # 清理场景和图像
            del scene, batch_image, batch_image_np
            import gc
            gc.collect()
            
            print('Done')
        
        # 转换回Mitsuba图像格式
        # 使用 mi.Bitmap 将numpy数组转换为Mitsuba图像格式
        bitmap = mi.Bitmap(accumulated_image)
        final_image = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
        
        print('  All batches rendered and merged')
        return final_image
    
    
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
                 samples=256,
                 num_workers=1,
                 use_batch_rendering=False,
                 single_batch_id=None):
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
        num_workers: 并行工作进程数（1表示单进程，>1需要多进程支持）
        use_batch_rendering: 是否使用基于batch_idx的分批渲染
        single_batch_id: 如果指定，只渲染这个batch ID（用于测试，需要use_batch_rendering=True）
    """
    import glob
    
    # 初始化Mitsuba
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
    print(f'Workers: {num_workers}')
    print('=' * 60)
    
    os.makedirs(output_folder, exist_ok=True)
    
    successful = 0
    failed = 0
    
    # 如果使用多进程，需要重新初始化Mitsuba（每个进程独立）
    if num_workers > 1:
        try:
            from multiprocessing import Pool
            import functools
            
            def render_single_file(args):
                """单文件渲染函数，用于多进程"""
                ply_file, output_folder, image_width, image_height, samples, file_idx, total_files = args
                
                # 每个进程需要重新初始化Mitsuba
                # 注意：多进程模式下，GPU可能无法共享，会回退到CPU
                try:
                    NormalColorRenderer.init_mitsuba_variant()
                except:
                    pass  # 如果初始化失败，继续尝试
                
                basename = os.path.basename(ply_file)
                try:
                    renderer = NormalColorRenderer(
                        ply_file,
                        output_folder=output_folder,
                        image_width=image_width,
                        image_height=image_height,
                        samples=samples
                    )
                    
                    positions, normals, batch_indices = renderer.load_point_cloud()
                    renderer._single_batch_id = single_batch_id if use_batch_rendering else None
                    image = renderer.render(positions, normals, batch_indices, use_batch_rendering)
                    renderer.save_image(
                        os.path.join(output_folder, renderer.filename),
                        image
                    )
                    
                    # 清理内存
                    del renderer, positions, normals, image
                    import gc
                    gc.collect()
                    
                    return (True, basename, None)
                except Exception as e:
                    import traceback
                    return (False, basename, f'{str(e)}\n{traceback.format_exc()}')
            
            # 准备参数
            render_args = [
                (f, output_folder, image_width, image_height, samples, i+1, total_files)
                for i, f in enumerate(ply_files)
            ]
            
            # 使用进程池
            print(f'\nStarting parallel rendering with {num_workers} workers...')
            print('Note: Multi-process rendering may use CPU instead of GPU')
            with Pool(processes=num_workers) as pool:
                results = pool.map(render_single_file, render_args)
            
            # 统计结果
            for success, basename, error in results:
                if success:
                    successful += 1
                    print(f'✓ {basename}')
                else:
                    failed += 1
                    print(f'✗ {basename}: {error}')
            
        except ImportError:
            print('Warning: multiprocessing not available, falling back to single process')
            num_workers = 1
    
    # 单进程渲染
    if num_workers == 1:
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
    parser.add_argument('--samples', type=int, default=256,
                        help='Samples per pixel')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (multiprocessing)')
    parser.add_argument('--use-batch-rendering', action='store_true',
                        help='Use batch_idx-based batch rendering (256 batches, 2048 particles each)')
    parser.add_argument('--single-batch', type=int, default=None,
                        help='Render only a single batch ID (0-255) for testing (requires --use-batch-rendering)')
    args = parser.parse_args()
    
    # 验证参数
    if args.single_batch is not None and not args.use_batch_rendering:
        print('Warning: --single-batch requires --use-batch-rendering. Ignoring --single-batch.')
        args.single_batch = None
    
    batch_render(
        input_folder=args.input,
        output_folder=args.output,
        pattern=args.pattern,
        start_frame=args.start,
        end_frame=args.end,
        image_width=args.width,
        image_height=args.height,
        samples=args.samples,
        num_workers=args.workers,
        use_batch_rendering=args.use_batch_rendering,
        single_batch_id=args.single_batch
    )

