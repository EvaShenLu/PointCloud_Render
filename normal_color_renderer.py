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
                 samples=256,
                 render_batch_size=20000):
        """
        初始化渲染器
        
        Args:
            file_path: PLY文件路径
            output_folder: 输出文件夹
            particle_radius: 粒子半径（None表示自适应）
            image_width: 图像宽度
            image_height: 图像高度
            samples: 采样数
            render_batch_size: 分批渲染的每批粒子数（默认20000，None表示不分批）
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
        self.render_batch_size = render_batch_size  # None=不分批, >0=每批粒子数（默认20000）
        
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
        """加载PLY文件，返回位置和法向量"""
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
            
            return positions, normals
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
    
    def build_scene(self, positions, colors, radius):
        """
        使用Mitsuba Python API构建场景，优化大量粒子的渲染
        
        Args:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组
            radius: 粒子半径
            
        Returns:
            scene: Mitsuba场景对象
        """
        # 计算点云边界框以设置合适的相机位置
        bbox_min = np.min(positions, axis=0)
        bbox_max = np.max(positions, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        bbox_size = np.max(bbox_max - bbox_min)
        
        # 相机位置：在点云上方和侧面
        camera_distance = bbox_size * 2.5
        camera_origin = center + np.array([camera_distance * 0.7, 
                                           camera_distance * 0.7, 
                                           camera_distance * 0.5])
        
        # 构建场景字典
        scene_dict = {
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
        
        # 使用instancing：创建一个shapegroup包含模板sphere，然后用N个instance复制
        num_points = len(positions)
        
        print(f'    Creating sphere shapegroup and {num_points:,} instances...', flush=True)
        
        # 创建shapegroup（包含模板sphere，共享几何体）
        scene_dict['particle_shapegroup'] = {
            'type': 'shapegroup',
            'id': 'particle_group',
            'shape': {
                'type': 'sphere',
                'radius': radius
            }
        }
        
        # 创建所有粒子instances
        # 分批添加到场景字典，避免一次性创建太多键
        batch_size = 50000  # 每批处理50000个粒子
        total_batches = (num_points + batch_size - 1) // batch_size
        last_progress = -1
        
        for batch_idx, batch_start in enumerate(range(0, num_points, batch_size)):
            batch_end = min(batch_start + batch_size, num_points)
            batch_positions = positions[batch_start:batch_end]
            batch_colors = colors[batch_start:batch_end]
            
            for local_idx, (pos, color) in enumerate(zip(batch_positions, batch_colors)):
                global_idx = batch_start + local_idx
                # 创建instance，引用shapegroup，但使用不同的变换和BSDF
                scene_dict[f'particle_instance_{global_idx}'] = {
                    'type': 'instance',
                    'shapegroup': {
                        'id': 'particle_group'
                    },
                    'to_world': mi.ScalarTransform4f.translate(pos.tolist()),
                    'bsdf': {
                        'type': 'diffuse',
                        'reflectance': {
                            'type': 'rgb',
                            'value': color.tolist()
                        }
                    }
                }
            
            # 显示进度（每批或每10%更新一次）
            current_progress = int((batch_idx + 1) * 100 / total_batches)
            if current_progress != last_progress:
                particles_created = min(batch_end, num_points)
                print(f'      Progress: {particles_created:,}/{num_points:,} instances ({current_progress}%)', 
                      end='\r', flush=True)
                last_progress = current_progress
        
        print(f'      Progress: {num_points:,}/{num_points:,} instances (100%) - Done!', flush=True)
        
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
        
        # 加载场景
        print('    Loading scene into Mitsuba (this may take a while for large scenes)...', end=' ', flush=True)
        scene = mi.load_dict(scene_dict)
        print('Done')
        return scene
    
    def render(self, positions, normals):
        """
        渲染点云（支持分批渲染以显示进度）
        
        Args:
            positions: (N, 3) 位置数组
            normals: (N, 3) 法向量数组
            
        Returns:
            image: 渲染的图像
        """
        print('  Standardizing point cloud...', end=' ', flush=True)
        # 标准化点云
        positions_std, center, scale = self.standardize_point_cloud(positions)
        print('Done')
        
        print('  Converting normals to RGB colors...', end=' ', flush=True)
        # 法向量到RGB映射
        colors = self.normal_to_rgb(normals)
        print('Done')
        
        print('  Computing particle radius...', end=' ', flush=True)
        # 计算粒子半径
        if self.particle_radius is None:
            radius = self.compute_adaptive_radius(positions_std)
            print(f'Done (adaptive: {radius:.6f})')
        else:
            radius = self.particle_radius
            print(f'Done (fixed: {radius:.6f})')
        
        num_points = len(positions_std)
        
        # 如果指定了batch_size且粒子数足够多，使用分批渲染
        if self.render_batch_size is not None and self.render_batch_size > 0 and num_points > self.render_batch_size:
            return self._render_batched(positions_std, colors, radius, self.render_batch_size)
        else:
            # 一次性渲染所有粒子
            scene = self.build_scene(positions_std, colors, radius)
            print('  Rendering scene (this is the most time-consuming step)...', end=' ', flush=True)
            image = mi.render(scene)
            print('Done')
            return image
    
    def _render_batched(self, positions, colors, radius, batch_size):
        """
        分批渲染点云，每批显示进度
        
        Args:
            positions: (N, 3) 位置数组
            colors: (N, 3) RGB颜色数组
            radius: 粒子半径
            batch_size: 每批粒子数
            
        Returns:
            image: 合成的渲染图像
        """
        num_points = len(positions)
        num_batches = (num_points + batch_size - 1) // batch_size
        
        print(f'  Rendering {num_points:,} particles in {num_batches} batches ({batch_size:,} particles/batch)...')
        
        # 计算点云边界框以设置相机（用于所有批次）
        bbox_min = np.min(positions, axis=0)
        bbox_max = np.max(positions, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        bbox_size = np.max(bbox_max - bbox_min)
        camera_distance = bbox_size * 2.5
        camera_origin = center + np.array([camera_distance * 0.7, 
                                           camera_distance * 0.7, 
                                           camera_distance * 0.5])
        
        # 创建基础场景设置（相机、积分器等）
        base_scene_dict = {
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
        
        # 存储每批的渲染结果
        batch_images = []
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_points)
            batch_positions = positions[batch_start:batch_end]
            batch_colors = colors[batch_start:batch_end]
            
            # 创建当前批次的场景
            scene_dict = base_scene_dict.copy()
            
            # 创建shapegroup（每批共享）
            scene_dict['particle_shapegroup'] = {
                'type': 'shapegroup',
                'id': 'particle_group',
                'shape': {
                    'type': 'sphere',
                    'radius': radius
                }
            }
            
            # 添加当前批次的粒子instances
            for local_idx, (pos, color) in enumerate(zip(batch_positions, batch_colors)):
                global_idx = batch_start + local_idx
                scene_dict[f'particle_instance_{local_idx}'] = {
                    'type': 'instance',
                    'shapegroup': {
                        'id': 'particle_group'
                    },
                    'to_world': mi.ScalarTransform4f.translate(pos.tolist()),
                    'bsdf': {
                        'type': 'diffuse',
                        'reflectance': {
                            'type': 'rgb',
                            'value': color.tolist()
                        }
                    }
                }
            
            # 添加地面和背景（只在第一批添加，避免重复）
            if batch_idx == 0:
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
            
            # 渲染当前批次
            scene = mi.load_dict(scene_dict)
            particles_rendered = batch_end
            progress = int(particles_rendered * 100 / num_points)
            print(f'      Batch {batch_idx+1}/{num_batches}: Rendering particles {batch_start:,}-{batch_end-1:,} ({particles_rendered:,}/{num_points:,}, {progress}%)...', 
                  end=' ', flush=True)
            
            batch_image = mi.render(scene)
            batch_images.append(batch_image)
            
            print('Done')
        
        # 合成所有批次的图像（使用alpha混合或叠加）
        print('  Compositing batch images...', end=' ', flush=True)
        final_image = batch_images[0].copy()
        
        # 对于后续批次，使用alpha混合叠加
        for batch_image in batch_images[1:]:
            # 简单的叠加：取最大值（适用于粒子渲染）
            final_image = np.maximum(final_image, batch_image)
        
        print('Done')
        return final_image
    
    def save_image(self, output_file_path, image):
        """
        保存渲染图像
        
        Args:
            output_file_path: 输出文件路径（不含扩展名）
            image: Mitsuba渲染的图像
        """
        mi.util.write_bitmap(f'{output_file_path}.png', image)
    
    def process(self):
        """处理单帧点云"""
        # 加载点云
        positions, normals = self.load_point_cloud()
        
        # 渲染
        image = self.render(positions, normals)
        
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
                 render_batch_size=20000):
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
        render_batch_size: 分批渲染的每批粒子数（None表示不分批，一次性渲染更快但无进度显示）
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
                        samples=samples,
                        render_batch_size=render_batch_size
                    )
                    
                    positions, normals = renderer.load_point_cloud()
                    image = renderer.render(positions, normals)
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
                    samples=samples,
                    render_batch_size=render_batch_size
                )
                
                print('  [1/4] Loading point cloud...', end=' ', flush=True)
                positions, normals = renderer.load_point_cloud()
                print(f'Done ({len(positions):,} points)')
                
                print('  [2/4] Processing and rendering...', flush=True)
                image = renderer.render(positions, normals)
                
                print('  [3/4] Saving image...', end=' ', flush=True)
                output_path = os.path.join(output_folder, renderer.filename)
                renderer.save_image(output_path, image)
                print(f'Done -> {os.path.basename(output_path)}.png')
                
                # 清理内存
                del renderer, positions, normals, image
                import gc
                gc.collect()
                
                successful += 1
                print('  [4/4] Frame completed')
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
    parser.add_argument('--render-batch-size', type=int, default=20000,
                        help='Particles per batch for rendering progress (default: 20000)')
    
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
        num_workers=args.workers,
        render_batch_size=args.render_batch_size
    )

