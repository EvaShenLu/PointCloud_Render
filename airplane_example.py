import numpy as np
import os
import sys
from plyfile import PlyData
import mitsuba as mi
import subprocess
import argparse
from pathlib import Path


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
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
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


class TrajectoryVelRenderer:
    XML_HEAD = XMLTemplates.HEAD
    XML_BALL_SEGMENT = XMLTemplates.BALL_SEGMENT
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path, output_folder=None):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder

    @staticmethod
    def compute_color():
        return np.array([0.3, 0.3, 0.3])

    @staticmethod
    def standardize_point_cloud(pcl):
        """标准化点云位置信息"""
        positions = pcl[:, :3]
        center = np.mean(positions, axis=0)
        scale = np.amax(positions - np.amin(positions, axis=0))
        normalized_positions = ((positions - center) / scale).astype(np.float32)
        return normalized_positions

    @staticmethod
    def transform_coordinates(pcl):
        """坐标变换：重新排列位置坐标，统一坐标系"""
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.0125
        return pcl

    def load_point_cloud(self):
        """加载点云文件，只加载位置信息"""
        file_extension = os.path.splitext(self.file_path)[1]
        if file_extension == '.npy':
            data = np.load(self.file_path, allow_pickle=True)
            # 只使用前3列（位置信息）
            positions = data[:, :3] if data.shape[1] >= 3 else data
            print(f'  Loaded data shape: {positions.shape}')
            return positions
        elif file_extension == '.npz':
            data = np.load(self.file_path)['pred']
            positions = data[:, :3] if data.shape[1] >= 3 else data
            return positions
        elif file_extension == '.ply':
            ply_data = PlyData.read(self.file_path)
            vertex_data = ply_data['vertex']
            data = np.column_stack([vertex_data['x'], vertex_data['y'], vertex_data['z']])
            print(f'  Loaded PLY: shape={data.shape}')
            return data
        else:
            raise ValueError('Unsupported file format.')

    @staticmethod
    def compute_camera_position(frame_index, total_frames=220, actual_data_frames=41):
        """根据帧数计算相机位置
        Args:
            frame_index: 当前帧索引
            total_frames: 总帧数（包括额外的淡出帧）
            actual_data_frames: 实际点云数据的帧数（如41帧，对应frame_0000到frame_0040）
        """
        # 前actual_data_frames帧：平滑移动（使用实际点云数据）
        # 之后10帧：使用最后一帧的点云数据，继续拉近相机
        fade_start_frame = actual_data_frames - 1  # frame_0040对应的索引是40
        fade_frames = 10
        
        if frame_index <= fade_start_frame:
            # 前actual_data_frames帧：从起始位置平滑移动到中间位置
            start_pos = (2.8, 2.8, 3.0)
            mid_pos = (1.8, 1.8, 1.8)  # frame_0040对应的相机位置
            progress = frame_index / max(fade_start_frame, 1)
            origin_x = start_pos[0] + (mid_pos[0] - start_pos[0]) * progress
            origin_y = start_pos[1] + (mid_pos[1] - start_pos[1]) * progress
            origin_z = start_pos[2] + (mid_pos[2] - start_pos[2]) * progress
        else:
            # 后10帧：从中间位置继续移动到最终位置
            mid_pos = (1.8, 1.8, 1.8)
            end_pos = (1.6, 1.6, 1.6)
            fade_progress = (frame_index - fade_start_frame) / max(fade_frames, 1)
            origin_x = mid_pos[0] + (end_pos[0] - mid_pos[0]) * fade_progress
            origin_y = mid_pos[1] + (end_pos[1] - mid_pos[1]) * fade_progress
            origin_z = mid_pos[2] + (end_pos[2] - mid_pos[2]) * fade_progress
        
        return origin_x, origin_y, origin_z

    def generate_xml_content(self, pcl, frame_index=0, total_frames=220, actual_data_frames=41):
        origin_x, origin_y, origin_z = self.compute_camera_position(frame_index, total_frames, actual_data_frames)
        xml_segments = [self.XML_HEAD.format(origin_x, origin_y, origin_z)]
        color = self.compute_color()
        
        for point in pcl:
            position = point[:3]
            
            # 使用小球渲染点云
            xml_segments.append(self.XML_BALL_SEGMENT.format(
                position[0], position[1], position[2],
                color[0], color[1], color[2]
            ))
        
        xml_segments.append(self.XML_TAIL)
        return ''.join(xml_segments)

    @staticmethod
    def save_xml_content_to_file(output_file_path, xml_content):
        xml_file_path = f'{output_file_path}.xml'
        with open(xml_file_path, 'w', encoding='utf-8') as f:
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

    def process(self, frame_index=0, total_frames=220, suffix=None, actual_data_frames=41):
        """处理单帧点云：标准化、坐标变换、渲染
        Args:
            frame_index: 当前帧索引
            total_frames: 总帧数
            suffix: 文件后缀
            actual_data_frames: 实际点云数据的帧数
        """
        pcl = self.load_point_cloud()
        if len(pcl.shape) == 3:
            pcl = pcl[0]
        
        pcl = self.standardize_point_cloud(pcl)
        pcl = self.transform_coordinates(pcl)
        
        # 从文件名中提取后缀（如果未提供）
        if suffix is None:
            # 尝试从文件名中提取后缀（如 frame_0000_b1.ply -> b1）
            if '_b' in self.filename:
                suffix = '_' + self.filename.split('_b')[1].split('.')[0]
            else:
                suffix = '_b1'  # 默认使用b1
        
        output_filename = f'frame_{frame_index:04d}{suffix}'
        
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            output_file_path = os.path.join(self.output_folder, output_filename)
        else:
            output_file_path = os.path.join(self.folder, output_filename)
        
        print('  Generating XML...', end=' ', flush=True)
        xml_content = self.generate_xml_content(pcl, frame_index=frame_index, total_frames=total_frames, actual_data_frames=actual_data_frames)
        xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
        
        print('Rendering...', end=' ', flush=True)
        rendered_scene = self.render_scene(xml_file_path)
        
        print('Saving...', end=' ', flush=True)
        self.save_scene(output_file_path, rendered_scene)
        
        if os.path.exists(xml_file_path):
            os.remove(xml_file_path)
        
        print('Done!')



def create_video_from_frames(frames_dir, output_video_path, framerate=24, pattern="frame_*_b1.png"):
    """使用FFmpeg从渲染的帧创建视频"""
    print(f'\n正在创建视频...')
    print(f'  输入目录: {frames_dir}')
    print(f'  输出视频: {output_video_path}')
    print(f'  帧率: {framerate} fps')
    
    # 检查FFmpeg是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, 
                      check=True, 
                      creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('  错误: 未找到FFmpeg。请确保FFmpeg已安装并添加到系统PATH中。')
        print('  下载FFmpeg: https://ffmpeg.org/download.html')
        return False
    
    # 构建FFmpeg命令
    # Windows路径处理
    if sys.platform == 'win32':
        # Windows上使用绝对路径
        frames_pattern = os.path.join(os.path.abspath(frames_dir), pattern).replace('\\', '/')
        output_path = os.path.abspath(output_video_path)
    else:
        frames_pattern = os.path.join(frames_dir, pattern)
        output_path = output_video_path
    
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-framerate', str(framerate),
        '-pattern_type', 'glob',
        '-i', frames_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    try:
        subprocess.run(cmd, 
                      check=True,
                      creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        print(f'✓ 视频创建成功: {output_video_path}')
        return True
    except subprocess.CalledProcessError as e:
        print(f'✗ 视频创建失败: {e}')
        return False


def process_single_folder(trajectory_ply_folder, output_folder, folder_name):
    """处理单个文件夹的trajectory_ply子文件夹"""
    import re
    
    # 扫描所有PLY文件并按后缀分类
    ply_files_by_suffix = {}  # {suffix: [file_paths]}
    
    if not os.path.exists(trajectory_ply_folder):
        print(f'错误: 输入文件夹不存在: {trajectory_ply_folder}')
        return 0
    
    # 查找所有PLY文件
    all_ply_files = [f for f in os.listdir(trajectory_ply_folder) if f.endswith('.ply')]
    
    if not all_ply_files:
        print(f'错误: 在文件夹中未找到任何PLY文件: {trajectory_ply_folder}')
        return 0
    
    # 按后缀分类文件
    for filename in all_ply_files:
        # 提取后缀（如 frame_0000_b0.ply -> b0）
        match = re.search(r'_b(\d+)\.ply$', filename)
        if match:
            suffix = f'_b{match.group(1)}'
            if suffix not in ply_files_by_suffix:
                ply_files_by_suffix[suffix] = []
            ply_files_by_suffix[suffix].append(os.path.join(trajectory_ply_folder, filename))
    
    if not ply_files_by_suffix:
        print(f'错误: 未找到符合格式的文件（格式应为: frame_XXXX_bN.ply）')
        return 0
    
    # 对每个后缀的文件进行排序
    for suffix in ply_files_by_suffix:
        ply_files_by_suffix[suffix].sort()
    
    print(f'找到 {len(all_ply_files)} 个PLY文件')
    print(f'文件分类: {list(ply_files_by_suffix.keys())}')
    
    # 根据实际文件数量确定实际数据帧数
    # 找到最大的帧号
    max_frame = 0
    for suffix, files in ply_files_by_suffix.items():
        for file_path in files:
            filename = os.path.basename(file_path)
            match = re.search(r'frame_(\d+)_', filename)
            if match:
                frame_num = int(match.group(1))
                max_frame = max(max_frame, frame_num)
    
    actual_data_frames = max_frame + 1  # 实际点云数据帧数（如41帧，对应frame_0000到frame_0040）
    fade_frames = 10  # 额外10帧使用最后一帧的数据
    total_frames = actual_data_frames + fade_frames  # 总帧数（51帧）
    
    print(f'检测到 {actual_data_frames} 个实际数据帧（frame_0000 到 frame_{max_frame:04d}）')
    print(f'将生成 {total_frames} 帧（前{actual_data_frames}帧使用实际数据，后{fade_frames}帧使用frame_{max_frame:04d}的数据）')
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 为每个后缀处理文件
    total_processed = 0
    for suffix, ply_files in ply_files_by_suffix.items():
        suffix_name = suffix.replace('_', '')  # b0 -> b0
        
        print(f'\n处理后缀 {suffix_name}')
        print(f'输出文件夹: {output_folder}')
        print('-' * 60)
        
        # 排序文件并找到最后一帧的文件（用于后续10帧）
        sorted_ply_files = sorted(ply_files)
        last_frame_file = sorted_ply_files[-1]  # frame_0040的文件
        
        # 构建渲染任务列表：前actual_data_frames帧使用实际文件，后10帧使用最后一帧的文件
        render_tasks = []
        for idx, ply_file in enumerate(sorted_ply_files):
            filename = os.path.basename(ply_file)
            match = re.search(r'frame_(\d+)_', filename)
            if match:
                frame_index = int(match.group(1))
            else:
                frame_index = idx
            render_tasks.append((frame_index, ply_file))
        
        # 添加后10帧的任务（使用最后一帧的文件）
        for i in range(fade_frames):
            frame_index = actual_data_frames + i  # 41, 42, ..., 50
            render_tasks.append((frame_index, last_frame_file))
        
        total_files = len(render_tasks)
        print(f'将渲染 {total_files} 帧（{actual_data_frames}个实际数据帧 + {fade_frames}个复制帧）')
        print('-' * 60)
        
        # 批量渲染
        for idx, (frame_index, ply_file) in enumerate(render_tasks):
            filename = os.path.basename(ply_file)
            is_copy_frame = frame_index >= actual_data_frames
            
            frame_info = f'帧 {frame_index} (复制frame_{max_frame:04d})' if is_copy_frame else f'帧 {frame_index}'
            print(f'\n[{idx+1}/{total_files}] ({(idx+1)*100//total_files}%) 处理{frame_info}: {filename}')
            print('-' * 60)
            try:
                renderer = TrajectoryVelRenderer(ply_file, output_folder=output_folder)
                renderer.process(frame_index, total_frames, suffix=suffix, actual_data_frames=actual_data_frames)
                print(f'✓ 成功处理: {frame_info}')
                total_processed += 1
            except Exception as e:
                print(f'✗ 处理失败 {frame_info}: {str(e)}')
                import traceback
                traceback.print_exc()
    
    return total_processed


def main():
    # ============ 配置参数 - 请根据需要修改 ============
    # 基础输入文件夹路径（包含所有轨迹文件夹的父文件夹）
    BASE_INPUT_FOLDER = '40steps_12_traj'
    
    # 输出基础文件夹路径（渲染的PNG图像将保存到这里）
    OUTPUT_BASE_FOLDER = 'render_results'
    
    # 帧率（用于视频创建）
    FRAMERATE = 24
    
    # 是否在渲染完成后创建视频
    CREATE_VIDEO = False
    
    # 是否在完成后压缩结果文件夹
    CREATE_ZIP = False
    # ====================================================
    
    # 使用命令行参数覆盖配置（可选）
    parser = argparse.ArgumentParser(description='批量渲染轨迹点云并生成视频')
    parser.add_argument('--input', type=str, help=f'基础输入文件夹路径 (默认: {BASE_INPUT_FOLDER})')
    parser.add_argument('--output', type=str, help=f'输出基础文件夹路径 (默认: {OUTPUT_BASE_FOLDER})')
    parser.add_argument('--framerate', type=int, default=FRAMERATE, help=f'视频帧率 (默认: {FRAMERATE})')
    parser.add_argument('--video', action='store_true', help='创建视频')
    parser.add_argument('--zip', action='store_true', help='压缩结果文件夹')
    
    args = parser.parse_args()
    
    if args.input:
        BASE_INPUT_FOLDER = args.input
    if args.output:
        OUTPUT_BASE_FOLDER = args.output
    if args.framerate:
        FRAMERATE = args.framerate
    if args.video:
        CREATE_VIDEO = True
    if args.zip:
        CREATE_ZIP = True
    
    # 验证基础输入文件夹
    if not os.path.exists(BASE_INPUT_FOLDER):
        print(f'错误: 基础输入文件夹不存在: {BASE_INPUT_FOLDER}')
        print('请检查路径是否正确，或使用 --input 参数指定正确的路径。')
        return
    
    # 初始化Mitsuba
    TrajectoryVelRenderer.init_mitsuba_variant()
    print('=' * 60)
    
    # 创建输出基础文件夹
    os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)
    
    # 扫描所有包含 "airplane" 的文件夹
    airplane_folders = []
    for item in os.listdir(BASE_INPUT_FOLDER):
        item_path = os.path.join(BASE_INPUT_FOLDER, item)
        if os.path.isdir(item_path) and 'airplane' in item.lower():
            trajectory_ply_path = os.path.join(item_path, 'trajectory_ply')
            if os.path.exists(trajectory_ply_path):
                airplane_folders.append((item, trajectory_ply_path))
    
    if not airplane_folders:
        print(f'错误: 在 {BASE_INPUT_FOLDER} 中未找到任何包含 "airplane" 的文件夹')
        return
    
    print(f'找到 {len(airplane_folders)} 个包含 "airplane" 的文件夹:')
    for folder_name, _ in airplane_folders:
        print(f'  - {folder_name}')
    print('=' * 60)
    
    # 处理每个文件夹
    total_all_processed = 0
    for folder_name, trajectory_ply_folder in airplane_folders:
        print(f'\n\n处理文件夹: {folder_name}')
        print(f'输入路径: {trajectory_ply_folder}')
        output_folder = os.path.join(OUTPUT_BASE_FOLDER, folder_name)
        print(f'输出路径: {output_folder}')
        print('=' * 60)
        
        processed = process_single_folder(trajectory_ply_folder, output_folder, folder_name)
        total_all_processed += processed
        print(f'\n文件夹 {folder_name} 处理完成，已处理 {processed} 个文件')
    
    print('\n\n' + '=' * 60)
    print(f'批量渲染完成! 总共处理 {total_all_processed} 个文件。')
    print(f'输出文件保存到: {OUTPUT_BASE_FOLDER}/')
    
    # 创建视频（为每个文件夹创建视频）
    if CREATE_VIDEO:
        import re
        for folder_name, trajectory_ply_folder in airplane_folders:
            # 扫描该文件夹的所有后缀
            all_ply_files = [f for f in os.listdir(trajectory_ply_folder) if f.endswith('.ply')]
            suffixes = set()
            for filename in all_ply_files:
                match = re.search(r'_b(\d+)\.ply$', filename)
                if match:
                    suffixes.add(f'_b{match.group(1)}')
            
            output_folder_base = os.path.join(OUTPUT_BASE_FOLDER, folder_name)
            
            for suffix in suffixes:
                pattern = f"frame_*{suffix}.png"
                video_path = os.path.join(OUTPUT_BASE_FOLDER, f'{folder_name}.mp4')
                video_created = create_video_from_frames(
                    output_folder_base, 
                    video_path, 
                    framerate=FRAMERATE,
                    pattern=pattern
                )
                if video_created:
                    print(f'✓ 视频已保存到: {video_path}')
    
    # 压缩结果（可选）
    if CREATE_ZIP:
        import shutil
        zip_path = f'{OUTPUT_BASE_FOLDER}.zip'
        print(f'\n正在压缩结果文件夹...')
        try:
            shutil.make_archive(OUTPUT_BASE_FOLDER, 'zip', OUTPUT_BASE_FOLDER)
            print(f'✓ 压缩完成: {zip_path}')
        except Exception as e:
            print(f'✗ 压缩失败: {e}')


if __name__ == '__main__':
    main()