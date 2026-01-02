import numpy as np
import os
from traj_ball_renderer import TrajectoryBallRenderer


class FixedFrame199Renderer(TrajectoryBallRenderer):
    """使用第199帧的参数（相机位置和尾迹长度）渲染指定帧"""
    
    # 覆盖XML模板，调整target点使飞机居中
    XML_HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{},{},{}" target="0.0,-0.02,0.0" up="0,0,1"/>
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
    
    # 覆盖XML_TAIL，延伸白底以覆盖背景
    XML_TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="20" y="20" z="1"/>
            <translate x="10.0" y="10.0" z="-0.8"/>
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
    
    @staticmethod
    def transform_coordinates(pcl):
        """坐标变换：重新排列位置和速度坐标，统一坐标系（不flip x，保持物理真实坐标）"""
        has_velocity = pcl.shape[1] == 6
        if has_velocity:
            pcl_positions = pcl[:, [2, 0, 1]]
            # 移除 x 轴 flip，保持物理真实坐标
            # pcl_positions[:, 0] *= -1  # 已删除
            pcl_positions[:, 2] += 0.0125
            
            pcl_velocities = pcl[:, [5, 3, 4]]
            # 移除速度 x 轴 flip，保持物理真实坐标
            # pcl_velocities[:, 0] *= -1  # 已删除
            
            return np.column_stack([pcl_positions, pcl_velocities])
        else:
            pcl = pcl[:, [2, 0, 1]]
            # 移除 x 轴 flip，保持物理真实坐标
            # pcl[:, 0] *= -1  # 已删除
            pcl[:, 2] += 0.0125
            return pcl
    
    @staticmethod
    def compute_camera_position(frame_index=0, total_frames=220):
        """根据帧数计算相机位置（从远到近的动画）
        0-199帧：从起始位置拉近到中间位置
        200-219帧：从中间位置拉近到最终位置(-2.0, -1.2, 1.0)
        """
        last_motion_frame = 199
        fade_frames = 20
        
        # 最终位置（第219帧）
        final_pos = (-2.0, -1.2, 1.0)
        
        # 中间位置（第199帧）- 比最终位置稍远一些
        mid_pos = (-2.3, -1.5, 1.2)
        
        # 起始位置（第0帧）- 更远的位置
        start_pos = (-3.5, -2.5, 2.8)
        
        if frame_index <= last_motion_frame:
            # 0-199帧：从起始位置拉近到中间位置
            progress = frame_index / max(last_motion_frame, 1)
            origin_x = start_pos[0] + (mid_pos[0] - start_pos[0]) * progress
            origin_y = start_pos[1] + (mid_pos[1] - start_pos[1]) * progress
            origin_z = start_pos[2] + (mid_pos[2] - start_pos[2]) * progress
        else:
            # 200-219帧：从中间位置拉近到最终位置
            fade_progress = (frame_index - last_motion_frame) / max(fade_frames, 1)
            origin_x = mid_pos[0] + (final_pos[0] - mid_pos[0]) * fade_progress
            origin_y = mid_pos[1] + (final_pos[1] - mid_pos[1]) * fade_progress
            origin_z = mid_pos[2] + (final_pos[2] - mid_pos[2]) * fade_progress
        
        return origin_x, origin_y, origin_z
    
    def _add_velocity_trail(self, xml_segments, position, velocity, point_index=0, frame_index=199):
        """使用第199帧的尾迹长度参数（length_scale = 1.0）"""
        velocity = np.array(velocity, dtype=np.float64)
        vel_norm = np.linalg.norm(velocity)
        
        # 如果速度太小，不添加尾迹
        if vel_norm < 1e-6:
            return
        
        # 始终使用第199帧的尾迹长度缩放因子（length_scale = 1.0）
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


def main():
    FixedFrame199Renderer.init_mitsuba_variant()
    print('=' * 60)
    
    input_folder = 'batch_1'
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
            target_files.append(f'frame_{num:04d}_b1.ply')
        else:
            target_files.append(f'frame_0199_b1.ply')
    
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
            frame_index = frame_numbers[idx]
            print(f'\n[{idx+1}/{total_files}] ({(idx+1)*100//total_files}%) Processing frame {frame_index}: {os.path.basename(ply_file)}')
            print('-' * 60)
            try:
                renderer = FixedFrame199Renderer(ply_file, output_folder=output_folder)
                renderer.process(frame_index=frame_index, total_frames=220)
                print(f'✓ Successfully processed frame {frame_index}: {os.path.basename(ply_file)}')
            except Exception as e:
                print(f'✗ Error processing frame {frame_index} ({os.path.basename(ply_file)}): {str(e)}')
                import traceback
                traceback.print_exc()
    finally:
        FixedFrame199Renderer.cleanup_temp_curves_dir()
    
    print('\n' + '=' * 60)
    print(f'Batch processing completed! Processed {total_files} files.')
    print(f'Output files saved to: {output_folder}/')


if __name__ == '__main__':
    main()

