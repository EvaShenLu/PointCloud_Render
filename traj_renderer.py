import numpy as np
import sys
import os
import glob
from plyfile import PlyData
import mitsuba as mi


class XMLTemplates:
    # XML template for the scene (camera, sampler, surface material, etc.)
    HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="2,2,2" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="30"/>
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
    # XML template for a single droplet (teardrop mesh) in the scene
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
    # XML template for the ground plane and the background plane
    TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.2"/>
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
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path, output_folder=None, droplet_mesh_path=None):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_folder = output_folder
        self.droplet_mesh_path = droplet_mesh_path or self._create_droplet_mesh()

    @staticmethod
    def _create_droplet_mesh():
        """创建水滴形状的mesh文件（OBJ格式）"""
        # 创建临时文件夹存储mesh文件
        temp_dir = 'temp_meshes'
        os.makedirs(temp_dir, exist_ok=True)
        mesh_path = os.path.join(temp_dir, 'droplet.obj')
        
        # 如果文件已存在，删除它以便重新生成（确保使用最新的形状）
        if os.path.exists(mesh_path):
            os.remove(mesh_path)
        
        # 生成水滴形状的顶点和面
        # 水滴形状：上半部分是球体，下半部分逐渐变尖
        n_segments = 20
        n_rings = 16
        base_radius = 0.01  # 基础半径（进一步放大）
        length = 0.03  # 水滴总长度（进一步放大）
        
        vertices = []
        normals = []
        faces = []
        
        # 生成顶点和法线
        for i in range(n_rings + 1):
            theta = np.pi * i / n_rings  # 从0到π
            for j in range(n_segments):
                phi = 2 * np.pi * j / n_segments  # 从0到2π
                
                # 水滴形状：上半部分（0到π/3）是球体，下半部分逐渐变尖
                if theta <= np.pi / 3:
                    r = base_radius  # 上半部分保持圆形
                    z_offset = 0
                else:
                    # 下半部分逐渐变尖，使用更陡的曲线
                    t = (theta - np.pi / 3) / (2 * np.pi / 3)
                    # 使用二次曲线让变尖更明显
                    r = base_radius * (1 - t) ** 2  # 从base_radius逐渐缩小到0
                    z_offset = -length * t * 0.8  # 向下延伸
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta) + z_offset
                
                vertices.append([x, y, z])
                
                # 计算法线：从中心（0,0,0）指向顶点
                # 对于所有顶点都使用相同的方法，确保法线正确
                normal = np.array([x, y, z])
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    normal = normal / norm
                else:
                    # 如果顶点在原点（理论上不应该发生），使用默认法线
                    normal = np.array([0, 0, 1])
                
                normals.append(normal)
        
        # 生成面（确保顶点顺序正确，法线指向外）
        for i in range(n_rings):
            for j in range(n_segments):
                v0 = i * n_segments + j
                v1 = i * n_segments + (j + 1) % n_segments
                v2 = (i + 1) * n_segments + j
                v3 = (i + 1) * n_segments + (j + 1) % n_segments
                
                # 两个三角形组成一个四边形（反转顺序以确保法线指向外）
                faces.append([v2, v1, v0])  # 反转顺序
                faces.append([v2, v3, v1])  # 反转顺序
        
        # 计算法线：从中心指向顶点（简单直接的方法）
        normals = []
        for v in vertices:
            normal = np.array(v)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normals.append(normal / norm)
            else:
                normals.append(np.array([0, 0, 1]))
        
        # 写入OBJ文件（包含法线）
        with open(mesh_path, 'w') as f:
            # 写入顶点
            for v in vertices:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
            
            # 写入法线
            for n in normals:
                f.write(f'vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n')
            
            # 写入面（包含法线索引）
            for face in faces:
                # 格式：f v1//vn1 v2//vn2 v3//vn3
                f.write(f'f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n')
        
        return os.path.abspath(mesh_path)

    @staticmethod
    def compute_color(x, y, z, noise_seed=0):
        g = 0.3 
        return np.array([g, g, g])

    @staticmethod
    def generate_random_rotation_matrix(seed, translation):
        """生成包含随机旋转和平移的变换矩阵（Rodrigues旋转公式）"""
        np.random.seed(seed)
        # 随机旋转轴（归一化）
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        # 随机旋转角度（0到2π）
        angle = np.random.uniform(0, 2 * np.pi)
        
        # 使用Rodrigues公式计算旋转矩阵
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        
        # 创建4x4齐次变换矩阵：先旋转，再平移
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = translation
        
        # 返回矩阵的16个元素（按行优先）
        return matrix.flatten()

    @staticmethod
    def standardize_point_cloud(pcl):
        center = np.mean(pcl, axis=0)
        scale = np.amax(pcl - np.amin(pcl, axis=0))
        return ((pcl - center) / scale).astype(np.float32)

    def load_point_cloud(self):
        file_extension = os.path.splitext(self.file_path)[1]
        if file_extension == '.npy':
            return np.load(self.file_path, allow_pickle=True)
        elif file_extension == '.npz':
            return np.load(self.file_path)['pred']
        elif file_extension == '.ply':
            ply_data = PlyData.read(self.file_path)
            return np.column_stack([ply_data['vertex'][t] for t in ('x', 'y', 'z')])
        else:
            raise ValueError('Unsupported file format.')

    def generate_xml_content(self, pcl):
        xml_segments = [self.XML_HEAD]
        pcl_min = np.min(pcl, axis=0)
        pcl_max = np.max(pcl, axis=0)
        pcl_range = pcl_max - pcl_min
        
        for idx, point in enumerate(pcl):
            normalized_point = (point - pcl_min) / (pcl_range + 1e-8)
            color = self.compute_color(
                normalized_point[0], normalized_point[1], normalized_point[2], 
                noise_seed=idx)
            
            # 生成包含随机旋转和平移的变换矩阵
            transform_matrix = self.generate_random_rotation_matrix(idx, point)
            
            xml_segments.append(self.XML_DROPLET_SEGMENT.format(
                self.droplet_mesh_path,
                *transform_matrix,  # 变换矩阵（16个值：旋转+平移）
                color[0], color[1], color[2]  # 颜色
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

    @staticmethod
    def render_scene(xml_file_path):
        scene = mi.load_file(xml_file_path)
        img = mi.render(scene)
        return img

    @staticmethod
    def save_scene(output_file_path, rendered_scene):
        mi.util.write_bitmap(f'{output_file_path}.png', rendered_scene)

    def process(self):
        pcl_data = self.load_point_cloud()
        if len(pcl_data.shape) < 3:
            pcl_data = pcl_data[np.newaxis, :, :]

        total_frames = len(pcl_data)
        for index, pcl in enumerate(pcl_data):
            pcl = self.standardize_point_cloud(pcl)
            pcl = pcl[:, [2, 0, 1]]
            pcl[:, 0] *= -1
            pcl[:, 2] += 0.0125

            output_filename = f'{self.filename}'
            if self.output_folder:
                os.makedirs(self.output_folder, exist_ok=True)
                output_file_path = os.path.join(self.output_folder, output_filename)
            else:
                output_file_path = os.path.join(self.folder, output_filename)
            
            if total_frames > 1:
                print(f'  Frame {index+1}/{total_frames}: Generating XML...', end=' ', flush=True)
            else:
                print(f'  Generating XML...', end=' ', flush=True)
            
            xml_content = self.generate_xml_content(pcl)
            xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
            
            print('Rendering...', end=' ', flush=True)
            rendered_scene = self.render_scene(xml_file_path)
            
            print('Saving...', end=' ', flush=True)
            self.save_scene(output_file_path, rendered_scene)
            
            if os.path.exists(xml_file_path):
                os.remove(xml_file_path)
            
            print('Done!')

    @staticmethod
    def cleanup_temp_meshes():
        """清理临时mesh文件"""
        temp_dir = 'temp_meshes'
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)


def main(argv):
    TrajectoryRenderer.init_mitsuba_variant()
    print('=' * 60)
    
    input_folder = 'ply'
    output_folder = 'render'
    
    # 测试单个文件
    target_files = ['pts_0.ply']
    
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
    
    try:
        for idx, ply_file in enumerate(ply_files, 1):
            print(f'\n[{idx}/{total_files}] ({idx*100//total_files}%) Processing: {os.path.basename(ply_file)}')
            print('-' * 60)
            try:
                renderer = TrajectoryRenderer(ply_file, output_folder=output_folder)
                renderer.process()
                print(f'✓ Successfully processed: {os.path.basename(ply_file)}')
            except Exception as e:
                print(f'✗ Error processing {os.path.basename(ply_file)}: {str(e)}')
    finally:
        # 清理临时mesh文件
        TrajectoryRenderer.cleanup_temp_meshes()
    
    print('\n' + '=' * 60)
    print(f'Batch processing completed! Processed {total_files} files.')
    print(f'Output files saved to: {output_folder}/')


if __name__ == '__main__':
    main(sys.argv)

