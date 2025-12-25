import numpy as np
import sys
import os
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
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="matteStone">
        <string name="distribution" value="ggx"/>
        <float name="intIOR" value="1.5"/>
        <float name="alpha" value="0.35"/> 
    </bsdf>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.1"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>

    <bsdf type="roughplastic" id="matteMarble">
        <string name="distribution" value="ggx"/>
        <float name="intIOR" value="1.5"/>
        <float name="alpha" value="0.6"/>
        <rgb name="diffuseReflectance" value="0.5,0.5,0.5"/>
    </bsdf>
"""
    # XML template for a single point (ball) in the scene
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.01"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
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
    
    <!-- 顶光：从正上方照射，产生明显的阴影 -->
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


class PointCloudRenderer:
    XML_HEAD = XMLTemplates.HEAD
    XML_BALL_SEGMENT = XMLTemplates.BALL_SEGMENT
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)

    @staticmethod
    def compute_color(x, y, z, noise_seed=0):
        # 用 z 作为主渐变（从下到上：z小=深灰，z大=浅灰）
        # 确保渐变方向正确，避免中间白两边灰
        t = np.clip(z, 0.0, 1.0)
        
        # 用 gamma 曲线增强对比
        t = t ** 0.7
        
        # 整体偏黑灰：从深灰（0.1）到中灰（0.4），不再到浅灰
        g = 0.1 + 0.3 * t
        
        # 添加非常轻的随机纹理扰动
        np.random.seed(noise_seed)
        noise = 0.02 * np.random.randn()
        g = np.clip(g + noise, 0.08, 0.45)
        
        return np.array([g, g, g])

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
        # 计算点云的边界用于归一化颜色计算
        pcl_min = np.min(pcl, axis=0)
        pcl_max = np.max(pcl, axis=0)
        pcl_range = pcl_max - pcl_min
        pcl_center = (pcl_min + pcl_max) / 2.0
        
        for idx, point in enumerate(pcl):
            # 归一化坐标用于颜色计算（使渐变更明显）
            normalized_point = (point - pcl_min) / (pcl_range + 1e-8)
            color = self.compute_color(
                normalized_point[0], normalized_point[1], normalized_point[2], 
                noise_seed=idx)
            xml_segments.append(self.XML_BALL_SEGMENT.format(
                point[0], point[1], point[2], *color))
        xml_segments.append(self.XML_TAIL)
        return ''.join(xml_segments)

    @staticmethod
    def save_xml_content_to_file(output_file_path, xml_content):
        xml_file_path = f'{output_file_path}.xml'
        with open(xml_file_path, 'w') as f:
            f.write(xml_content)
        return xml_file_path

    @staticmethod
    def render_scene(xml_file_path):
        mi.set_variant('scalar_rgb')
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

        for index, pcl in enumerate(pcl_data):
            pcl = self.standardize_point_cloud(pcl)
            pcl = pcl[:, [2, 0, 1]]
            pcl[:, 0] *= -1
            pcl[:, 2] += 0.0125

            output_filename = f'{self.filename}_{index:02d}'
            output_file_path = f'{self.folder}/{output_filename}'
            print(f'Processing {output_filename}...')
            xml_content = self.generate_xml_content(pcl)
            xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
            rendered_scene = self.render_scene(xml_file_path)
            self.save_scene(output_file_path, rendered_scene)
            print(f'Finished processing {output_filename}.')


def main(argv):
    if len(argv) < 2:
        print('Filename not provided as argument.')
        return

    renderer = PointCloudRenderer(argv[1])
    renderer.process()


if __name__ == '__main__':
    main(sys.argv)
