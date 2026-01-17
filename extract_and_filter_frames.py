#!/usr/bin/env python3
"""
解压 to_render 文件夹下的所有压缩文件，并根据 to_render_indices.txt 保留指定的帧
"""
import os
import zipfile
import re
import shutil
from pathlib import Path


def parse_indices_file(indices_file_path):
    """解析索引文件，返回字典 {zip_index: [frame_indices]}"""
    indices_map = {}
    
    with open(indices_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 逐行解析
    for line in lines:
        line = line.strip()
        # 匹配模式：数字: 数字列表（支持中文和英文逗号）
        match = re.match(r'(\d+):\s*(.+)', line)
        if match:
            zip_index_str = match.group(1)
            frames_str = match.group(2)
            zip_index = int(zip_index_str)
            # 处理中文逗号和英文逗号
            frames_str = frames_str.replace('，', ',')
            # 提取所有数字
            frame_indices = [int(x.strip()) for x in re.split(r'[,，\s]+', frames_str) if x.strip().isdigit()]
            if frame_indices:
                indices_map[zip_index] = frame_indices
    
    return indices_map


def find_zip_file_by_index(zip_index, zip_files):
    """根据索引找到对应的 zip 文件"""
    for zip_file in zip_files:
        # 匹配文件名中的索引，例如 3D_6_... 中的 6
        match = re.search(r'3D_(\d+)_', zip_file)
        if match and int(match.group(1)) == zip_index:
            return zip_file
    return None


def extract_and_filter_zip(zip_path, target_frames, output_base_dir):
    """解压 zip 文件并只保留指定的帧"""
    zip_name = os.path.basename(zip_path)
    zip_stem = os.path.splitext(zip_name)[0]
    extract_dir = os.path.join(output_base_dir, zip_stem)
    
    print(f"\n处理: {zip_name}")
    print(f"  目标帧: {target_frames}")
    
    # 解压 zip 文件
    print(f"  解压到: {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 查找 ply 文件夹
    ply_dir = os.path.join(extract_dir, 'ply')
    if not os.path.exists(ply_dir):
        # 可能直接在根目录
        ply_dir = extract_dir
    
    # 获取所有 ply 文件
    ply_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    
    if not ply_files:
        print(f"  警告: 在 {extract_dir} 中未找到 .ply 文件")
        return
    
    # 提取帧号并保留目标帧
    kept_count = 0
    deleted_count = 0
    
    for ply_file in ply_files:
        # 从文件名提取帧号，例如 pts_3.ply -> 3
        match = re.search(r'pts_(\d+)\.ply', os.path.basename(ply_file))
        if match:
            frame_index = int(match.group(1))
            if frame_index in target_frames:
                kept_count += 1
                print(f"  保留: {os.path.basename(ply_file)}")
            else:
                os.remove(ply_file)
                deleted_count += 1
        else:
            # 如果文件名格式不匹配，也删除
            os.remove(ply_file)
            deleted_count += 1
    
    print(f"  完成: 保留 {kept_count} 个文件，删除 {deleted_count} 个文件")


def main():
    # 使用脚本所在目录，如果脚本在项目根目录，则指向 to_render 子目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'to_render')
    indices_file = os.path.join(base_dir, 'to_render_indices.txt')
    
    # 解析索引文件
    print("解析索引文件...")
    indices_map = parse_indices_file(indices_file)
    print(f"找到 {len(indices_map)} 个索引映射:")
    for zip_idx, frames in indices_map.items():
        print(f"  {zip_idx}: {frames}")
    
    # 获取所有 zip 文件
    zip_files = [f for f in os.listdir(base_dir) if f.endswith('.zip')]
    zip_files = [os.path.join(base_dir, f) for f in zip_files]
    
    print(f"\n找到 {len(zip_files)} 个 zip 文件")
    
    # 处理每个 zip 文件
    processed = 0
    skipped = 0
    
    for zip_index, target_frames in indices_map.items():
        zip_file = find_zip_file_by_index(zip_index, zip_files)
        if zip_file:
            extract_and_filter_zip(zip_file, target_frames, base_dir)
            processed += 1
        else:
            print(f"\n警告: 未找到索引 {zip_index} 对应的 zip 文件")
            skipped += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  成功处理: {processed} 个文件")
    if skipped > 0:
        print(f"  跳过: {skipped} 个索引")


if __name__ == '__main__':
    main()
