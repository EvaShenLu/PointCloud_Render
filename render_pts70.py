import os
from traj_b0 import FixedFrame199Renderer


def main():
    FixedFrame199Renderer.init_mitsuba_variant()
    print('=' * 60)
    
    input_file = 'pts_70_normal.ply'
    output_folder = 'render'
    
    if not os.path.exists(input_file):
        print(f'Error: File not found: {input_file}')
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    print(f'Input file: {input_file}')
    print(f'Output folder: {output_folder}')
    print('Trail direction: REVERSED (forward direction)')
    print('=' * 60)
    
    try:
        print(f'\nProcessing: {input_file}')
        print('-' * 60)
        renderer = FixedFrame199Renderer(input_file, output_folder=output_folder)
        # 使用第199帧的参数进行渲染
        renderer.process(frame_index=199, total_frames=220)
        print(f'✓ Successfully processed: {input_file}')
    except Exception as e:
        print(f'✗ Error processing {input_file}: {str(e)}')
        import traceback
        traceback.print_exc()
    finally:
        FixedFrame199Renderer.cleanup_temp_curves_dir()
    
    print('\n' + '=' * 60)
    print(f'Rendering completed!')
    print(f'Output file saved to: {output_folder}/pts_70_normal.png')


if __name__ == '__main__':
    main()
