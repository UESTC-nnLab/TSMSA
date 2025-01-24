import os
import pdb
def rename_images_in_directory(directory, start_index=0):
    """
    重命名指定目录下的图片文件，按照指定的起始索引开始计数。

    :param directory: 字符串，图片所在目录的路径。
    :param start_index: 整数，重命名后文件的起始索引，默认为0。
    """
    # 确保目录存在
    
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    # 获取目录下的所有文件
    files = os.listdir(directory)
    # pdb.set_trace()
    # pdb.set_trace()
    # 筛选并排序符合条件的图片文件
    image_files = sorted([f for f in files if  f.endswith('.png') or f.endswith('.jpg')], key=lambda x: int(x.split('.')[0])) #x.split('-')[1]
    # pdb.set_trace()
    # 重命名文件
    for index, filename in enumerate(image_files, start=start_index):
        # 构建新的文件名
        base_name, ext = os.path.splitext(filename)
        new_filename = f'{index}{ext}'
        # 构建完整的原文件路径和新文件路径
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed "{filename}" to "{new_filename}"')

    print(f"All files in {directory} have been renamed starting from index {start_index}.")

# 示例用法
if __name__=='__main__':
    for i in range(1,121):
        directories_to_process = ['/home/pengshuang/Public/NUDT_MIRSDT/masks/'+f"sequence{i}"]#, 'path/to/directory2' 'images/'+str(i), 
        # save_dirctory='Sequences/train/'+str(i)+'/modified_masks'
        for directory in directories_to_process:
            print(f"Processing directory: {directory}")
            try:
                rename_images_in_directory(directory,start_index=0)
            except:
                print("no directory",i)

        