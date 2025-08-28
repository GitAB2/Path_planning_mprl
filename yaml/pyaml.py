import yaml

def save_safe_paths_to_yaml(safe_paths, filename='safe_paths.yaml'):
    """
    将安全路径保存到 YAML 文件中。

    参数:
    safe_paths (list): 包含多条路径的列表，每条路径为一个坐标元组的列表。
    filename (str): 输出的 YAML 文件名，默认值为 'safe_paths.yaml'。
    """
    # 构建一个字典以便写入 YAML
    path_info = {
        'safe_paths': [
            {'path_index': index + 1, 'coordinates': path}
            for index, path in enumerate(safe_paths)
        ]
    }

    # 将路径信息写入 YAML 文件
    with open(filename, 'w') as yaml_file:
        yaml.dump(path_info, yaml_file, default_flow_style=False)

    print(f"YAML 文件 '{filename}' 已生成！")

# 示例调用
safe_paths_example = [
    [(0, 0), (1, 1), (2, 2), (3, 3)],  # 示例路径1
    [(0, 0), (1, 0), (1, 1), (2, 1)],  # 示例路径2
    [(2, 3), (2, 4), (3, 4), (4, 4)]   # 示例路径3
]

# 使用函数保存路径到 YAML 文件
save_safe_paths_to_yaml(safe_paths_example)
