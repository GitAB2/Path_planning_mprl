import yaml
import numpy as np

# 假设从 environment.py 提取的信息
height = 10  # 环境高度
width = 10   # 环境宽度
unit_size = 100  # 单位大小

# 障碍物信息，使用静态障碍物函数生成
obstacles = [(j, i) for j in range(height) for i in range(width) if (j + 1) % 2 == 0 and 1 <= i <= 8]

# 创建字典以便写入 YAML
environment_info = {
    'environment': {
        'width': width,
        'height': height,
        'unit_size': unit_size,
        'obstacles': obstacles
    }
}

# 将数据写入 YAML 文件
with open('map.yaml', 'w') as yaml_file:
    yaml.dump(environment_info, yaml_file, default_flow_style=False)

print("YAML 文件已生成！")
