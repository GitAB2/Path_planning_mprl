import argparse
import yaml
from reportlab.pdfgen import canvas

def create_grid_map(filename, obstacles, path=None, rows=10, cols=10, cell_size=20, margin=20):
        """
        创建栅格地图PDF文件(含路径绘制)
        参数：
        filename - 输出文件名
        path     - 路径坐标列表 [(x0,y0), (x1,y1)...]
        rows     - 行数
        cols     - 列数
        cell_size- 单元格大小（单位：点）
        margin   - 页边距
        """
        # 初始化路径参数
        
        path = path or []
        
        # 计算页面尺寸
        page_width = cols * cell_size + 2 * margin
        page_height = rows * cell_size + 2 * margin
        filename = "map/"+filename
        # 创建PDF画布
        c = canvas.Canvas(filename, pagesize=(page_width, page_height))
        
        # 绘制栅格线
        for i in range(rows + 1):
            y = margin + i * cell_size
            c.line(margin, y, page_width - margin, y)
            
        for j in range(cols + 1):
            x = margin + j * cell_size
            c.line(x, margin, x, page_height - margin)
        
        # 绘制障碍物
        for (x, y) in obstacles:
            pdf_y = rows - y - 1
            c.rect(
                margin + x * cell_size,
                margin + pdf_y * cell_size,
                cell_size, cell_size,
                fill=1,
                stroke=0
            )
        # for (x,y) in candidate_postions:
        #     pdf_y = rows - y - 1
        #     center_x = margin + x * cell_size + cell_size/2
        #     center_y = margin + pdf_y * cell_size + cell_size/2
        #     radius = cell_size*0.4
        #     c.circle(center_x, center_y, radius, fill=1, stroke=0)

        # # 插入自定义图片（新增功能）
        # for img_info in self.images:
        #     try:
        #         # 加载图片
        #         img = ImageReader(img_info["path"])
                
        #         # 计算插入位置
        #         grid_x, grid_y = img_info["grid_pos"]
        #         offset_x, offset_y = img_info.get("offset", (0, 0))
                
        #         # 坐标转换
        #         pdf_y = rows - grid_y - 1  # 转换为PDF坐标系
        #         x_pos = margin + grid_x * cell_size + offset_x
        #         y_pos = margin + pdf_y * cell_size + offset_y
                
        #         # 绘制图片
        #         c.drawImage(
        #             img,
        #             x_pos,
        #             y_pos,
        #             width=img_info["size"][0],
        #             height=img_info["size"][1],
        #             preserveAspectRatio=True,
        #             mask='auto'
        #         )
        #     except Exception as e:
        #         print(f"图片插入失败：{str(e)}")

        # 绘制路径（新增功能）
        if len(path) >= 2:
            # 设置路径样式
            c.setStrokeColorRGB(0, 0, 1)  # 蓝色路径
            c.setLineWidth(3)             # 3点线宽
            
            # 绘制路径线段
            for i in range(len(path)-1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                
                # 坐标系转换
                pdf_y1 = rows - y1 - 1
                pdf_y2 = rows - y2 - 1
                
                # 计算线段端点坐标
                start_x = margin + x1 * cell_size + cell_size/2
                start_y = margin + pdf_y1 * cell_size + cell_size/2
                end_x = margin + x2 * cell_size + cell_size/2
                end_y = margin + pdf_y2 * cell_size + cell_size/2
                
                c.line(start_x, start_y, end_x, end_y)
            
            # 绘制路径端点标记
            # self._draw_path_markers(c, path, rows, cols, cell_size, margin)
        c.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create grid map with path")
    parser.add_argument("--param",type=str,default="yaml/40x40_obst_origin.yaml", help="Input file containing map and obstacles")
    parser.add_argument("--output",type=str,default="40_map_origin.pdf", help='output file with the schedule')
    args = parser.parse_args()

    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    create_grid_map(args.output, obstacles, rows=dimension[0], cols=dimension[1])
    
