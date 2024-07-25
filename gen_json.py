import os
import json

def generate_json(conditioning_image_folder, image_folder, text_folder, output_path):
    # 获取所有文件名
    conditioning_images = os.listdir(conditioning_image_folder)
    images = os.listdir(image_folder)
    texts = os.listdir(text_folder)
    output_path = output_path + '\\train.json'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 打开文件并写入数据
    with open(output_path, 'w', encoding='utf-8') as json_file:
        # 遍历所有文件
        for filename in conditioning_images:
            # 获取文件名（不包含扩展名）
            name, _ = os.path.splitext(filename)

            # 构建文件路径
            conditioning_image_path = os.path.join(conditioning_image_folder, filename)
            image_path = os.path.join(image_folder, filename)
            text_path = os.path.join(text_folder, name + ".txt")

            # 读取文本内容
            with open(text_path, 'r', encoding='utf-8') as file:
                text_content = file.read().strip()

            # 构建字典
            data = {
                "text": text_content,
                "image": image_path,
                "conditioning_image": conditioning_image_path
            }

            # 将数据写入文件，每个对象占一行
            json_file.write(json.dumps(data, ensure_ascii=False) + '\n')

    return f"数据已成功写入 {output_path} 文件中"