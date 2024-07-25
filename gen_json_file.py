import os
import json

conditioning_image_folder = ""
image_folder = ""
text_folder = ""

output_path = ""  # 指定保存train.json文件的路径

conditioning_images = os.listdir(conditioning_image_folder)
images = os.listdir(image_folder)
texts = os.listdir(text_folder)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as json_file:
    for filename in conditioning_images:
        name, _ = os.path.splitext(filename)
        conditioning_image_path = os.path.join(conditioning_image_folder, filename)
        image_path = os.path.join(image_folder, filename)
        text_path = os.path.join(text_folder, name + ".txt")

        with open(text_path, 'r', encoding='utf-8') as file:
            text_content = file.read().strip()
        data = {
            "text": text_content,
            "image": image_path,
            "conditioning_image": conditioning_image_path
        }
        json_file.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"数据已成功写入{output_path}文件中")