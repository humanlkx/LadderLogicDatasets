from ultralytics import YOLO
from cnocr import CnOcr
import cv2
import re
import math
import os
import time

instruction_dict = {
    "contact": "A",        # 常开触点
    "contact_N": "AN",     # 常闭触点
    "contact_down": "FN",  # 下降沿检测
    "contact_P": "FP",      # 上升沿检测
    "coil": "=",  # 线圈输出
    "coil_N": "=N",  # 线圈取反
    "coil_reset": "R",  # 线圈复位
    "coil_set": "S",  # 线圈置位
    "coil_SD": "SD",  # 线圈置位
}

ocr_model = CnOcr()


def filter_yolo_outputs(source_results, confidence_threshold=0.5):
    target_results = []
    boxes = source_results[0].boxes.cpu().numpy()
    for box in boxes:
        if box.conf[0] >= confidence_threshold:
            bbox_coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            class_name = source_results[0].names[class_id]
            target_results.append({
                'loc': {
                    'x1': round(bbox_coords[0],2),
                    'y1': round(bbox_coords[1],2),
                    'x2': round(bbox_coords[2],2),
                    'y2': round(bbox_coords[3],2)
                },
                'class': class_name,
                'confidence': round(box.conf[0],2)
            })
        else:
            continue
    return target_results


def calculate_iou(box1, box2):
    # 计算交集坐标
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算每个框的面积
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    # 避免除零错误
    return inter_area / union_area if union_area > 0 else 0


def merge_boxes(list1, list2, iou_threshold=0.85):
    merged_results = list1.copy()  # 初始结果为第一组框

    for box2 in list2:
        is_merged = False
        for box1 in merged_results:
            # 计算 IoU
            iou = calculate_iou(box1['loc'], box2['loc'])

            if iou > iou_threshold:
                if box1['class'] == box2['class']:
                    # 类别相同，合并框：取置信度更高的框
                    box1['loc'] = {
                        'x1': round((box1['loc']['x1'] + box2['loc']['x1']) / 2, 2),
                        'y1': round((box1['loc']['y1'] + box2['loc']['y1']) / 2, 2),
                        'x2': round((box1['loc']['x2'] + box2['loc']['x2']) / 2, 2),
                        'y2': round((box1['loc']['y2'] + box2['loc']['y2']) / 2, 2)
                    }
                    box1['confidence'] = max(box1['confidence'], box2['confidence'])
                    is_merged = True
                else:
                    # 类别不同，选择置信度更高的框
                    if box2['confidence'] > box1['confidence']:
                        box1['class'] = box2['class']
                        box1['confidence'] = box2['confidence']
                        box1['loc'] = box2['loc']
                    is_merged = True

        # 如果没有与 list1 的任何框合并，则直接添加到结果
        if not is_merged:
            merged_results.append(box2)

    return merged_results


def classify_rungs(boxes):
    # 筛选出 'rung' 框，并按 y1 排序
    rung_boxes = sorted(
        [box for box in boxes if box['class'] == 'rung'],
        key=lambda x: x['loc']['y1']
    )

    classified_groups = []  # 存储分类后的框分组
    unclassified_boxes = boxes.copy()  # 未分类的框列表

    while rung_boxes:
        # 取出当前的 'rung' 框
        outer_box = rung_boxes.pop(0)  # 按排序依次处理
        unclassified_boxes = [box for box in unclassified_boxes if box != outer_box]  # 从未分类框中移除

        # 当前组框列表
        current_group = []

        # 检查其余框是否属于该组
        remaining_boxes = []
        for inner_box in unclassified_boxes:
            if (inner_box['loc']['y1'] >= outer_box['loc']['y1'] and
                    inner_box['loc']['y2'] <= outer_box['loc']['y2']):
                # 满足条件，归入当前组
                current_group.append(inner_box)
            else:
                # 否则保留在剩余列表中
                remaining_boxes.append(inner_box)

        # 统计 current_group 中 'addr' 类别的数量
        addr_count = sum(1 for box in current_group if box['class'] == 'addr')

        # 如果只有 'addr' 或没有 'addr'，跳过该组
        if addr_count == len(current_group) or addr_count == 0:
            # 更新未分类的框列表
            unclassified_boxes = remaining_boxes
            continue

        # 保存当前组，排除 'rung' 类别框
        classified_groups.append([box for box in current_group if box['class'] != 'rung'])

        # 更新未分类的框列表
        unclassified_boxes = remaining_boxes

    return classified_groups


def adjust_code(s):
    # 移除开头的单引号或双引号
    s = re.sub(r'^[\'"]', '', s)

    # 删除开头的特定字母序列，排除 DIMCSTQ
    if s and s[0].isalpha():
        s = re.sub(r'^[^DIMCSTQ]+', '', s, flags=re.IGNORECASE)
    # 替换特定字符
    if "D8" in s or "0B" in s:
        s = s.replace("D8", "DB").replace("0B", "DB")
    elif s.startswith("4M") or s.startswith("KM"):
        s = s.replace("4M", "%M").replace("KM", "%M")
    elif s.startswith("%0") or s.startswith("4Q"):
        s = "%Q" + s[2:]
    elif s.startswith("4") or s.startswith("J"):
        s = "%I" + s[1:]
    elif s.startswith("7"):
        s = "T" + s[1:]
    elif s.startswith("14"):
        s = "%" + s[2:]

    # 处理 纯数字情况
    if s.isdigit():
        return f"%I{s[:-1]}.{s[-1]}"  # 数字转为 %I 格式，默认最后一位前添加 "."
    # 处理带 DBX 的地址
    if "DBX" in s and re.search(r"DBX\d+$", s):
        s = s[:-1] + "." + s[-1]  # 将末尾数字前插入点号

    # 处理 5T# 格式
    if "5T#" in s and (s.startswith("55T") or s.endswith("5")):
        s = "S" + s[1:-1] + "S"
        return s
    elif "T#" in s and s.endswith("5"):
        return s[:-1] + "S"

    # 处理数字开头的地址 (如 124.0) 为 %I 格式
    if re.match(r'^\d+\.\d+$', s):  # 匹配类似 124.0 的格式
        s = f"%I{s}"

    # 处理 %00.* 或 %Q00.* 格式为 %Q*
    if s.startswith("%00.") or s.startswith("%Q00."):
        s = s.replace("%00.", "%Q0.").replace("%Q00.", "%Q0.")

    # 如果不以 % 开头，添加 %
    if not s.startswith("%"):
        s = "%" + s

        # 处理 %MO.* 或 %M0.* 格式为 %M*
    if s.startswith("%MO.") or s.startswith("%M0."):
        s = s.replace("%MO.", "%M0.").replace("%M0.", "%M0.")
    if s.endswith("O"):
        s = s.replace("O", "0")

    return s


def addr_name(bbox, img_num):
    x1 = int(bbox['x1']) - 1
    x2 = int(bbox['x2']) + 1
    y1 = int(bbox['y1']) - 1
    y2 = int(bbox['y2']) + 1
    roi = img_num[y1:y2, x1:x2]
    res = ocr_model.ocr(roi)
    return [line['text'] for line in res]


def classify_elements_with_one_name(classified_result, img):
    label = "Siemens"
    # 提取所有 addr 元素
    addr_elements = []
    for classified in classified_result:
        addr_elements.extend([e for e in classified if e['class'] == 'addr'])

    # 如果没有 addr 元素，直接返回输入结果
    if not addr_elements:
        return classified_result

    class_name = ['addr', 'func', 'or', 'union', 'rung']
    # 遍历分类结果中的 contact 和 coil，寻找最近的 addr
    for classified in classified_result:
        to_remove_addrs = []  # 记录已分配的 addr 元素
        for element in classified:
            if element['class'] not in class_name:
                # 获取目标元素的 <x1, y1>
                target_point = (element['loc']['x1'], element['loc']['y1'])

                # 寻找最近的 addr
                min_distance = float('inf')
                nearest_addr = None
                for addr in addr_elements:
                    # 仅考虑 addr 在目标元素上方的情况
                    if addr['loc']['y1'] < target_point[1]:
                        # 获取 addr 的 <x1, y1>
                        addr_point = (addr['loc']['x1'], addr['loc']['y1'])

                        # 计算欧几里得距离
                        distance = math.sqrt(
                            (target_point[0] - addr_point[0]) ** 2 +
                            (target_point[1] - addr_point[1]) ** 2
                        )
                        if distance < min_distance:
                            min_distance = distance
                            nearest_addr = addr

                # 如果找到最近的 addr，进行 OCR 提取并更新 name 字段
                if nearest_addr:
                    ocr_result = addr_name(nearest_addr['loc'], img)
                    for i, line in enumerate(ocr_result):
                        if '%' in line:
                            if i == 0:  # 第一行
                                label = "Siemens"
                            elif i == len(ocr_result) - 1:  # 最后一行
                                label = "Schneider"
                            break  # 找到 '%' 就可以退出
                    element['name_1'] = ocr_result
                    to_remove_addrs.append(nearest_addr)  # 标记该 addr 为已分配

        # 从 classified 中删除已分配的 addr
        classified[:] = [e for e in classified if e not in to_remove_addrs]

    return classified_result, label


def classify_elements_with_two_name(classified_result, img):

    # 提取所有 addr 元素
    addr_elements = []
    for classified in classified_result:
        addr_elements.extend([e for e in classified if e['class'] == 'addr'])

    # 如果没有 addr 元素，直接返回输入结果
    if not addr_elements:
        return classified_result

    # 遍历分类结果中的目标元素
    for classified in classified_result:
        to_remove_addrs = []  # 记录已分配的 addr 元素
        for element in classified:
            # 仅处理 contact_down, contact_P, coil_SD 类别
            if element['class'] in ['contact_down', 'contact_P', 'coil_SD']:
                # 获取目标元素的 <x1, y1>
                target_point = (element['loc']['x1'], element['loc']['y1'])

                # 查找下方最近的 addr
                nearest_addr_down = None
                min_distance_down = float('inf')

                for addr in addr_elements:
                    addr_point = (addr['loc']['x1'], addr['loc']['y1'])

                    if addr_point[1] > target_point[1]:  # addr 在目标下方
                        distance = math.sqrt(
                            (target_point[0] - addr_point[0]) ** 2 +
                            (target_point[1] - addr_point[1]) ** 2
                        )
                        if distance < min_distance_down:
                            min_distance_down = distance
                            nearest_addr_down = addr

                # 填充下方最近的 addr OCR 结果到 name 字段
                element['name_2'] = addr_name(nearest_addr_down['loc'], img) if nearest_addr_down else None

                # 标记已分配的 addr
                if nearest_addr_down:
                    to_remove_addrs.append(nearest_addr_down)

        # 从 classified 中删除已分配的 addr
        classified[:] = [e for e in classified if e not in to_remove_addrs]

    return classified_result


def find_nearest_addr(addrs, target_x, target_y, condition):
    """
    查找符合条件的最近 addr。
    """
    nearest_addr = None
    min_distance = float('inf')

    for addr in addrs:
        addr_x, addr_y = addr['loc']['x1'], addr['loc']['y1']
        if condition(addr):  # 符合条件
            distance = math.sqrt((target_x - addr_x) ** 2 + (target_y - addr_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_addr = addr

    return nearest_addr


def classify_func_elements(classified_result, img):

    # 提取所有 addr 元素
    addr_elements = []
    for classified in classified_result:
        addr_elements.extend([e for e in classified if e['class'] == 'addr'])

    # 如果没有 addr 元素，直接返回输入结果
    if not addr_elements:
        return classified_result

    # 遍历分类结果中的 func 元素
    for classified in classified_result:
        for element in classified:
            if element['class'] == 'func':  # 仅处理 func 类别
                # 获取 func 元素位置
                func_x1, func_y1 = element['loc']['x1'], element['loc']['y1']
                func_x2 = element['loc']['x2']

                # 查找上方最近的 addr
                nearest_addr_up = find_nearest_addr(
                    addr_elements,
                    func_x1,
                    func_y1,
                    lambda addr: addr['loc']['y1'] < func_y1
                )

                # 查找左侧最近的 addr
                left_addrs = [
                    addr for addr in addr_elements
                    if addr['loc']['x2'] < func_x1 and abs(func_x1 - addr['loc']['x2']) <= 20
                ]
                left_addrs.sort(key=lambda x: x['loc']['y1'])  # 按 y1 排序
                nearest_addr_left = left_addrs[0] if left_addrs else None

                # 查找右侧最近的 addr
                right_addrs = [
                    addr for addr in addr_elements
                    if addr['loc']['x1'] > func_x2 and abs(addr['loc']['x1'] - func_x2) <= 20
                ]
                right_addrs.sort(key=lambda x: x['loc']['y1'])  # 按 y1 排序
                nearest_addr_right = right_addrs[0] if right_addrs else None

                # 填充 func 元素字段
                element['name'] = addr_name(nearest_addr_up['loc'], img) if nearest_addr_up else None  # 上方最近 addr
                element['func_name'] = addr_name(element['loc'], img)  # 自身 OCR 结果
                element['params'] = {
                    'input': addr_name(nearest_addr_left['loc'], img) if nearest_addr_left else None,  # 左侧最近 addr
                    'output': addr_name(nearest_addr_right['loc'], img) if nearest_addr_right else None  # 右侧最近 addr
                }

                # 移除已分配的 addr
                for addr in [nearest_addr_up, nearest_addr_left, nearest_addr_right]:
                    if addr in addr_elements:
                        addr_elements.remove(addr)
                        for group in classified_result:
                            if addr in group:
                                group.remove(addr)

    return classified_result


def segregate_and_allocate_elements(elements, y_threshold=5):
    container_elements = [e for e in elements if e['class'] in ['or', 'union']]
    other_elements = [e for e in elements if e['class'] not in ['or', 'union']]

    def sort_elements_by_relation(elements):
        """
        对元素按包含关系排序并嵌套。
        """
        elements = sorted(elements, key=lambda e: (e['loc']['y1'], e['loc']['x1']))
        arranged_elements = []

        while elements:
            current = elements.pop(0)
            contained = []
            for other in elements[:]:
                if (current['loc']['x1'] <= other['loc']['x1'] and
                        current['loc']['y1'] <= other['loc']['y1'] and
                        current['loc']['x2'] >= other['loc']['x2'] and
                        current['loc']['y2'] >= other['loc']['y2']):
                    contained.append(other)
                    elements.remove(other)
            contained = sort_elements_by_relation(contained)
            arranged_elements.append({"element": current, "contained": contained})
        return arranged_elements

    def allocate_to_contained(elements, other):
        """
        动态嵌套到最内层的 contained 列表中。
        """
        for elem in elements:
            if 'element' in elem:
                elem_loc = elem["element"]['loc']
                other_loc = other['loc']
                if (elem_loc['x1'] <= other_loc['x1'] and
                        elem_loc['y1'] <= other_loc['y1'] and
                        elem_loc['x2'] >= other_loc['x2'] and
                        elem_loc['y2'] >= other_loc['y2']):
                    allocate_to_contained(elem['contained'], other)
                    return
        elements.append({"element": other})

    def sort_contained_elements_by_layers(elements, y_threshold):
        """
        对 `contained` 的元素按层次分组和排序，并更新层次坐标。
        """
        # 先递归处理更深层的 `contained`
        for elem in elements:
            if 'contained' in elem and elem['contained']:
                sort_contained_elements_by_layers(elem['contained'], y_threshold)

        # 然后对当前层次的 `contained` 进行分组和排序
        for elem in elements:
            if 'contained' in elem and elem['contained']:
                # 分组和排序 contained 元素
                contained = elem['contained']
                contained = sorted(contained, key=lambda x: (x['element']['loc']['y1'], x['element']['loc']['x1']))

                # 按 y1 分层次
                layers = []
                current_layer = []
                for c in contained:
                    if not current_layer:
                        current_layer.append(c)
                    elif abs(c['element']['loc']['y1'] - current_layer[0]['element']['loc']['y1']) <= y_threshold:
                        current_layer.append(c)
                    else:
                        layers.append(current_layer)
                        current_layer = [c]
                if current_layer:
                    layers.append(current_layer)

                # 更新 contained 的排序为层次分组（嵌套结构）
                elem['contained'] = layers

                # 更新层次坐标为内部第一层第一个元素的坐标
                if elem['contained'] and elem['contained'][0]:
                    first_child = elem['contained'][0][0]['element']['loc']
                    elem["element"]['loc']['x1'] = first_child['x1']
                    elem["element"]['loc']['y1'] = first_child['y1']

    # 排列容器类元素
    organized_ladder = sort_elements_by_relation(container_elements)

    # 分配其他元素到容器类元素中
    for other in other_elements:
        allocate_to_contained(organized_ladder, other)

    # 对 or 和 union 内部元素分层次排序并更新坐标
    sort_contained_elements_by_layers(organized_ladder, y_threshold)

    return organized_ladder


def sort_by_layers(elements, y_threshold=5):
    # 按 y1 从小到大、x1 从小到大初步排序
    elements = sorted(elements, key=lambda e: (e['loc']['y1'], e['loc']['x1']))

    layers = []  # 分层存储
    current_layer = []  # 当前层

    for element in elements:
        if not current_layer:
            # 初始化第一层
            current_layer.append(element)
        else:
            # 检查是否属于当前层
            if abs(element['loc']['y1'] - current_layer[0]['loc']['y1']) <= y_threshold:
                current_layer.append(element)
            else:
                # 开启新的一层
                layers.append(sorted(current_layer, key=lambda e: e['loc']['x1']))
                current_layer = [element]

    # 添加最后一层
    if current_layer:
        layers.append(sorted(current_layer, key=lambda e: e['loc']['x1']))

    return layers


def ladder_diagram2instruction_list(ladder_diagram, output_file):
    # 定义不处理的 class_name
    excluded_classes = ['or', 'union']

    def process_or(element, f):
        """
        转化 `or` 元素，按递归方式处理。
        """
        f.write("A(\n")  # 开始 `or` 转化
        for layer in element['contained']:  # 逐层处理 `or` 内部元素
            f.write("o(\n")
            for sub_element in layer:
                if sub_element['element']['class'] == 'or':
                    process_or(sub_element, f)
                else:
                    process_other(sub_element['element'], f)
            f.write(")\n")
        f.write(")\n")  # 结束 `or` 转化

    def process_union(element, f):
        """
        转化 `union` 元素，按递归方式处理。
        """
        f.write("=(\n")  # 开始 `union` 转化
        for layer in element['contained']:  # 逐层处理 `union` 内部元素
            for sub_element in layer:
                if sub_element['element']['class'] == 'union':
                    process_union(sub_element, f)
                else:
                    process_other(sub_element['element'], f)
        f.write(")\n")  # 结束 `union` 转化

    def process_func(element, f):
        """
        处理 `func` 类型元素。
        """
        func_name = element.get('func_name', None)
        params = element.get('params', {})
        if any(name in ['S ODT','S_ODT', 'TON', 'Time'] for name in func_name):
            input_value = params.get('input', [])[0]
            input_value = adjust_code(input_value)
            output_value = element['name'][0]
            output_value = adjust_code(output_value)
            f.write(f"L\t{input_value}\n")
            f.write(f"SD\t{output_value}\n")
        elif func_name[0] in ['ADD', 'SUB', 'MUL', 'DIV']:
            input_1 = params.get('input', [])[0]
            input_2 = params.get('input', [])[1]
            operator = {
                'ADD': '+',
                'SUB': '-',
                'MUL': '*',
                'DIV': '/'
            }[func_name[0]]
            input_1 = adjust_code(input_1)
            input_2 = adjust_code(input_2)
            f.write(f"L\t{input_1}\n")
            f.write(f"L\t{input_2}\n")
            f.write(f"{operator}+{func_name[1]}\n")

    def process_other(element, f):
        """
        转化其他元素，按字典方式处理。
        """
        class_type = element['class']
        if class_type == 'func':
            process_func(element, f)
        if class_type in instruction_dict:
            if class_type in ['contact_P', 'contact_down']:
                address = element['name_2'][0]
                address = adjust_code(address)
                instruction = instruction_dict[class_type]
                f.write(f"{instruction}\t{address}\n")
            elif class_type == 'coil_SD':
                address = element['name_2'][0]
                address = adjust_code(address)
                f.write(f"L\t{address}\n")
                address = element['name_1'][0]
                address = adjust_code(address)
                f.write(f"SD\t{address}\n")
            else:
                address = element['name_1'][0]
                address = adjust_code(address)
                instruction = instruction_dict[class_type]
                f.write(f"{instruction}\t{address}\n")

    # 打开输出文件
    with open(output_file, "w") as f:
        # 遍历每一个梯级
        for stage in ladder_diagram:
            # 检查是否包含需要递归处理的类别
            if any(e['class'] in excluded_classes for e in stage):
                # 对包含 `or` 和 `union` 的梯级递归处理
                stage = segregate_and_allocate_elements(stage)
                stage = sorted(stage, key=lambda x: x['element']['loc']['x1'])
                for element in stage:
                    if element['element']['class'] == 'or':
                        process_or(element, f)
                    elif element['element']['class'] == 'union':
                        process_union(element, f)
                    else:
                        process_other(element['element'], f)
                # 写入空行分隔每个梯级
                f.write("\n")
                continue

            # 对其他元素按 x1 从左到右排序
            sorted_elements = sorted(stage, key=lambda x: x['loc']['x1'])

            # 遍历排序后的元素
            for idx, element in enumerate(sorted_elements):
                if element['class'] == 'func':
                    process_func(element, f)
                else:
                    process_other(element, f)

            # 写入空行分隔每个梯级
            f.write("\n")


def main():
    # 模型加载
    model1 = YOLO('yolo10_best.pt')
    model2 = YOLO('yolo11_best.pt')

    # 输入和输出文件夹
    input_folder = "testdata"
    output_folder = "output_results_test"
    os.makedirs(output_folder, exist_ok=True)

    # 统计处理时间
    #start_time = time.time()
    for idx, filename in enumerate(sorted(os.listdir(input_folder))):
        input_path = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"result_{filename[:-4]}.il")

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 读取图像
        img = cv2.imread(input_path)

        # 模型推理
        results1 = model1(img)
        results2 = model2(img)


        # 剔除低置信度结果
        results1 = filter_yolo_outputs(results1)
        results2 = filter_yolo_outputs(results2)

        # 合并结果
        final_result = merge_boxes(results2, results1)

        # 分类和归类
        classified_result = classify_rungs(final_result)
        classified_result, flag = classify_elements_with_one_name(classified_result, img)

        if flag == 'Siemens':
            classified_result = classify_elements_with_two_name(classified_result, img)
            classified_result = classify_func_elements(classified_result, img)

        # 转换并保存指令
        ladder_diagram2instruction_list(classified_result, output_file)

    # 统计总处理时间
    # total_time = time.time() - start_time
    # with open(os.path.join(output_folder, "processing_time.txt"), "w") as time_file:
    #     time_file.write(f"total 耗时: {total_time:.2f} 秒\n")


if __name__ == "__main__":
    main()


# # v10s
# model1 = YOLO('best.pt')
# #  v11s
# model2 = YOLO('test\\11s\\best.pt')
#
# # 读取图像
# img = cv2.imread('testdata\\test.png')
#
# # 分别用两个模型进行推理
# results1 = model1(img)
# results2 = model2(img)
# # 剔除结果中置信度小于0.5的类别
# results1 = filter_yolo_outputs(results1)
# results2 = filter_yolo_outputs(results2)
#
# # 以yolo11s结果为基准进行结果的合并
# final_result = merge_boxes(results2, results1)
#
# # 开始第二部分以rung进行归类
# classified_result = classify_rungs(final_result)
#
# classified_result, flag = classify_elements_with_one_name(classified_result, img)
#
# if flag == 'Siemens':
#     classified_result = classify_elements_with_two_name(classified_result, img)
#     classified_result = classify_func_elements(classified_result, img)
#
# for classified in classified_result:
#     print("    ")
#     for elements in classified:
#         print(elements)
# # 输出文件路径
# output_file = "ladder_to_il_output.il"
#
# # 执行转换
# ladder_diagram2instruction_list(classified_result, output_file)
#
# print(f"指令表已写入文件: {output_file}")
