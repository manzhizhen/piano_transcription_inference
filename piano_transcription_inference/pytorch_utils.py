import os
import numpy as np
import time
import torch

from .utilities import pad_truncate_sequence


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
 

def forward(model, x, batch_size):
    """Forward data to model in mini-batch.
    向模型传递数据以进行小批量处理。
    
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int

    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
    """
    
    output_dict = {}
    # 获取模型所在的设备信息
    device = next(model.parameters()).device
    
    pointer = 0
    # 计算总的处理段数
    total_segments = int(np.ceil(len(x) / batch_size))
    
    while True:
        print('Segment {} / {}'.format(pointer, total_segments))
        # 如果指针超出范围，则退出循环
        if pointer >= len(x):
            break

        # 将数据移动到指定的设备
        batch_waveform = move_data_to_device(x[pointer: pointer + batch_size], device)
        pointer += batch_size

        with torch.no_grad():  # 不进行梯度计算
            # 设置模型为评估模式
            model.eval()
            # 模型进行前向传播得到输出
            batch_output_dict = model(batch_waveform)

        for key in batch_output_dict.keys():  # 遍历输出字典的键
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())  # 将每个批次的输出添加到输出字典中

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)  # 将每个批次的输出连接成一个整体

    return output_dict
