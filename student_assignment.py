#!/usr/bin/env python3

"""
学生作业：实现Memory-Mapped数据集
执行：
```
python student_assignment.py
```
🎯 作业目标：
基于传统的TraditionalDataset实现，创建一个内存友好的MmapDataset类

📋 提交要求：
1. 完成下面的 MmapDataset 类实现
2. 运行测试对比两种方式的内存使用

"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import time
import psutil
import pickle
import mmap
import struct
import gc


def get_memory_mb():
    """获取当前进程内存使用量（MB）"""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


class TraditionalDataset(Dataset):
    """
    传统实现（已完成，无需修改）
    问题：一次性加载所有数据到内存，占用大量内存
    """
    
    def __init__(self, data_file, labels_file):
        print("🔄 传统方式：正在将整个数据集加载到内存...")
        
        with open(data_file + '.meta', 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.num_samples = self.metadata['num_samples']
        self.data_shape = self.metadata['data_shape']
        
        # 一次性读取所有数据到内存
        with open(data_file, 'rb') as f:
            data_bytes = f.read()
        with open(labels_file, 'rb') as f:
            labels_bytes = f.read()
        
        # 转换为numpy数组，再转为tensor
        total_shape = (self.num_samples,) + self.data_shape
        data_array = np.frombuffer(data_bytes, dtype=np.float32).reshape(total_shape)
        labels_array = np.frombuffer(labels_bytes, dtype=np.int32)
        
        self.data = torch.from_numpy(data_array.copy())
        self.labels = torch.from_numpy(labels_array.copy())
        
        print(f"✅ 传统方式加载完成：{self.data.shape}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MmapDataset(Dataset):
    """
    🚧 【学生实现区域】Memory-Mapped数据集实现
    
    💡 实现提示：
    1. 使用mmap模块创建内存映射，而不是一次性读取文件
    2. 实现懒加载：只在__getitem__中按需读取数据
    3. 使用np.frombuffer()直接从mmap创建numpy数组视图（零拷贝）
    4. 正确计算每个样本在文件中的偏移量
    
    📚 需要的模块：
    - mmap: 创建内存映射
    - os: 打开文件描述符
    - struct: 解析二进制数据（标签）
    - numpy: 创建数组视图
    """
    
    def __init__(self, data_file, labels_file):
        """
        🎯 TODO 1: 初始化MmapDataset
        
        要求：
        1. 读取元数据文件 (data_file + '.meta')
        2. 使用os.open()以只读模式打开数据文件和标签文件
        3. 使用mmap.mmap()创建内存映射
        4. 保存必要的信息用于后续数据访问
        
        提示：
        - self.metadata = pickle.load(...)
        metadata = {
        'num_samples': num_samples,
        'data_shape': image_shape,
        'sample_size': sample_size
        }
        - self._data_fd = os.open(data_file, os.O_RDONLY)
        - self._data_mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        """
        print("🚀 Mmap方式：创建内存映射（不加载数据到内存）...")
        
        # TODO: 在这里实现初始化代码
        # 步骤1: 读取元数据
        # 步骤2: 打开文件并创建内存映射
        # 步骤3: 保存必要的信息
        
        pass  # 删除这行，写入你的实现
    
    def __len__(self):
        """
        🎯 TODO 2: 返回数据集大小
        
        提示：从元数据中获取样本数量
        """
        pass  # 删除这行，写入你的实现
    
    def __getitem__(self, idx):
        """
        🎯 TODO 3: 获取单个样本（核心部分）
        
        要求：
        1. 计算样本idx在数据文件中的字节偏移量
        2. 使用np.frombuffer()从内存映射创建数组视图（零拷贝）
        3. 读取对应的标签
        4. 转换为PyTorch tensor并返回
        
        提示：
        - 数据偏移量 = idx * sample_size（字节）
        - 标签偏移量 = idx * 4（int32 = 4字节）
        - np.frombuffer(mmap, dtype=np.float32, count=..., offset=...)
        - struct.unpack('i', label_bytes)[0] 或 np.frombuffer()解析标签
        """
        pass  # 删除这行，写入你的实现
    
    def __del__(self):
        """
        🎯 TODO 4: 清理资源
        
        要求：
        1. 关闭内存映射
        2. 关闭文件描述符
        3. 使用try-except防止重复关闭的错误
        """
        pass  # 删除这行，写入你的实现


def create_test_dataset(num_samples=8000, image_shape=(3, 64, 64)):
    """创建测试数据集（已完成，无需修改）"""
    data_file = 'student_test_data.bin'
    labels_file = 'student_test_labels.bin'
    
    print(f"📁 创建测试数据集：{num_samples} 个样本，形状 {image_shape}")
    
    sample_size = np.prod(image_shape) * 4
    total_size_mb = num_samples * sample_size / 1024 / 1024
    print(f"   预计大小：{total_size_mb:.1f} MB")
    
    # 分批写入数据
    batch_size = 1000
    np.random.seed(42)
    
    with open(data_file, 'wb') as f:
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_data = np.random.randn(end_idx - i, *image_shape).astype(np.float32)
            f.write(batch_data.tobytes())
    
    # 写入标签
    labels = np.random.randint(0, 10, num_samples, dtype=np.int32)
    with open(labels_file, 'wb') as f:
        f.write(labels.tobytes())
    
    # 保存元数据
    metadata = {
        'num_samples': num_samples,
        'data_shape': image_shape,
        'sample_size': sample_size
    }
    with open(data_file + '.meta', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ 测试数据集创建完成")
    return data_file, labels_file, total_size_mb


def test_dataset(dataset_class, data_file, labels_file, name):
    """测试数据集性能（已完成，无需修改）"""
    print(f"\n{'='*50}")
    print(f"🧪 测试 {name}")
    print(f"{'='*50}")
    
    gc.collect()
    initial_memory = get_memory_mb()
    
    # 创建数据集
    start_time = time.time()
    try:
        dataset = dataset_class(data_file, labels_file)
        init_time = time.time() - start_time
        
        after_init_memory = get_memory_mb()
        init_memory_increase = after_init_memory - initial_memory
        
        print(f"⏱️  初始化时间：{init_time:.2f} 秒")
        print(f"📈 初始化内存增长：{init_memory_increase:.1f} MB")
        
        # 测试数据访问
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        max_memory = after_init_memory
        for i, (data, labels) in enumerate(dataloader):
            current_memory = get_memory_mb()
            max_memory = max(max_memory, current_memory)
            
            if i >= 30:  # 测试30个batch
                break
            
            if i % 10 == 0:
                print(f"   批次 {i:2d}: 内存 {current_memory:6.1f} MB, 数据形状 {data.shape}")
        
        total_memory_increase = max_memory - initial_memory
        
        print(f"📊 {name} 结果：")
        print(f"   📈 总内存增长：{total_memory_increase:.1f} MB")
        print(f"   🔝 峰值内存：{max_memory:.1f} MB")
        
        # 清理
        del dataset, dataloader
        gc.collect()
        
        return {
            'init_time': init_time,
            'memory_increase': total_memory_increase,
            'peak_memory': max_memory
        }
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        print(f"💡 检查你的 {dataset_class.__name__} 实现")
        return None


def main():
    """主测试函数"""
    print("🎓 学生作业：Memory-Mapped数据集实现")
    print("=" * 60)
    
    # 创建测试数据
    data_file, labels_file, file_size = create_test_dataset(num_samples=6000)
    print(f"📁 测试数据大小：{file_size:.1f} MB")
    
    # 测试传统方式
    traditional_result = test_dataset(TraditionalDataset, data_file, labels_file, "传统数据集")
    
    # 测试Mmap方式
    mmap_result = test_dataset(MmapDataset, data_file, labels_file, "Mmap数据集")
    
    # 对比结果
    if traditional_result and mmap_result:
        print(f"\n{'='*60}")
        print("📊 对比结果")
        print(f"{'='*60}")
        
        print(f"📁 数据集大小：{file_size:.1f} MB")
        print(f"📈 传统方式内存：{traditional_result['memory_increase']:.1f} MB")
        print(f"📈 Mmap方式内存：{mmap_result['memory_increase']:.1f} MB")
        
        if mmap_result['memory_increase'] > 0:
            memory_saved = traditional_result['memory_increase'] - mmap_result['memory_increase']
            percentage_saved = (memory_saved / traditional_result['memory_increase']) * 100
            print(f"💾 内存节省：{memory_saved:.1f} MB ({percentage_saved:.1f}%)")
            
            if percentage_saved > 30:
                print("🎉 优秀！你的Mmap实现显著节省了内存！")
            elif percentage_saved > 10:
                print("✅ 不错！你的Mmap实现节省了一些内存")
            else:
                print("🤔 Mmap实现可能还有优化空间")
        
        print(f"\n🎯 作业评估：")
        if mmap_result['memory_increase'] < traditional_result['memory_increase']:
            print("✅ Mmap实现成功减少了内存使用")
        else:
            print("❌ Mmap实现没有减少内存使用，请检查实现")
    
    # 清理文件
    for f in ['student_test_data.bin', 'student_test_labels.bin', 'student_test_data.bin.meta']:
        try:
            os.remove(f)
        except:
            pass



if __name__ == "__main__":
    main()
