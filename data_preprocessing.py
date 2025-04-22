import os
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances

def load_and_preprocess_data(data_dir, img_size=(224, 224), batch_size=32, test_size=0.2):
    """
    加载和预处理CT图像数据
    
    参数:
        data_dir: 数据目录路径
        class_name: 类别名称（默认使用 'COVID-19'）
        img_size: 图像目标尺寸
        batch_size: 批量大小
        test_size: 验证集比例
        
    返回:
        预处理后的训练集、验证集和测试集
    """
    
    # 创建数据生成器
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=test_size,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    # 加载训练数据
    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # 二分类任务
        shuffle=True
    )
    
    # 加载验证数据
    val_generator = datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    # 加载测试数据
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def prepare_mrf_data(images, labels):
    """
    准备MRF优化所需的数据格式
    
    参数:
        images: 图像数据
        labels: 对应标签
        
    返回:
        适合MRF处理的数据格式
    """
    # 检查输入数据
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("Input images or labels are empty.")
    # 将图像数据展平
    flattened_images = np.reshape(images, (images.shape[0], -1))
    
    # 创建邻接矩阵（这里使用简单的欧氏距离）
    distances = euclidean_distances(flattened_images)
    adjacency = np.exp(-distances / distances.mean())
    
    return {
        'features': flattened_images,
        'labels': labels,
        'adjacency': adjacency
    }