import tensorflow as tf
import numpy as np
import os

def train_model(model, train_generator, val_generator, epochs=20, model_save_path='best_model.h5'):
    """
    训练模型
    
    参数:
        model: 要训练的模型
        train_generator: 训练数据生成器
        val_generator: 验证数据生成器
        epochs: 训练轮数
        model_save_path: 模型保存路径
        
    返回:
        训练历史记录
    """
    # 回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    ]
    
    # 训练模型
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, test_generator):
    """
    评估模型性能
    
    参数:
        model: 要评估的模型
        test_generator: 测试数据生成器
        
    返回:
        评估结果字典
    """
    results = model.evaluate(test_generator)
    return dict(zip(model.metrics_names, results))

def predict_proba(model, generator):
    """
    获取模型预测概率
    
    参数:
        model: 训练好的模型
        generator: 数据生成器
        
    返回:
        预测概率和真实标签
    """
    # 获取所有批次的预测
    y_pred = []
    y_true = []
    generator.reset()
    for i in range(len(generator)):
        x, y = generator.next()
        y_pred.extend(model.predict(x))
        y_true.extend(y)
    
    return np.array(y_pred), np.array(y_true)