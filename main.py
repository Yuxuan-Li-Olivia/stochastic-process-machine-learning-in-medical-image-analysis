import os
import numpy as np
import tensorflow as tf
from data_preprocessing import load_and_preprocess_data, prepare_mrf_data
from model import build_softmax_model, build_mlp_model
from train import train_model, evaluate_model, predict_proba
from mrf_optimization import MRFOptimizer
from evaluate import plot_training_history, plot_confusion_matrix, plot_roc_curve, print_classification_report

def main():
    # Download latest version
    data_dir  = r"/mnt/e/archive/COVID-19/COVID-19"
    # 设置参数
    img_size = (224, 224)
    batch_size = 32
    epochs = 20
    
    # 1. 数据预处理
    print("Loading and preprocessing data...")
    train_gen, val_gen, test_gen = load_and_preprocess_data(
        data_dir, img_size, batch_size)
    
    # 2. 构建和训练模型
    print("Building and training model...")
    model = build_softmax_model(input_shape=(*img_size, 3))
    history = train_model(model, train_gen, val_gen, epochs=epochs)
    
    # 3. 评估模型
    print("Evaluating model...")
    results = evaluate_model(model, test_gen)
    print(f"Test Loss: {results['loss']:.4f}, Test Accuracy: {results['accuracy']:.4f}")
    
    # 4. 获取预测概率
    print("Getting predictions...")
    y_probs, y_true = predict_proba(model, test_gen)
    y_pred = np.argmax(y_probs, axis=1)
    
    # 5. 可视化结果
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred, classes=["COVID", "NonCOVID"])
    plot_roc_curve(y_true, y_probs, class_names=["COVID", "NonCOVID"])
    print_classification_report(y_true, y_pred, class_names=["COVID", "NonCOVID"])
    
    # 6. MRF优化
    print("Preparing data for MRF optimization...")
    feature_extractor = tf.keras.Model(
    inputs=model.inputs, 
    outputs=model.layers[-2].output
    )

    test_features = MRFOptimizer.collect_features(test_gen, feature_extractor)

    # 确保样本数量一致
    assert len(test_features) == len(y_true), "特征与标签数量不匹配"

    print("Building adjacency matrix...")
    adjacency_matrix = MRFOptimizer.build_adjacency_matrix(test_features, k=10,sigma=np.median(np.std(test_features, axis=0)))

    print("Running MRF optimization...")
    mrf_optimizer = MRFOptimizer(adjacency_matrix, beta=3.0,max_iter=20)
    optimized_labels = mrf_optimizer.optimize(y_probs)
    
    # 评估优化后的结果
    print("\nResults after MRF optimization:")
    print_classification_report(y_true, optimized_labels, class_names=["COVID", "NonCOVID"])
    plot_confusion_matrix(y_true, optimized_labels, classes = ["COVID", "NonCOVID"])
    
    
    # 添加优化后的ROC曲线
    # 将优化后的标签转换为概率形式 (0或1)
    optimized_probs = np.zeros_like(y_probs)
    optimized_probs[np.arange(len(optimized_labels)), optimized_labels] = 1
    plot_roc_curve(y_true, optimized_probs, class_names=["COVID", "NonCOVID"])
if __name__ == "__main__":
    main()