import numpy as np
from sklearn.neighbors import NearestNeighbors

class MRFOptimizer:
    """
    马尔可夫随机场优化器，用于后处理分类结果
    """
    def __init__(self, adjacency_matrix, beta=1.0, max_iter=10):
        """
        初始化MRF优化器
        
        参数:
            adjacency_matrix: 邻接矩阵，表示样本之间的关系
            beta: MRF参数，控制平滑程度
            max_iter: 最大迭代次数
        """
        self.adjacency = adjacency_matrix
        self.beta = beta
        self.max_iter = max_iter
    def _compute_energy(self, labels, probabilities):
        """
        计算当前标签配置的能量
        
        参数:
            labels: 当前标签
            probabilities: 原始分类概率
            
        返回:
            总能量
        """
        probabilities=np.clip(probabilities,1e-10,1.0)
        # 一元项能量（与原始分类结果的差异）
        unary_energy = -np.sum(np.log(probabilities[np.arange(len(labels)), labels.astype(int)]))
        
        # 二元项能量（相邻样本标签不一致的惩罚）
        pairwise_energy = 0
        for i in range(len(labels)):
            for j in range(i+1,len(labels)):
                if self.adjacency[i, j] > 0 and labels[i] != labels[j]:
                    pairwise_energy += self.adjacency[i, j]
        
        return unary_energy + self.beta * pairwise_energy
    
    def optimize(self, initial_probs):
        """
        执行MRF优化
        
        参数:
            initial_probs: 初始分类概率
            
        返回:
            优化后的标签
        """
        n_samples, n_classes = initial_probs.shape
        current_labels = np.argmax(initial_probs, axis=1)
        current_energy = self._compute_energy(current_labels, initial_probs)
        
        for _ in range(self.max_iter):
            improved = False
            
            # 随机顺序访问样本
            order = np.random.permutation(n_samples)
            for i in order:
                original_label = current_labels[i]
                min_energy = float('inf')
                best_label = original_label
                
                # 尝试所有可能的标签
                for candidate in range(n_classes):
                    if candidate == original_label:
                        continue
                    
                    temp_labels = current_labels.copy()
                    temp_labels[i] = candidate
                    temp_energy = self._compute_energy(temp_labels, initial_probs)
                    
                    if temp_energy < min_energy:
                        min_energy = temp_energy
                        best_label = candidate
                
                # 如果找到更好的标签，则更新
                if best_label != original_label:
                    current_labels[i] = best_label
                    improved = True
            
            # 如果没有改进，提前终止
            if not improved:
                break
        
        return current_labels
    
    @staticmethod
    # 方法3的推荐实现
    def collect_features(generator, model):
        generator.reset()
        features = []
        for _ in range(len(generator)):
            x, _ = generator.next()
            batch_features = model.predict(x, verbose=0)
            if batch_features.ndim > 2:
                batch_features = batch_features.reshape(len(x), -1)
            features.append(batch_features)
        return np.concatenate(features, axis=0)
    def build_adjacency_matrix(features, k=10,sigma=None):
        """
        构建邻接矩阵
        
        参数:
            features: 样本特征
            k: 每个样本的邻居数量
            
        返回:
            邻接矩阵
        """
        n_samples = features.shape[0]
        adjacency = np.zeros((n_samples, n_samples))
        
        # 使用k近邻构建邻接关系
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(features)
        distances, indices = nbrs.kneighbors(features)
        if sigma is None:
            sigma = np.mean(distances)
        for i in range(n_samples):
            for j, idx in enumerate(indices[i][1:]):  # 跳过自己
                # 使用高斯核函数计算权重
                weight = np.exp(-distances[i][j+1] ** 2 / (2 * np.mean(distances) ** 2))
                adjacency[i, idx] = weight
                adjacency[idx, i] = weight  # 对称矩阵
        
        return adjacency