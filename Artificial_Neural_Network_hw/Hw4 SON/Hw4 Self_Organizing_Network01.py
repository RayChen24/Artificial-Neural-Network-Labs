import numpy as np #導入numpy函式庫用於數值計算
import matplotlib.pyplot as plt #導入matplotlib.pyplot函式庫用於繪圖顯示
import os #導入os函式庫用於系統操作
from sklearn.cluster import KMeans #導入KMeans函式庫用於K-Means分群

class SelfOrganizingMap: #定義SOM這個類別
    def __init__(self, input_data, x_mesh=10, y_mesh=10, learning_rate=0.1, max_iterations=1000):#初始化函數
        self.input_data = input_data #輸入數據
        self.x_mesh = x_mesh #設定網格的x軸大小
        self.y_mesh = y_mesh #設定網格的y軸大小
        self.learning_rate = learning_rate #設定學習率
        self.max_iterations = max_iterations #設定最大的迭帶次數
        
        self.weights = self._initialize_weights() #初始化weights
        self.convergence_curve = []#初始化收斂曲線
    
    def _initialize_weights(self): #定義初始化wights的函數
        min_data = np.min(self.input_data, axis=0) #計算input data的最小值
        max_data = np.max(self.input_data, axis=0) #計算input data的最大值
        
        weights = np.random.uniform(#初始化weights為隨機值
            low=min_data, #weights的最小值
            high=max_data, #weights的最大值
            size=(self.x_mesh, self.y_mesh, self.input_data.shape[1])#weights的形狀為(x_mesh, y_mesh,input data 的特徵數)
        )
        return weights#返回初始化的 weights
    
    def _find_best_matching_unit(self, input_vector):
        # 找到與輸入向量最相似的神經元（最佳匹配單元）
        distances = np.sum((self.weights - input_vector)**2, axis=2)  # 計算每個神經元與輸入向量的距離
        return np.unravel_index(np.argmin(distances), distances.shape)  # 返回距離最小的神經元坐標

    def _gaussian_neighborhood(self, distance, sigma):
        # 使用高斯函數計算鄰域效應，用於調整權重更新
        return np.exp(-distance**2 / (2 * sigma**2))

    def train(self):
        # SOM神經網絡的訓練過程
        initial_sigma = max(self.x_mesh, self.y_mesh) / 2  # 初始鄰域半徑
        
        for t in range(self.max_iterations):
            # 動態調整學習參數
            sigma = initial_sigma * np.exp(-t / self.max_iterations)  # 逐漸縮小鄰域
            current_learning_rate = self.learning_rate * np.exp(-t / self.max_iterations)  # 逐漸減小學習率
            
            # 隨機選擇一個輸入向量
            input_vector = self.input_data[np.random.randint(len(self.input_data))]
            
            # 找到最佳匹配單元
            bmu_x, bmu_y = self._find_best_matching_unit(input_vector)
            
            # 更新整個網格的權重
            for x in range(self.x_mesh):
                for y in range(self.y_mesh):
                    # 計算當前神經元到最佳匹配單元的距離
                    distance = np.sqrt((x - bmu_x)**2 + (y - bmu_y)**2)
                    
                    # 計算高斯鄰域影響
                    influence = self._gaussian_neighborhood(distance, sigma)
                    
                    # 更新權重
                    self.weights[x, y] += current_learning_rate * influence * (input_vector - self.weights[x, y])
            
            # 每10次迭代記錄一次總誤差
            if t % 10 == 0:
                total_error = np.mean(np.min(np.sum((self.input_data[:, np.newaxis, np.newaxis, :] - self.weights)**2, axis=3), axis=(1,2)))
                self.convergence_curve.append(total_error)

    def plot_results(self, input_data, dataset_name, mesh_size, save_path=None):
        # 繪製SOM算法的結果可視化
        plt.figure(figsize=(12, 3))  # 創建一個寬度為12，高度為3的圖形
        
        # 原始數據散點圖
        plt.subplot(131)
        plt.scatter(input_data[:, 0], input_data[:, 1], c='blue', alpha=0.5)
        plt.title(f'{dataset_name} - Original Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # SOM權重網格可視化
        plt.subplot(132)
        plt.scatter(input_data[:, 0], input_data[:, 1], c='blue', alpha=0.3)
        # 繪製權重網格的連接線
        for x in range(self.x_mesh):
            for y in range(self.y_mesh):
                if x < self.x_mesh - 1:
                    plt.plot([self.weights[x, y, 0], self.weights[x+1, y, 0]],
                             [self.weights[x, y, 1], self.weights[x+1, y, 1]], 'r-')
                if y < self.y_mesh - 1:
                    plt.plot([self.weights[x, y, 0], self.weights[x, y+1, 0]],
                             [self.weights[x, y, 1], self.weights[x, y+1, 1]], 'r-')
        plt.title(f'{dataset_name} - SOM Weight Mesh ({mesh_size})')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # 收斂曲線
        plt.subplot(133)
        plt.plot(self.convergence_curve)
        plt.title(f'{dataset_name} - SOM Convergence Curve')
        plt.xlabel('Iteration (×10)')
        plt.ylabel('Mapping Error')
        
        plt.tight_layout()  # 自動調整子圖間距
        if save_path:
            plt.savefig(save_path)  # 如果提供了保存路徑，則保存圖形
        plt.show()  # 顯示圖形

def plot_Kmeans(data, dataset_name, save_path=None, n_clusters=None):
    # K-Means聚類結果的繪製函數
    # 根據數據集自動設置聚類數量
    if n_clusters is None:
        if dataset_name == 'ThreeGroups':
            n_clusters = 3
        elif dataset_name in ['TwoCircles', 'TwoRings']:
            n_clusters = 2
        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}')
    
    # 執行K-Means算法
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_  # 獲取每個數據點的聚類標籤
    centers = kmeans.cluster_centers_  # 獲取聚類中心

    # 繪製K-Means聚類結果
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)  # 數據點著色
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')  # 繪製聚類中心
    plt.title(f'{dataset_name} - K-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if save_path:
        plt.savefig(save_path)  # 如果提供了保存路徑，則保存圖形
    plt.show()

def load_data(filepath):
    # 從文本文件載入數據
    return np.loadtxt(filepath)

def print_final_weights(self):
    # 打印最終的網絡權重（用於調試）
    print("Final Weights:")
    print(self.weights)

def main():
    # 主函數，執行整個數據處理和分析流程
    base_path = r'C:\Users\User\.vscode'  # 設置基本文件路徑
    datasets = [
        'ThreeGroups.txt', 
        'TwoCircles.txt', 
        'TwoRings.txt'
    ]  # 要處理的數據集列表
    mesh_sizes = [(10, 10), (5, 5), (15, 15)]  # 不同大小的SOM網格
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # 載入數據
        filepath = os.path.join(base_path, dataset)
        data = load_data(filepath)
        
        for x_mesh, y_mesh in mesh_sizes:
            print(f"Mesh size: {x_mesh} x {y_mesh}")
            
            # 創建並訓練SOM
            som = SelfOrganizingMap(
                input_data=data, 
                x_mesh=x_mesh, 
                y_mesh=y_mesh, 
                learning_rate=0.1, 
                max_iterations=1000
            )
            som.train()
            print_final_weights(som)

            # 保存SOM結果圖
            som_save_path = os.path.join(base_path, f'som_result_{dataset.split(".")[0]}_{x_mesh}x{y_mesh}.png')
            som.plot_results(
                input_data=data, 
                dataset_name=dataset.split(".")[0],
                mesh_size=f'{x_mesh}x{y_mesh}',
                save_path=som_save_path
            )

            # 保存K-Means結果圖
            kmeans_save_path = os.path.join(base_path, f'kmeans_result_{dataset.split(".")[0]}_{x_mesh}x{y_mesh}.png')
            plot_Kmeans(
                data=data, 
                dataset_name=dataset.split(".")[0], 
                save_path=kmeans_save_path
            )

if __name__ == '__main__':
    main()  # 如果是直接運行此腳本，則執行main函數