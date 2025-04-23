import pandas as pd #導入pandas函式庫
import numpy as np #導入numpy函式庫
import matplotlib.pyplot as plt #導入 matplotlib函式庫

class ART1:
    def __init__(self, num_neurons, vigilance):
        # 初始化 ART1 網路，設定神經元數量和警戒參數
        self.num_neurons = num_neurons  # 設定神經元數量
        self.vigilance = vigilance      # 設定警戒參數（用於判斷模式相似度）
        self.weights = np.zeros((num_neurons, 16))  # 創建權重矩陣，每個神經元有16個特徵
        self.weight_updates = []  # 創建列表來記錄權重更新的歷史

    def train(self, input_data):
        # 訓練函數：遍歷所有輸入數據
        for x in input_data:
            self.classify(x)

    def classify(self, input_vector):
        # 對輸入向量進行分類
        # 首先將輸入向量標準化S
        input_vector = input_vector / np.linalg.norm(input_vector)
        
        # 計算輸入向量與所有神經元權重的相似度
        similarities = np.dot(self.weights, input_vector)
        
        # 找出最相似的神經元
        best_neuron = np.argmax(similarities)

        # 如果最佳匹配的相似度大於警戒參數
        if similarities[best_neuron] >= self.vigilance * np.linalg.norm(input_vector):
            # 更新獲勝神經元的權重
            previous_weights = self.weights[best_neuron].copy()
            self.weights[best_neuron] += (input_vector - self.weights[best_neuron])
            # 標準化更新後的權重
            self.weights[best_neuron] /= np.linalg.norm(self.weights[best_neuron])
            # 記錄權重更新
            self.weight_updates.append(f'Neuron {best_neuron+1}: Updated from {previous_weights} to {self.weights[best_neuron]}')
        else:
            # 如果相似度不夠高，且還有空閒的神經元，則創建新的類別
            if np.count_nonzero(self.weights) < self.num_neurons:
                new_index = np.count_nonzero(self.weights)
                self.weights[new_index] = input_vector
                self.weight_updates.append(f'Neuron {new_index+1}: Added {input_vector}')

    def plot_weights(self):
        # 繪製權重向量的視覺化圖表
        plt.figure(figsize=(10, 6))
        for i in range(self.num_neurons):
            plt.plot(self.weights[i], marker='o', label=f'Neuron {i+1}')
        
        # 設置圖表標題和軸標籤
        plt.title('Weight Vectors of Neurons')
        plt.xlabel('Features')
        plt.ylabel('Weight Value')
        plt.xticks(ticks=np.arange(16), labels=[f'x{i+1}' for i in range(16)])
        plt.legend()
        plt.grid()
        
        # 保存和顯示圖表
        plt.savefig('weights_plot.png')
        plt.show()

def main():
    # 主函數
    # 讀取訓練數據
    file_path = "C:\\Users\\User\\.vscode\\ART1_training_set.csv"
    data = pd.read_csv(file_path)
    
    # 設定參數
    num_neurons = 5
    vigilance_values = [0.7, 0.8, 0.9]
    
    # 對不同的警戒參數值進行測試
    for vigilance in vigilance_values:
        # 創建 ART1 網路實例
        art1 = ART1(num_neurons, vigilance)
        # 準備訓練數據（除去第一列）
        training_data = data.iloc[:, 1:].values
        # 訓練網路
        art1.train(training_data)
        # 輸出結果
        print(f'Vigilance: {vigilance}')
        for update in art1.weight_updates:
            print(update)
        # 繪製權重圖
        art1.plot_weights()

if __name__ == "__main__":
    main()
