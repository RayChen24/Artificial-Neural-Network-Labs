import pandas as pd #導入pandas 函式庫
import numpy as np #導入numpy 函式庫
import matplotlib.pyplot as plt #導入matplotlib函式庫

class LVQ:
    def __init__(self, num_prototypes, learning_rate, epochs):
        self.num_prototypes = num_prototypes  # 原型數量
        self.learning_rate = learning_rate      # 學習率
        self.epochs = epochs                      # 訓練輪次
        self.prototypes = None                    # 原型權重
        self.classes = None                       # 類別標籤
        self.error_rate_history = []              # 用於記錄每一輪的錯誤率

    def fit(self, X, y):
        # 隨機選擇原型權重
        indices = np.random.choice(len(X), self.num_prototypes, replace=False)
        self.prototypes = X[indices]
        self.classes = y[indices]

        # 訓練過程
        for epoch in range(self.epochs):
            total_loss = 0  # 初始化每一輪的總損失
            correct_predictions = 0  # 計算正確預測數量

            for i in range(len(X)):
                distances = np.linalg.norm(X[i] - self.prototypes, axis=1)  # 計算距離
                winner_index = np.argmin(distances)  # 找到最近的原型

                # 更新原型權重
                if y[i] == self.classes[winner_index]:
                    self.prototypes[winner_index] += self.learning_rate * (X[i] - self.prototypes[winner_index])
                    correct_predictions += 1  # 計算正確預測
                else:
                    self.prototypes[winner_index] -= self.learning_rate * (X[i] - self.prototypes[winner_index])

                # 計算損失（這裡使用距離作為損失）
                total_loss += distances[winner_index]

            # 計算錯誤率並記錄
            error_rate = 1 - (correct_predictions / len(X))
            self.error_rate_history.append(error_rate)

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            distances = np.linalg.norm(X[i] - self.prototypes, axis=1)
            winner_index = np.argmin(distances)
            predictions.append(self.classes[winner_index])
        return np.array(predictions)

def main():
    # 讀取訓練和測試數據
    train_data = pd.read_csv("power_consumption.csv")
    test_data = pd.read_csv("testing_data.csv")

    # 準備訓練數據
    X_train = train_data.iloc[:, 1:-1].values  # 去掉第一列（Day）和最後一列（Class）
    y_train = train_data['Class'].values

    # 準備測試數據
    X_test = test_data.iloc[:, 1:-1].values  # 去掉第一列（Day）和最後一列（Class）
    y_test = test_data['Class'].fillna(-1).values  # 用-1替代NaN值

    # 設定不同的學習率進行實驗
    learning_rates = [0.01, 0.05, 0.1]
    num_prototypes = 4  # 根據類別數量設定原型數量
    epochs = 100

    plt.figure(figsize=(10, 5))  # 繪圖設置

    for lr in learning_rates:
        print(f"Training with Learning Rate: {lr}")
        
        lvq_model = LVQ(num_prototypes, lr, epochs)
        lvq_model.fit(X_train, y_train)

        # 預測測試數據的類別
        predictions = lvq_model.predict(X_test)

        # 顯示預測結果
        for i in range(len(predictions)):
            print(f'Test Sample {i+1}: Predicted Class: {predictions[i]}, Actual Class: {y_test[i]}')

        # 繪製收斂曲線（錯誤率）
        plt.plot(lvq_model.error_rate_history, label=f'Learning Rate: {lr}')
    
    plt.title('Convergence Curve (Error Rate)')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid()
    
    plt.savefig('convergence_curve.png')  # 保存收斂曲線圖為PNG文件
    plt.show()                             # 顯示收斂曲線圖

if __name__ == "__main__":
    main()
