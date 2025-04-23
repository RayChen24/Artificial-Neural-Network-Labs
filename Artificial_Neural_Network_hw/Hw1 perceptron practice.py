import numpy as np #導入numpy函式庫，進行數值運算
import matplotlib.pyplot as plt#導入matplotlib函式庫，用於繪圖

class Perceptron: #定義名為perceptron 的類別
    def __init__(self, learning_rate=0.1, n_iterations=80, random_state=1):#初始化感知器、設定學習率、迭代次數和隨機的狀態
        self.learning_rate = learning_rate#每次更新權重的步伐大小
        self.n_iterations = n_iterations#最大迭代次數
        self.random_state = random_state#用來初始化隨機生成器，使結果可以重現

    def fit(self, X, y):#定義fit 這個方法來訓練感知器(X是input的特徵集，y是對應的標籤數據)
        gen = np.random.RandomState(self.random_state)#初始化隨機生成器
        self.w_ = gen.uniform(low=0.0, high=0.0, size=X.shape[1])  # 隨機初始化權重向量w
        self.b_ = gen.uniform(low=-1.0, high=-1.0)  # 隨機初始化偏差b(即為權重的常數項)
        self.n_misclassifications_ = []#用來存放每次迭代中的錯誤分類數

        for epoch in range(self.n_iterations):#進行多次的迭代(max:n_iterations)
            n_misclassifications = 0#紀錄每次迭代中的錯誤分類數
            for xi, yi in zip(X, y):#對每個樣訓練樣本進行訓練
                yi_hat = self.predict(xi)#用當前的權重w和偏差b來預測樣本xi的類別
                move = self.learning_rate * (yi - yi_hat)#計算預測錯誤，根據差異調整權重
                self.w_ += move * xi#更新權重w
                self.b_ += move#更新誤差b
                n_misclassifications += int(np.abs(move) > 0)#若有錯誤發生，增加錯誤的次數
            self.n_misclassifications_.append(n_misclassifications)#紀錄每次迭代的錯誤分類
            # 如果模型提前收斂，後續的錯誤記錄為0，直到滿足80次
            if n_misclassifications == 0:#如果分類沒有錯誤，模型已收斂，則停止訓練
                while len(self.n_misclassifications_) < self.n_iterations:
                    self.n_misclassifications_.append(0)#補足後續的0直到達到n_iterations次數
                break
        return self#返回模型

    def predict(self, x):#定義預測函式，基於當前的權重w偏重b來判斷x屬於哪個類別
        return np.where(np.dot(x, self.w_) + self.b_ >= 0., 1, -1)#np.dot(x,self.w_)計算X合的內積，如果內積結果加上偏差b大於等於0，則預測為1，否則為-1

# 設置訓練集
X = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.3, -0.5], [-0.1, 0.1]])#設定訓練數據的特徵集(四個二維向量)
y = np.array([1, 1, -1, -1])#設定訓練數據的標籤集

# 創建和訓練感知器
perceptron = Perceptron(learning_rate=0.1, n_iterations=80)#創建感知器對象，設定學習率以及最大迭代次數(訓練輪數)
perceptron.fit(X, y)#訓練感知器，將X和y傳入

# 繪製分類圖
plt.figure(figsize=(10, 5))#創建圖片並設定大小
plt.subplot(1, 2, 1)#在視窗中化分子區域(第一行第二列的第一個)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')#繪製訓練樣本點，使用不同顏色代表不同類別
plt.xlabel('P(1)')
plt.ylabel('P(2)')
plt.title('Classification')#設定子圖的標題

# 繪製決策邊界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1#設定x軸的標題
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1#設定y軸的標題
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))#使用網格生成decision boundary的座標點
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])#使用感知器的權重預測每個網格點的類別
Z = Z.reshape(xx.shape)#重新塑型，使其與網格座標形狀一致
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')#使用等高線圖繪製decision bondary的區域

# 繪製錯誤收斂圖
plt.subplot(1, 2, 2)#在圖中劃分另一個子圖的區域
plt.plot(range(1, len(perceptron.n_misclassifications_) + 1), 
         perceptron.n_misclassifications_, marker='o')# 繪製錯誤隨迭代次數的變化
plt.axhline(y=0, color='r', linestyle='--')  # 在 y=0 處繪製一條紅色虛線，代表收斂到 0 錯誤

# 調整 Y 軸範圍，讓 0 在中間
max_misclassifications = max(perceptron.n_misclassifications_) # 找出錯誤分類數的最大值
plt.ylim(-max_misclassifications, max_misclassifications) # 設定 Y 軸範圍，使 0 位於中間(對稱)
plt.xlabel('Iterations')# 設定 x 軸標籤
plt.ylabel('Error')#設定y軸標籤名稱
plt.title('Convergence Curve')#設定子圖的標題

plt.tight_layout() # 自動調整子圖間距，以防止重疊
plt.show()#顯示圖形

# print出最終的權重及偏差
print("Final weights (w_):", perceptron.w_)
print("Final bias (b_):", perceptron.b_)