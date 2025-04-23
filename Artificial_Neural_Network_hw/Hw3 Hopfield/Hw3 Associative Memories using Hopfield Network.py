# 導入必要的函式庫
import numpy as np  # 用於數值計算
import matplotlib.pyplot as plt  # matplotlib用於繪圖顯示

# 定義激activation函數，使用雙曲正切函數
# beta 參數控制函數的陡峭度，值越大，函數越接近離散的階躍函數
def activation_function(x, beta=100):
    return np.tanh(beta*x)

# Hopfield 網路class定義
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size  # 網路大小（輸入模式的維度）
        self.weights = np.zeros((size, size))  # 初始化權重矩陣為零矩陣

    # 訓練方法：使用 Hebbian 學習規則
    def train(self, patterns):
        for pattern in patterns:
            # 使用外積更新weights矩陣
            self.weights += np.outer(pattern, pattern)
            # 將權重矩陣的對角線元素設為 0，避免自己連接
            np.fill_diagonal(self.weights, 0)

    # 回想方法：從輸入模式中恢復儲存的模式
    def recall(self, pattern, beta=100, steps=10):
        for _ in range(steps):
            # 反覆更新模式，直到收斂或達到最大步數
            pattern = activation_function(self.weights @ pattern, beta)
        return np.sign(pattern)  # 返回二值化結果（+1 或 -1）

# 定義阿拉伯數字 1-4 的pattern（每個數字為 9x5 的矩陣45 bits）
# 數字1的pattern
pattern_1 = np.array([
    [1, 1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1]
])

# 數字2的pattern
pattern_2 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [1, 1, 1, -1, -1],
    [1, 1, 1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 1, 1, 1],
    [-1, -1, 1, 1, 1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
])

# 數字3的pattern
pattern_3 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [1, 1, 1, -1, -1],
    [1, 1, 1, -1, -1],
    [-1, -1, -1, -1, -1],
    [1, 1, 1, -1, -1],
    [1, 1, 1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
])

# 數字4的pattern
pattern_4 = np.array([
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [1, 1, 1, -1, -1],
    [1, 1, 1, -1, -1],
    [1, 1, 1, -1, -1],
    [1, 1, 1, -1, -1]
])

# 將 2D 模式轉換為 1D 向量（展平）
pattern_1_flat = pattern_1.flatten()
pattern_2_flat = pattern_2.flatten()
pattern_3_flat = pattern_3.flatten()
pattern_4_flat = pattern_4.flatten()

patterns = [pattern_1_flat, pattern_2_flat, pattern_3_flat, pattern_4_flat]

# 定義用於測試的隨機種子
random_seeds = [42, 123, 456]
# 儲存所有結果的list
all_corrupted_patterns = []
all_recovered_patterns = []

# 初始化 Hopfield 網路（輸入大小為 45 = 9x5）
network = HopfieldNetwork(size=45)
# 使用所有上面建立的pattern訓練網路
network.train(patterns)

# 定義添加雜訊的function
def corrupt_pattern(pattern, corruption_level=0.2):# corruption_level是雜訊等級
    corrupted = np.copy(pattern)  # 複製原始模式
    # 計算需要翻轉的位元數量
    num_corrupt = int(len(pattern) * corruption_level)
    # 隨機選擇位置
    indices = np.random.choice(len(pattern), num_corrupt, replace=False)
    corrupted[indices] *= -1  # 翻轉選中的位元
    return corrupted

# 產生並恢復corrupted patterns
corrupted_patterns = [corrupt_pattern(p, corruption_level=0.2) for p in patterns]
recovered_patterns = [network.recall(p) for p in corrupted_patterns]

# 定義評估恢復效果的function
def evaluate_recovery(original, recovered):# original是原始pattern，recovered是恢復的pattern
    accuracy = np.mean(original == recovered)  # 計算匹配的比例
    return accuracy * 100  # return百分比

# 測試不同雜訊等級的function
def test_noise_levels(pattern, noise_levels=[0.2, 0.4, 0.6, 0.8]):#0.2,0.4,0.6,0.8代表不同的雜訊等級
    results = []# 儲存結果的list
    for level in noise_levels:#遍歷不同的noise_levels
        corrupted = corrupt_pattern(pattern, corruption_level=level)# 添加雜訊
        recovered = network.recall(corrupted)# 恢復
        results.append((level, corrupted, recovered))# 儲存結果
    return results

# 定義繪圖函數，用於顯示原始、受損和恢復的patterns
def plot_images(original, corrupted, recovered, titles):
    # 檢查圖像數量以確定 axes 的結構
    num_patterns = len(original)# 獲取圖像數量
    fig, axes = plt.subplots(3, num_patterns, figsize=(2*num_patterns, 4))  # 動態調整圖表的大小
    
    # 如果只有一個圖像，axes 會是一維陣列
    if num_patterns == 1:
        axes = np.array([axes]).T  # 確保 axes 結構統一為二維陣列

    # 設置背景顏色為白色
    fig.patch.set_facecolor('white')

    for i in range(num_patterns):# 遍歷每個pattern
        # 顯示三種圖像（原始、失真、恢復）
        for j, img in enumerate([original[i], corrupted[i], recovered[i]]):# 遍歷每種圖像(原始的,受損的,恢復的)
            ax = axes[j, i] if num_patterns > 1 else axes[j, 0]  # 正確索引
            ax.imshow(img.reshape(9, 5), cmap='gray', vmin=-1, vmax=1, aspect='equal')# 顯示圖像
            ax.set_title(f"{['Original', 'Corrupted', 'Recovered'][j]} {titles[i]}")#設定標題
            
            # 添加網格
            ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.5)
            
            # 設置刻度
            ax.set_xticks(np.arange(-0.5, 5, 1))
            ax.set_yticks(np.arange(-0.5, 9, 1))
            
            # 移除刻度標籤但保留網格線
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # 保持軸的顯示
            ax.axis('on')

    plt.tight_layout()# 調整子圖的間距
    plt.show()# 顯示圖像

# 定義繪製雜訊實驗結果的函數
def plot_noise_experiment(pattern, results):# pattern是原始pattern，results是不同雜訊等級的結果
    """顯示不同噪聲等級的執行結果"""
    fig, axes = plt.subplots(len(results), 3, figsize=(8, 1*len(results)))# 動態調整圖表的大小
    for i, (level, corrupted, recovered) in enumerate(results):# 遍歷每個結果
        accuracy = evaluate_recovery(pattern, recovered)# 評估recover效果
        axes[i, 0].imshow(pattern.reshape(9, 5), cmap='gray', vmin=-1, vmax=1)# 顯示原始圖像
        axes[i, 0].set_title(f'Original')# 設定original標題
        axes[i, 1].imshow(corrupted.reshape(9, 5), cmap='gray', vmin=-1, vmax=1)# 顯示corrupted圖像
        axes[i, 1].set_title(f'Corrupted (noise={level:.1f})')# 設定corrupted標題
        axes[i, 2].imshow(recovered.reshape(9, 5), cmap='gray', vmin=-1, vmax=1)# 顯示recovered圖像
        axes[i, 2].set_title(f'Recovered (acc={accuracy:.1f}%)')# 設定recovered標題
        
        # 設定網格和刻度
        for ax in [axes[i, 0], axes[i, 1], axes[i, 2]]:# 遍歷每個axes
            ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.5)# 添加網格
            ax.set_xticks(np.arange(-0.5, 5, 1))# 設定X刻度
            ax.set_yticks(np.arange(-0.5, 9, 1))# 設定Y刻度
            ax.set_xticklabels([])# 移除X刻度標籤
            ax.set_yticklabels([])# 移除Y刻度標籤
            ax.axis('on')# 保持軸的顯示
    
    plt.tight_layout()
    plt.show()

# 執行主要實驗並顯示結果
titles = ['1', '2', '3', '4']
plot_images([pattern_1, pattern_2, pattern_3, pattern_4],
           [p.reshape(9, 5) for p in corrupted_patterns],
           [p.reshape(9, 5) for p in recovered_patterns],
           titles)

# 使用不同隨機種子進行實驗
print("\n=== 使用不同隨機種子的測試 ===")
for seed in random_seeds:# 遍歷每個隨機種子
    print(f"\n使用隨機種子 {seed}")
    np.random.seed(seed)  # 設置隨機種子以確保結果可重現
    for i, pattern in enumerate(patterns):# 遍歷每個patterns
        # 執行實驗並顯示結果
        corrupted = corrupt_pattern(pattern)# 添加雜訊
        recovered = network.recall(corrupted)# 恢復
        accuracy = evaluate_recovery(pattern.flatten(), recovered)# 評估recover效果
        print(f"Pattern {i+1} 的恢復準確率: {accuracy:.2f}%")
        # 儲存結果並繪圖
        all_corrupted_patterns.append(corrupted)# 儲存corrupted pattern
        all_recovered_patterns.append(recovered)# 儲存recovered pattern
        plot_images([pattern], [corrupted.reshape(9, 5)],# 繪製圖像
                   [recovered.reshape(9, 5)], [f'Pattern {i+1}, Seed {seed}'])

# 測試不同雜訊等級對每個模式的影響
print("\n=== 測試不同雜訊等級的影響 ===")
for i, pattern in enumerate(patterns):# 遍歷每個patterns
    print(f"\n測試 Pattern {i+1}")
    results = test_noise_levels(pattern)# 測試不同雜訊等級
    plot_noise_experiment(pattern, results)# 繪製圖像