import csv  # 導入csv模組，用於處理CSV檔案的讀寫操作

# 準確的用電量數據，每個子列表包含：
# [日期, 7點用電量, 8點用電量, 9點用電量, 10點用電量, 11點用電量, 12點用電量, 用電類型]
data = [
    # 第1-4天 (類型1的用電模式)
    [1, 2.3976, 1.5328, 1.9044, 1.1937, 2.4184, 1.8649, 1],
    [2, 2.3936, 1.4804, 1.9907, 1.2732, 2.2719, 1.8110, 1],
    [3, 2.2880, 1.4585, 1.9867, 1.2451, 2.3389, 1.8099, 1],
    [4, 2.2904, 1.4766, 1.8876, 1.2706, 2.2966, 1.7744, 1],
    # 第5-8天 (類型2的用電模式)
    [5, 1.1201, 0.0587, 1.3154, 5.3783, 3.1849, 2.4276, 2],
    # ... 後續數據省略 ...
]

# 開啟CSV檔案進行寫入，使用 'w' 模式表示覆寫，newline='' 避免產生空行
with open('power_consumption.csv', 'w', newline='') as f:
    writer = csv.writer(f)  # 創建CSV寫入器物件
    
    # 寫入CSV檔案的表頭行
    writer.writerow(['Day', '7:00', '8:00', '9:00', '10:00', '11:00', '12:00', 'Class'])
    
    # 逐行寫入數據
    for row in data:
        writer.writerow(row)

# 輸出提示訊息，告知檔案已生成
print("數據文件已生成：power_consumption.csv")

# 顯示數據預覽
print("\n數據預覽：")
# 輸出表頭
print("Day  7:00   8:00   9:00   10:00  11:00  12:00  Class")

# 顯示前3行數據
# row[0]:<4 表示靠左對齊，寬度為4
# row[1]至row[6]:<6 表示靠左對齊，寬度為6
for row in data[:3]:  
    print(f"{row[0]:<4} {row[1]:<6} {row[2]:<6} {row[3]:<6} {row[4]:<6} {row[5]:<6} {row[6]:<6} {row[7]}")
