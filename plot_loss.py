import json
import matplotlib.pyplot as plt

def plot_log_metrics(log_file, target_metrics, img_path="loss_curve.png"):
    """
    從 log 檔案讀取數據並畫出指定的 metrics 曲線。
    
    :param log_file: log 檔案的路徑 (例如 'log.txt')
    :param target_metrics: 想要畫出的 loss 名稱列表
    """
    data = []
    
    # 1. 讀取並解析 log 檔案
    with open(log_file, 'r') as f:
        for line in f:
            try:
                # 處理可能夾雜在內容中的來源標籤 
                clean_line = line.split(']', 1)[-1] if ']' in line else line
                data.append(json.loads(clean_line))
            except json.JSONDecodeError:
                continue

    # 2. 提取 Epoch 和對應的 Metric 數值
    epochs = [d['epoch'] for d in data]
    
    plt.figure(figsize=(10, 6))
    
    for metric in target_metrics:
        if metric in data[0]:
            values = [d[metric] for d in data]
            plt.plot(epochs, values, marker='o', label=metric)
        else:
            print(f"Warning: Metric '{metric}' not found in log.")

    # 3. 圖表設定
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics Curve')
    plt.legend()
    plt.grid(True)
    
    # 4. 儲存圖片
    plt.savefig(img_path)
    print(f"圖表已儲存至: {img_path}")

# --- 使用範例 ---
if __name__ == "__main__":
    loss_txt = "/mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage2v4_mode3/log.txt"
    save_path = "/mnt/HDD4/ricky/feedforward/amb3r/outputs/exp_stage2v4_mode3/loss_curve.png"
    
    metrics_to_plot = [
        "train_loss_ins_contrast", 
        "train_loss_instid_cross_ins_det", 
        "train_loss_instid_push_ins_det"
    ]
    
    plot_log_metrics(  loss_txt, metrics_to_plot, img_path=save_path)