import json
import numpy as np
import matplotlib.pyplot as plt

# 加载脚本生成的JSON报告（CER总数据）
report_path = "../测评结果7-调整模型参数4.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

# 1. 提取全量样本的CER列表（核心：这就是你的CER总数据）
all_cer = [sample["cer"] for sample in report["all_samples_results"]]

# 2. 基础统计分析
print("=== CER全量数据统计 ===")
print(f"评测样本总数：{len(all_cer)}")
print(f"平均CER：{np.mean(all_cer):.4f}")
print(f"中位数CER：{np.median(all_cer):.4f}")
print(f"最小CER：{np.min(all_cer):.4f}")
print(f"最大CER：{np.max(all_cer):.4f}")
print(f"CER标准差：{np.std(all_cer):.4f}")

# 3. 按CER区间统计（分析不同错误率的样本分布）
cer_ranges = {
    "0.0-0.1 (优)": len([c for c in all_cer if 0.0 <= c <= 0.1]),
    "0.1-0.3 (良)": len([c for c in all_cer if 0.1 < c <= 0.3]),
    "0.3-0.5 (中)": len([c for c in all_cer if 0.3 < c <= 0.5]),
    "0.5-1.0 (差)": len([c for c in all_cer if 0.5 < c <= 1.0])
}
print("\n=== CER区间分布 ===")
for range_name, count in cer_ranges.items():
    print(f"{range_name}：{count} 条（占比 {count/len(all_cer)*100:.2f}%）")

# 4. 可视化CER分布（可选，直观查看数据分布）
plt.figure(figsize=(10, 6))
# 只有放这里不报错
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.hist(all_cer, bins=20, color="#4285F4", edgecolor="black", alpha=0.7)
plt.axvline(np.mean(all_cer), color="red", linestyle="--", label=f"平均CER: {np.mean(all_cer):.4f}")
plt.axvline(np.median(all_cer), color="orange", linestyle="--", label=f"中位数CER: {np.median(all_cer):.4f}")
plt.xlabel("CER（字符错误率）")
plt.ylabel("样本数量")
plt.title("SenseVoice四川方言CER分布")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.savefig("cer_distribution.png", dpi=300, bbox_inches="tight")
plt.show()