import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams

def setup_ieee_style():
    """
    配置 Matplotlib 以符合 IEEE 格式标准
    """
    # 1. 设置字体为 Times New Roman
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    
    # 2. 启用 MathText (stix 字体风格最接近 Times 的 LaTeX 公式)
    rcParams['mathtext.fontset'] = 'stix' 
    
    # 3. 设置字号 (论文通常要求较小的字号，但在屏幕上需要适当放大)
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 10
    
    # 4. 线条和网格设置
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.5
    rcParams['grid.linestyle'] = '--'
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 6
    
    # 5. 图片保存 DPI (IEEE 要求 300dpi 以上)
    rcParams['figure.dpi'] = 300

def plot_ieee_curves():
    csv_file = 'epoch_logs.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found.")
        return

    # 应用 IEEE 样式
    setup_ieee_style()

    # 1. 读取并处理数据
    df = pd.read_csv(csv_file)
    df['Accuracy'] = df['Accuracy'] * 100 # 转换为百分比
    
    # 获取所有数据集和模型
    datasets = df['Dataset'].unique()
    models = df['Model'].unique()
    
    # 2. 定义固定的样式映射 (保证三张图中同一个模型的颜色和标记一致)
    # IEEE 推荐：同时使用颜色、线型和标记来区分
    markers = ['o', 's', '^', 'D', 'v', 'X'] # 圆、方、上三角、菱形、下三角、叉
    linestyles = ['-', '--', '-.', ':', '-', '--'] # 实线、虚线、点划线...
    
    # 创建映射字典
    model_markers = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    model_styles = {model: linestyles[i % len(linestyles)] for i, model in enumerate(models)}
    
    # 使用 Seaborn 的色盲友好配色 (Deep Palette)
    palette = sns.color_palette("deep", n_colors=len(models))
    model_colors = {model: palette[i] for i, model in enumerate(models)}

    # 3. 循环绘图
    for ds_name in datasets:
        print(f"🎨 Plotting IEEE figure for: {ds_name} ...")
        
        ds_data = df[df['Dataset'] == ds_name]
        
        # 创建画布 (IEEE 双栏论文通常单张图宽 3.5英寸，这里设为 6x4 便于查看)
        plt.figure(figsize=(6, 4.5))
        
        sns.lineplot(
            data=ds_data,
            x='Epoch',
            y='Accuracy',
            hue='Model',
            style='Model',
            palette=model_colors,   # 固定颜色
            markers=model_markers,  # 固定标记
            dashes=False,           # 禁用 seaborn 自动虚线，我们自己控制还是保持实线
            linewidth=1.5,
            markersize=7
        )
        
        # 4. 设置轴标签 (使用 LaTeX 格式)
        plt.xlabel(r'Epoch ($N$)', fontweight='bold')
        plt.ylabel(r'Test Accuracy ($\%$)', fontweight='bold')
        
        # 设置标题 (可选，正式论文中有时不需要标题，直接用 Caption，这里先加上)
        plt.title(f'Performance on {ds_name.upper()}', pad=10)
        
        # 5. 设置 X 轴刻度为整数
        max_epoch = ds_data['Epoch'].max()
        plt.xticks(range(1, int(max_epoch) + 1))
        
        # 6. 优化图例 (去掉了 title，更紧凑)
        # loc='best' 会自动找空白地方放，frameon=True 加个边框
        plt.legend(title=None, loc='lower right', frameon=True, fancybox=False, edgecolor='black')
        
        # 7. 调整布局并保存
        plt.tight_layout()
        
        save_name = f'ieee_chart_{ds_name}.pdf'
        # 同时也保存 PDF 格式 (IEEE 排版通常首选矢量图 PDF/EPS)
        plt.savefig(save_name, bbox_inches='tight')
        plt.savefig(save_name.replace('.png', '.pdf'), bbox_inches='tight')
        
        print(f"✅ Saved: {save_name}")
        plt.close()

if __name__ == '__main__':
    plot_ieee_curves()