#############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# 全局字体设置：确保中文正常显示
plt.rcParams["font.family"] = ["FangSong", "仿宋"]  # 设置默认字体为仿宋
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题
mpl.rcParams['figure.max_open_warning'] = 0  # 关闭图形数量过多警告


def calculate_stage_x_coordinates(stage_count, page_width, margin_ratio=0.1):
    """
    动态计算每个阶段平面的X坐标，实现均匀分布且自适应边距

    参数:
        stage_count (int): 阶段数量（如3个发展时期）
        page_width (float): X轴方向的可用总宽度
        margin_ratio (float): 左右边距占总宽度的比例（默认0.1）

    返回:
        list: 每个阶段的X坐标列表
    """
    # 计算左右边距宽度
    margin = page_width * margin_ratio
    # 计算有效显示宽度（扣除边距后）
    valid_width = page_width - 2 * margin

    # 计算阶段间的间隔（处理只有1个阶段的特殊情况）
    if stage_count > 1:
        interval = valid_width / (stage_count - 1)
    else:
        interval = valid_width

    # 生成每个阶段的X坐标
    stage_x_list = [margin + i * interval for i in range(stage_count)]
    return stage_x_list


# 体育文化发展阶段原始数据（X坐标后续动态计算）
SPORTS_CULTURE_STAGES_RAW = [
    {
        "name": "一二五时期",
        "time": "2011-2015年",
        "color": "lightskyblue",
        "nodes": [
            ["突出体育文化特色", 400], ["体育文化中长期规划", 380],
            ["体育文化从被忽视走向重视", 420], ["体育文化与宏观目标绑定", 360],
            ["强化体育发展基础", 340], ["挖掘体育文化遗产", 280],
            ["建设体育博物馆", 300], ["规划中原武术文化交流中心", 320]
        ]
    },
    {
        "name": "一三五时期",
        "time": "2016-2020年",
        "color": "lightgreen",
        "nodes": [
            ["塑造社会价值观", 440], ["促进社会和谐", 400],
            ["弘扬社会主义核心价值观", 420], ["弘扬民族精神与时代精神", 380],
            ["弘扬中华体育精神", 360], ["未成年人思想道德建设", 340],
            ["独立章节/专项工程规划", 320], ["岭南传统体育文化传承", 300],
            ["民族体育项目推广", 280], ["体育影视与摄影创作", 260],
            ["体育文化创意与设计大赛", 240], ["体育文艺创作扶持", 260],
            ["参与全国体育文化展会", 220], ["全运会特色文化产品打造", 280]
        ]
    },
    {
        "name": "一四五时期",
        "time": "2021-2025年",
        "color": "lightcoral",
        "nodes": [
            ["中华体育精神弘扬", 440], ["弘扬女排精神", 420],
            ["打造辽宁体育铁军", 400], ["体育明星形象塑造工程", 380],
            ["国家文化软实力提升", 400], ["夺金之路专题打造", 320],
            ["海河回响女排精神展", 300], ["五禽戏挖掘传承", 280],
            ["武术与龙舟项目档案整理", 240], ["体育影视作品走出去", 340],
            ["青山绿水全民健身品牌", 360], ["体育文化创作精品工程", 340],
            ["构建全媒体传播格局", 320], ["鼓励体育影视创作", 260]
        ]
    }
]


def main():
    """主函数：绘制体育文化发展3D节点关系图"""
    # 1. 初始化画布（按屏幕分辨率适配）
    screen_dpi = plt.rcParams['figure.dpi']
    screen_width = 1920  # 屏幕宽度（可根据实际调整）
    screen_height = 1080  # 屏幕高度
    fig_size = (screen_width / screen_dpi, screen_height / screen_dpi)

    fig = plt.figure(figsize=fig_size)
    plt.tight_layout(pad=0)  # 紧凑布局，减少边距
    ax = fig.add_subplot(111, projection='3d')  # 创建3D坐标轴

    # 2. 计算阶段X坐标
    stage_count = len(SPORTS_CULTURE_STAGES_RAW)
    page_x_width = 50  # X轴总范围（控制整体缩放）
    stage_x_list = calculate_stage_x_coordinates(stage_count, page_x_width, 0.1)

    # 3. 完善阶段数据（添加X坐标）
    stages = []
    for i in range(stage_count):
        stage = SPORTS_CULTURE_STAGES_RAW[i].copy()
        stage["x"] = stage_x_list[i]
        stages.append(stage)

    # 4. 计算节点布局（使用NetworkX的弹簧布局）
    stage_node_positions = {}
    for stage in stages:
        node_labels = [node[0] for node in stage["nodes"]]
        graph = nx.Graph()
        graph.add_nodes_from(node_labels)

        # 生成节点布局（固定随机种子确保结果一致）
        pos = nx.spring_layout(graph, seed=42, k=1.2, scale=3.0)
        # 调整坐标方向（左下到右上倾斜）
        flipped_pos = {label: (-y, z) for label, (y, z) in pos.items()}
        stage_node_positions[stage["name"]] = flipped_pos

    # 5. 绘制平面、节点和标签
    total_nodes = sum(len(stage["nodes"]) for stage in stages)
    node_colors = plt.cm.tab20c(np.linspace(0, 1, total_nodes))
    color_index = 0  # 颜色索引计数器

    for stage in stages:
        x_plane = stage["x"]
        stage_name = stage["name"]

        # 绘制阶段平面
        y_range = [-6, 6]
        z_range = [-6, 6]
        y = np.linspace(y_range[0], y_range[1], 30)
        z = np.linspace(z_range[0], z_range[1], 30)
        y_grid, z_grid = np.meshgrid(y, z)
        x_grid = np.full_like(y_grid, x_plane)

        ax.plot_surface(
            x_grid, y_grid, z_grid,
            color=stage["color"],
            alpha=0.1,
            linewidth=0
        )

        # 绘制阶段标题
        ax.text(
            x_plane, y_range[1] + 1.5, z_range[1] / 2,
            f"{stage['name']}\n{stage['time']}",
            fontsize=22,
            ha='center', va='center',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=stage["color"], alpha=0.5)
        )

        # 绘制节点和标签
        for node in stage["nodes"]:
            label, size = node
            y_node, z_node = stage_node_positions[stage_name][label]

            # 绘制节点
            ax.scatter(
                x_plane, y_node, z_node,
                s=size,
                c=[node_colors[color_index]],
                alpha=0.9,
                edgecolors='black',
                linewidth=1.5,
                zorder=5
            )

            # 绘制标签
            font_size = max(10, min(14, size // 30))  # 自适应字体大小
            ax.text(
                x_plane, y_node, z_node,
                label,
                fontsize=font_size,
                ha='center', va='center',
                color='white',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5),
                zorder=10
            )

            color_index += 1

    # 6. 绘制连接线（平面内连线和跨平面传承连线）
    # 6.1 绘制平面内节点连线
    for stage in stages:
        x_plane = stage["x"]
        stage_name = stage["name"]
        node_labels = [node[0] for node in stage["nodes"]]

        # 两两连接平面内节点
        for i, label1 in enumerate(node_labels):
            y1, z1 = stage_node_positions[stage_name][label1]
            for label2 in node_labels[i + 1:]:
                y2, z2 = stage_node_positions[stage_name][label2]

                ax.plot(
                    [x_plane, x_plane], [y1, y2], [z1, z2],
                    color='darkgray',
                    linewidth=0.5,
                    alpha=0.3,
                    zorder=1
                )

    # 6.2 绘制跨平面传承连线（基于核心关键词匹配）
    keyword_node_map = {}
    for stage in stages:
        stage_name = stage["name"]
        x_plane = stage["x"]

        for node in stage["nodes"]:
            label, size = node
            # 核心关键词与对应节点的映射关系
            core_keywords = {
                "中华体育精神": ["弘扬中华体育精神", "中华体育精神弘扬"],
                "女排精神": ["弘扬女排精神", "海河回响女排精神展"],
                "传统体育文化": ["挖掘体育文化遗产", "岭南传统体育文化传承", "五禽戏挖掘传承"],
                "体育影视": ["体育影视与摄影创作", "体育影视作品走出去", "鼓励体育影视创作"]
            }

            # 匹配关键词并记录节点位置
            matched_key = None
            for key, keywords in core_keywords.items():
                if label in keywords:
                    matched_key = key
                    break

            if matched_key:
                if matched_key not in keyword_node_map:
                    keyword_node_map[matched_key] = []
                y, z = stage_node_positions[stage_name][label]
                keyword_node_map[matched_key].append((x_plane, y, z, size))

    # 绘制跨阶段传承连线
    for core_key, positions in keyword_node_map.items():
        if len(positions) > 1:  # 至少需要两个节点才能连线
            for i in range(len(positions) - 1):
                x1, y1, z1, _ = positions[i]
                x2, y2, z2, _ = positions[i + 1]

                ax.plot(
                    [x1, x2], [y1, y2], [z1, z2],
                    color='crimson',
                    linewidth=2.5,
                    alpha=0.9,
                    zorder=3,
                    label=core_key if i == 0 else ""  # 只在第一个连线添加图例
                )

    # 7. 图形美化与布局调整
    ax.set_title(
        '体育文化发展3D节点关系图（一二五-一四五时期）',
        fontsize=28,
        pad=50,
        fontweight='bold'
    )
    ax.set_axis_off()  # 隐藏坐标轴
    ax.view_init(elev=20, azim=-35)  # 设置3D视角

    # 调整坐标轴范围
    x_min = min(stage["x"] for stage in stages) - 3
    x_max = max(stage["x"] for stage in stages) + 3
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-6.5, 6.5)
    ax.set_zlim(-6.5, 6.5)

    # 消除边距，全屏显示
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc='upper right',
            fontsize=14,
            bbox_to_anchor=(0.95, 0.95),
            framealpha=0.9
        )

    plt.show()


if __name__ == "__main__":
    main()
#############################################################################################################