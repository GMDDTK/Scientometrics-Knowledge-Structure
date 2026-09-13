import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# 0. 文件路径
# ============================================================

embedding_file_path = r"embeddings.npy"

text_file_path = r"all.txt"

output_path = r"聚类结果.xlsx"

# 可视化及聚类信息 JSON
json_output_path = r"KMeans聚类评估可视化完整数据.json"


# ============================================================
# 1. 读取语义嵌入
# ============================================================

embeddings = np.load(
    embedding_file_path
)

print(
    f"Embeddings loaded from {embedding_file_path}, "
    f"shape: {embeddings.shape}"
)


# ============================================================
# 2. JSON / Matplotlib 辅助函数
# ============================================================

def get_tick_information(ax, orientation):
    """
    获取 Matplotlib 最终实际显示的刻度位置和刻度标签。
    """

    if orientation == "x":

        tick_values = ax.get_xticks()

        tick_labels = [
            label.get_text()
            for label in ax.get_xticklabels()
        ]

    elif orientation == "y":

        tick_values = ax.get_yticks()

        tick_labels = [
            label.get_text()
            for label in ax.get_yticklabels()
        ]

    else:

        raise ValueError(
            "orientation must be 'x' or 'y'"
        )

    return [
        {
            "position": float(value),
            "label": str(label)
        }
        for value, label
        in zip(
            tick_values,
            tick_labels
        )
    ]


def get_axes_position(ax):
    """
    获取子图在 Figure 中的最终位置。
    坐标为 0~1 范围内的 Figure 相对坐标。
    """

    pos = ax.get_position()

    return {
        "left": float(pos.x0),
        "bottom": float(pos.y0),
        "right": float(pos.x1),
        "top": float(pos.y1),
        "width": float(pos.width),
        "height": float(pos.height)
    }


def get_line_information(line):
    """
    获取折线最终绘图属性以及全部 X/Y 数据。
    """

    return {

        "x_data": [
            float(v)
            for v in line.get_xdata()
        ],

        "y_data": [
            float(v)
            for v in line.get_ydata()
        ],

        "color":
            str(line.get_color()),

        "line_style":
            str(line.get_linestyle()),

        "line_width":
            float(line.get_linewidth()),

        "marker":
            str(line.get_marker()),

        "marker_size":
            float(line.get_markersize()),

        "marker_face_color":
            str(
                line.get_markerfacecolor()
            ),

        "marker_edge_color":
            str(
                line.get_markeredgecolor()
            )
    }


# ============================================================
# 3. 选择最佳 K
# ============================================================

def choose_best_k(
        embeddings,
        max_k=90,
        json_save_path=json_output_path
):

    # --------------------------------------------------------
    # K 搜索范围
    # --------------------------------------------------------

    min_k = 30
    k_step = 1

    k_values = list(
        range(
            min_k,
            max_k + 1,
            k_step
        )
    )

    inertia_values = []
    silhouette_scores = []

    # 每个 k 的完整评估信息
    evaluation_points = []


    # ========================================================
    # 3.1 对所有 K 进行评估
    # ========================================================

    for k in k_values:

        print(
            f"Testing k = {k}"
        )

        kmeans = KMeans(
            n_clusters=k,
            random_state=0,
            n_init=10
        )

        kmeans.fit(
            embeddings
        )

        # ----------------------
        # SSE
        # ----------------------

        inertia = float(
            kmeans.inertia_
        )

        inertia_values.append(
            inertia
        )


        # ----------------------
        # Silhouette Score
        # ----------------------

        silhouette_avg = float(
            silhouette_score(
                embeddings,
                kmeans.labels_
            )
        )

        silhouette_scores.append(
            silhouette_avg
        )


        # ----------------------
        # 当前 k 的各簇大小
        # ----------------------

        unique_labels, label_counts = (
            np.unique(
                kmeans.labels_,
                return_counts=True
            )
        )

        cluster_sizes = [
            {
                "cluster":
                    int(cluster_id),

                "count":
                    int(count)
            }
            for cluster_id, count
            in zip(
                unique_labels,
                label_counts
            )
        ]


        # ----------------------
        # 保存 k 对应完整信息
        # ----------------------

        evaluation_points.append({

            "k":
                int(k),

            "number_of_clusters":
                int(
                    len(unique_labels)
                ),

            "inertia_sse":
                float(inertia),

            "silhouette_score":
                float(silhouette_avg),

            "cluster_sizes":
                cluster_sizes
        })


    # ========================================================
    # 3.2 选择最佳 K
    # ========================================================

    best_index = int(
        np.argmax(
            silhouette_scores
        )
    )

    optimal_k = int(
        k_values[
            best_index
        ]
    )

    best_silhouette_score = float(
        silhouette_scores[
            best_index
        ]
    )

    inertia_at_optimal_k = float(
        inertia_values[
            best_index
        ]
    )


    # ========================================================
    # 3.3 绘制图
    # ========================================================

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )


    # --------------------------------------------------------
    # 左图：Elbow Method
    # --------------------------------------------------------

    line_inertia, = ax1.plot(
        k_values,
        inertia_values,
        marker='o'
    )

    ax1.set_xlabel(
        'Number of Clusters (k)'
    )

    ax1.set_ylabel(
        'Inertia (SSE)'
    )

    ax1.set_title(
        'Elbow Method for Optimal k'
    )

    ax1.grid(
        False
    )


    # --------------------------------------------------------
    # 右图：Silhouette Score
    # --------------------------------------------------------

    line_silhouette, = ax2.plot(
        k_values,
        silhouette_scores,
        marker='o'
    )

    ax2.set_xlabel(
        'Number of Clusters (k)'
    )

    ax2.set_ylabel(
        'Silhouette Score'
    )

    ax2.set_title(
        'Silhouette Scores for Different k'
    )

    ax2.grid(
        False
    )


    # ========================================================
    # 3.4 Final Layout
    # ========================================================

    plt.tight_layout()

    # 强制 Matplotlib 完成最终坐标、
    # tick、layout 等计算
    fig.canvas.draw()


    # ========================================================
    # 3.5 构建可视化 JSON
    # ========================================================

    visualization_json = {

        # ====================================================
        # Figure 层面
        # ====================================================

        "figure": {

            "figure_type":
                "two-panel line chart",

            "purpose":
                "KMeans cluster-number evaluation using Elbow Method and Silhouette Score",

            "layout": {
                "rows": 1,
                "columns": 2,
                "number_of_subplots": 2
            },

            "figsize_inches": {
                "width":
                    float(
                        fig.get_size_inches()[0]
                    ),

                "height":
                    float(
                        fig.get_size_inches()[1]
                    )
            },

            "source_embedding_file":
                embedding_file_path,

            "embedding_shape": [
                int(v)
                for v in embeddings.shape
            ],

            "number_of_samples":
                int(
                    embeddings.shape[0]
                ),

            "embedding_dimension":
                (
                    int(
                        embeddings.shape[1]
                    )
                    if embeddings.ndim >= 2
                    else None
                )
        },


        # ====================================================
        # KMeans 参数
        # ====================================================

        "kmeans_settings": {

            "algorithm":
                "KMeans",

            "random_state":
                0,

            "n_init":
                10,

            "minimum_k":
                int(min_k),

            "maximum_k":
                int(max_k),

            "k_step":
                int(k_step),

            "number_of_tested_k_values":
                int(
                    len(k_values)
                ),

            "tested_k_values": [
                int(v)
                for v in k_values
            ]
        },


        # ====================================================
        # 全部评估数据
        # ====================================================

        "evaluation_data": {

            "k_values": [
                int(v)
                for v in k_values
            ],

            "inertia_values": [
                float(v)
                for v in inertia_values
            ],

            "silhouette_scores": [
                float(v)
                for v in silhouette_scores
            ],

            "points":
                evaluation_points
        },


        # ====================================================
        # 最优 K
        # ====================================================

        "optimal_k_selection": {

            "selection_method":
                "Maximum Silhouette Score",

            "optimal_point_index_zero_based":
                int(best_index),

            "optimal_point_index_one_based":
                int(best_index + 1),

            "optimal_k":
                int(optimal_k),

            "maximum_silhouette_score":
                float(
                    best_silhouette_score
                ),

            "inertia_sse_at_optimal_k":
                float(
                    inertia_at_optimal_k
                )
        },


        # ====================================================
        # 左图
        # ====================================================

        "subplot_1_elbow": {

            "subplot_index":
                1,

            "grid_position": {
                "row": 1,
                "column": 1
            },

            "chart_type":
                "line_chart_with_markers",

            "title":
                ax1.get_title(),

            "axes_position":
                get_axes_position(
                    ax1
                ),

            "x_axis": {

                "label":
                    ax1.get_xlabel(),

                "semantic_variable":
                    "Number of Clusters (k)",

                "data_values": [
                    int(v)
                    for v in k_values
                ],

                "displayed_ticks":
                    get_tick_information(
                        ax1,
                        "x"
                    ),

                "axis_limits": {
                    "min":
                        float(
                            ax1.get_xlim()[0]
                        ),

                    "max":
                        float(
                            ax1.get_xlim()[1]
                        )
                }
            },

            "y_axis": {

                "label":
                    ax1.get_ylabel(),

                "semantic_variable":
                    "Inertia (SSE)",

                "data_values": [
                    float(v)
                    for v in inertia_values
                ],

                "displayed_ticks":
                    get_tick_information(
                        ax1,
                        "y"
                    ),

                "axis_limits": {
                    "min":
                        float(
                            ax1.get_ylim()[0]
                        ),

                    "max":
                        float(
                            ax1.get_ylim()[1]
                        )
                }
            },

            "data_points": [
                {
                    "point_index":
                        int(i + 1),

                    "k":
                        int(k),

                    "inertia_sse":
                        float(inertia),

                    "is_optimal_k_by_silhouette":
                        bool(
                            i == best_index
                        )
                }

                for i, (k, inertia)
                in enumerate(
                    zip(
                        k_values,
                        inertia_values
                    )
                )
            ],

            "line_properties":
                get_line_information(
                    line_inertia
                ),

            "grid_displayed":
                False
        },


        # ====================================================
        # 右图
        # ====================================================

        "subplot_2_silhouette": {

            "subplot_index":
                2,

            "grid_position": {
                "row": 1,
                "column": 2
            },

            "chart_type":
                "line_chart_with_markers",

            "title":
                ax2.get_title(),

            "axes_position":
                get_axes_position(
                    ax2
                ),

            "x_axis": {

                "label":
                    ax2.get_xlabel(),

                "semantic_variable":
                    "Number of Clusters (k)",

                "data_values": [
                    int(v)
                    for v in k_values
                ],

                "displayed_ticks":
                    get_tick_information(
                        ax2,
                        "x"
                    ),

                "axis_limits": {
                    "min":
                        float(
                            ax2.get_xlim()[0]
                        ),

                    "max":
                        float(
                            ax2.get_xlim()[1]
                        )
                }
            },

            "y_axis": {

                "label":
                    ax2.get_ylabel(),

                "semantic_variable":
                    "Silhouette Score",

                "data_values": [
                    float(v)
                    for v in silhouette_scores
                ],

                "displayed_ticks":
                    get_tick_information(
                        ax2,
                        "y"
                    ),

                "axis_limits": {
                    "min":
                        float(
                            ax2.get_ylim()[0]
                        ),

                    "max":
                        float(
                            ax2.get_ylim()[1]
                        )
                }
            },

            "data_points": [
                {
                    "point_index":
                        int(i + 1),

                    "k":
                        int(k),

                    "silhouette_score":
                        float(score),

                    "is_optimal_k":
                        bool(
                            i == best_index
                        )
                }

                for i, (k, score)
                in enumerate(
                    zip(
                        k_values,
                        silhouette_scores
                    )
                )
            ],

            "line_properties":
                get_line_information(
                    line_silhouette
                ),

            "grid_displayed":
                False,

            "optimal_point": {

                "k":
                    int(optimal_k),

                "silhouette_score":
                    float(
                        best_silhouette_score
                    ),

                "inertia_sse":
                    float(
                        inertia_at_optimal_k
                    )
            }
        }
    }


    # ========================================================
    # 暂存 JSON
    #
    # 最终聚类完成后还会继续往里面添加信息
    # ========================================================

    return (
        optimal_k,
        visualization_json
    )


# ============================================================
# 4. 自动选择最佳 K
# ============================================================

optimal_k, visualization_json = choose_best_k(
    embeddings,
    max_k=90,
    json_save_path=json_output_path
)

print(
    f"Optimal number of clusters (k): "
    f"{optimal_k}"
)


# ============================================================
# 5. 使用最佳 K 执行最终 KMeans 聚类
# ============================================================

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=0,
    n_init=10
)

kmeans.fit(
    embeddings
)

labels = kmeans.labels_


# ============================================================
# 6. 载入文本数据
# ============================================================

with open(
    text_file_path,
    "r",
    encoding="utf-8"
) as f:

    dataset = [
        line.strip()
        for line in f.readlines()
    ]


assert (
    len(dataset)
    == embeddings.shape[0]
), "❌ 文本数量和 embeddings 数量不一致！"


df = pd.DataFrame({
    "text": dataset,
    "cluster": labels
})


# ============================================================
# 7. 各簇样本数量
# ============================================================

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

print(
    "Cluster Counts:"
)

print(
    cluster_counts
)


# ============================================================
# 8. 提取各类高频词
# ============================================================

vectorizer = CountVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(
    df["text"]
)


# ============================================================
# sklearn 新旧版本兼容
# ============================================================

try:

    terms = (
        vectorizer
        .get_feature_names_out()
        .tolist()
    )

except AttributeError:

    terms = (
        vectorizer
        .get_feature_names()
    )


cluster_high_freq_words = {}


for cluster_id in sorted(
    set(labels)
):

    # KMeans 理论上不会产生 -1，
    # 这里保留原代码逻辑
    if cluster_id == -1:
        continue


    cluster_texts = df[
        df["cluster"] == cluster_id
    ]["text"]


    cluster_X = vectorizer.transform(
        cluster_texts
    )


    sum_words = cluster_X.sum(
        axis=0
    )


    word_freq = [

        (
            word,
            int(
                sum_words[
                    0,
                    idx
                ]
            )
        )

        for word, idx
        in vectorizer.vocabulary_.items()
    ]


    word_freq = sorted(
        word_freq,
        key=lambda x: x[1],
        reverse=True
    )


    cluster_high_freq_words[
        int(cluster_id)
    ] = word_freq[:20]


# ============================================================
# 9. 打印高频词
# ============================================================

print(
    "\nHigh Frequency Words for Each Cluster:"
)


for cluster_id, words in (
    cluster_high_freq_words.items()
):

    print(
        f"Cluster {cluster_id}:",
        [
            word[0]
            for word in words
        ]
    )


# ============================================================
# 10. 在 JSON 中加入最终聚类结果摘要
# ============================================================

final_cluster_summary = []


for cluster_id in sorted(
    cluster_high_freq_words.keys()
):

    # 当前簇样本数量
    count = int(
        cluster_counts.loc[
            cluster_id
        ]
    )


    # 高频词 + 对应频数
    top_words = [

        {
            "rank":
                int(rank),

            "word":
                str(word),

            "frequency":
                int(freq)
        }

        for rank, (word, freq)
        in enumerate(
            cluster_high_freq_words[
                cluster_id
            ],
            start=1
        )
    ]


    final_cluster_summary.append({

        "cluster_id":
            int(cluster_id),

        "number_of_samples":
            int(count),

        "proportion_of_all_samples":
            float(
                count / len(df)
            ),

        "top_20_high_frequency_words":
            top_words
    })


visualization_json[
    "final_clustering"
] = {

    "algorithm":
        "KMeans",

    "optimal_k":
        int(optimal_k),

    "actual_number_of_clusters":
        int(
            len(
                np.unique(labels)
            )
        ),

    "number_of_samples":
        int(
            len(labels)
        ),

    "random_state":
        0,

    "n_init":
        10,

    "final_inertia_sse":
        float(
            kmeans.inertia_
        ),

    "cluster_counts": [

        {
            "cluster_id":
                int(cluster_id),

            "count":
                int(count),

            "proportion":
                float(
                    count / len(df)
                )
        }

        for cluster_id, count
        in cluster_counts.items()
    ],

    "clusters":
        final_cluster_summary
}


# ============================================================
# 11. 数据源及输出文件信息
# ============================================================

visualization_json[
    "files"
] = {

    "embedding_file":
        embedding_file_path,

    "text_file":
        text_file_path,

    "excel_output":
        output_path,

    "json_output":
        json_output_path
}


# ============================================================
# 12. 保存完整 JSON
# ============================================================

with open(
    json_output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        visualization_json,
        f,
        ensure_ascii=False,
        indent=4,
        allow_nan=False
    )


print(
    f"✅ 聚类评估与可视化完整 JSON 已保存到："
    f"{json_output_path}"
)


# ============================================================
# 13. 输出结果到 Excel
# ============================================================

with pd.ExcelWriter(
    output_path
) as writer:

    # --------------------------------------------------------
    # Sheet 1：文本及聚类标签
    # --------------------------------------------------------

    df.to_excel(
        writer,
        index=False,
        sheet_name='Clustered Texts'
    )


    # --------------------------------------------------------
    # Sheet 2：每类样本数量
    # --------------------------------------------------------

    cluster_counts_df = (
        cluster_counts
        .rename("count")
        .to_frame()
    )

    cluster_counts_df.to_excel(
        writer,
        sheet_name='Cluster Counts'
    )


    # --------------------------------------------------------
    # Sheet 3：高频词
    # --------------------------------------------------------

    high_freq_words_df = (
        pd.DataFrame.from_dict(
            cluster_high_freq_words,
            orient='index'
        )
    )


    high_freq_words_df.columns = [
        f'Word_{i + 1}'
        for i in range(
            high_freq_words_df.shape[1]
        )
    ]


    high_freq_words_df.to_excel(
        writer,
        sheet_name='High Frequency Words'
    )


print(
    f"✅ 聚类结果已保存到 {output_path}"
)