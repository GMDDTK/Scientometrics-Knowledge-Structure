import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

import umap


# ============================================================
# 0. 文件路径
# ============================================================

embedding_file_path = r"embeddings.npy"
text_file_path = r"all.txt"

output_path = r"SpectralClustering聚类结果.xlsx"

# 新增：完整可视化与聚类信息 JSON
json_output_path = r"SpectralClustering聚类评估可视化完整数据.json"


# ============================================================
# 1. 读取 Embeddings
# ============================================================

embeddings = np.load(
    embedding_file_path
)

print(
    f"Embeddings loaded from {embedding_file_path}, "
    f"shape: {embeddings.shape}"
)


# ============================================================
# 2. Matplotlib / JSON 辅助函数
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
    获取坐标轴在整个 Figure 中的实际位置。
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
    获取折线的实际数据及绘图属性。
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
            str(
                line.get_color()
            ),

        "line_style":
            str(
                line.get_linestyle()
            ),

        "line_width":
            float(
                line.get_linewidth()
            ),

        "marker":
            str(
                line.get_marker()
            ),

        "marker_size":
            float(
                line.get_markersize()
            ),

        "marker_face_color":
            str(
                line.get_markerfacecolor()
            ),

        "marker_edge_color":
            str(
                line.get_markeredgecolor()
            ),

        "alpha":
            (
                None
                if line.get_alpha() is None
                else float(line.get_alpha())
            )
    }


# ============================================================
# 3. UMAP 降维
# ============================================================

umap_n_components = 50
umap_random_state = 0
umap_n_neighbors = 15
umap_min_dist = 0.1
umap_metric = "cosine"


umap_model = umap.UMAP(

    n_components=
        umap_n_components,

    random_state=
        umap_random_state,

    n_neighbors=
        umap_n_neighbors,

    min_dist=
        umap_min_dist,

    metric=
        umap_metric
)


embeddings_reduced = (
    umap_model.fit_transform(
        embeddings
    )
)


print(
    f"Reduced embeddings shape (UMAP): "
    f"{embeddings_reduced.shape}"
)


# ============================================================
# 4. 搜索最佳聚类数
# ============================================================

def choose_best_k_spectral(
        embeddings,
        min_k=80,
        max_k=130,
        step=1
):

    # 转成 list，方便 JSON 保存和索引
    k_values = list(
        range(
            min_k,
            max_k + 1,
            step
        )
    )


    silhouette_scores = []

    # 每个 K 的完整结果
    evaluation_points = []


    # ========================================================
    # 4.1 遍历不同 K
    # ========================================================

    for k in k_values:

        print(
            f"Testing k={k}"
        )


        clustering = SpectralClustering(

            n_clusters=k,

            assign_labels="kmeans",

            random_state=0,

            n_jobs=-1,

            affinity="nearest_neighbors"
        )


        labels = clustering.fit_predict(
            embeddings
        )


        # ----------------------------------------------------
        # 实际产生多少个非空簇
        # ----------------------------------------------------

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )


        actual_cluster_count = int(
            len(
                unique_labels
            )
        )


        cluster_sizes = [

            {
                "cluster_id":
                    int(cluster_id),

                "count":
                    int(count)
            }

            for cluster_id, count
            in zip(
                unique_labels,
                counts
            )
        ]


        # ----------------------------------------------------
        # Silhouette Score
        # ----------------------------------------------------

        silhouette_status = "success"
        silhouette_error = None


        try:

            score = float(
                silhouette_score(
                    embeddings,
                    labels
                )
            )


            silhouette_scores.append(
                score
            )


            print(
                f"k={k}, "
                f"silhouette={score:.4f}"
            )


        except Exception as e:

            score = -1.0

            silhouette_scores.append(
                score
            )

            silhouette_status = "failed"
            silhouette_error = str(e)


        # ----------------------------------------------------
        # 保存当前 K 详细信息
        # ----------------------------------------------------

        evaluation_points.append({

            "k":
                int(k),

            "requested_number_of_clusters":
                int(k),

            "actual_number_of_nonempty_clusters":
                int(
                    actual_cluster_count
                ),

            "silhouette_score":
                float(score),

            "silhouette_status":
                silhouette_status,

            "silhouette_error":
                silhouette_error,

            "cluster_sizes":
                cluster_sizes
        })


    # ========================================================
    # 4.2 找到最佳 K
    # ========================================================

    best_index = int(
        np.argmax(
            silhouette_scores
        )
    )


    best_k = int(
        k_values[
            best_index
        ]
    )


    best_score = float(
        silhouette_scores[
            best_index
        ]
    )


    # ========================================================
    # 4.3 绘制曲线
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    line, = ax.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )


    ax.set_xlabel(
        "Number of Clusters (k)"
    )


    ax.set_ylabel(
        "Silhouette Score"
    )


    ax.set_title(
        "Spectral Clustering Silhouette Scores"
    )


    ax.grid(
        False
    )


    plt.tight_layout()


    # --------------------------------------------------------
    # 非常重要
    # 强制 Matplotlib 计算最终坐标范围、刻度及布局
    # --------------------------------------------------------

    fig.canvas.draw()


    # ========================================================
    # 4.4 构建完整 JSON
    # ========================================================

    visualization_json = {


        # ====================================================
        # Figure
        # ====================================================

        "figure": {

            "figure_type":
                "line_chart_with_markers",

            "purpose":
                "Spectral Clustering cluster-number selection using Silhouette Score",

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

            "original_embedding_shape": [
                int(v)
                for v in embeddings_original.shape
            ],

            "reduced_embedding_shape": [
                int(v)
                for v in embeddings.shape
            ],

            "number_of_samples":
                int(
                    embeddings.shape[0]
                )
        },


        # ====================================================
        # UMAP 参数
        # ====================================================

        "umap_settings": {

            "method":
                "UMAP",

            "n_components":
                int(
                    umap_n_components
                ),

            "random_state":
                int(
                    umap_random_state
                ),

            "n_neighbors":
                int(
                    umap_n_neighbors
                ),

            "min_dist":
                float(
                    umap_min_dist
                ),

            "metric":
                umap_metric,

            "original_dimension":
                int(
                    embeddings_original.shape[1]
                ),

            "reduced_dimension":
                int(
                    embeddings.shape[1]
                )
        },


        # ====================================================
        # Spectral Clustering 参数
        # ====================================================

        "spectral_clustering_settings": {

            "algorithm":
                "SpectralClustering",

            "assign_labels":
                "kmeans",

            "random_state":
                0,

            "n_jobs":
                -1,

            "affinity":
                "nearest_neighbors",

            "minimum_k":
                int(min_k),

            "maximum_k":
                int(max_k),

            "step":
                int(step),

            "number_of_tested_k_values":
                int(
                    len(
                        k_values
                    )
                ),

            "tested_k_values": [
                int(v)
                for v in k_values
            ]
        },


        # ====================================================
        # 完整评估结果
        # ====================================================

        "evaluation_data": {

            "k_values": [
                int(v)
                for v in k_values
            ],

            "silhouette_scores": [
                float(v)
                for v in silhouette_scores
            ],

            "points":
                evaluation_points
        },


        # ====================================================
        # 最佳 K
        # ====================================================

        "optimal_k_selection": {

            "selection_rule":
                "Maximum Silhouette Score",

            "best_point_index_zero_based":
                int(
                    best_index
                ),

            "best_point_index_one_based":
                int(
                    best_index + 1
                ),

            "optimal_k":
                int(
                    best_k
                ),

            "maximum_silhouette_score":
                float(
                    best_score
                )
        },


        # ====================================================
        # 图表信息
        # ====================================================

        "plot": {

            "chart_type":
                "line_chart_with_markers",

            "title":
                ax.get_title(),

            "axes_position":
                get_axes_position(
                    ax
                ),


            # ------------------------------------------------
            # X 轴
            # ------------------------------------------------

            "x_axis": {

                "label":
                    ax.get_xlabel(),

                "semantic_variable":
                    "Number of Spectral Clustering clusters",

                "data_values": [
                    int(v)
                    for v in k_values
                ],

                "displayed_ticks":
                    get_tick_information(
                        ax,
                        "x"
                    ),

                "axis_limits": {

                    "min":
                        float(
                            ax.get_xlim()[0]
                        ),

                    "max":
                        float(
                            ax.get_xlim()[1]
                        )
                }
            },


            # ------------------------------------------------
            # Y 轴
            # ------------------------------------------------

            "y_axis": {

                "label":
                    ax.get_ylabel(),

                "semantic_variable":
                    "Silhouette Score",

                "data_values": [
                    float(v)
                    for v in silhouette_scores
                ],

                "displayed_ticks":
                    get_tick_information(
                        ax,
                        "y"
                    ),

                "axis_limits": {

                    "min":
                        float(
                            ax.get_ylim()[0]
                        ),

                    "max":
                        float(
                            ax.get_ylim()[1]
                        )
                }
            },


            # ------------------------------------------------
            # 图中每一个点
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 折线实际绘图属性
            # ------------------------------------------------

            "line_properties":
                get_line_information(
                    line
                ),


            "grid_displayed":
                False
        }
    }


    # ========================================================
    # 4.5 显示图
    # ========================================================

    plt.show()


    return (
        best_k,
        visualization_json
    )


# ============================================================
# 5. 搜索最佳 Spectral Clustering K
# ============================================================

# 保留原始 Embeddings，用于 JSON
embeddings_original = embeddings


optimal_k, visualization_json = (
    choose_best_k_spectral(
        embeddings_reduced,
        min_k=80,
        max_k=130,
        step=1
    )
)


print(
    f"Optimal number of clusters (Spectral): "
    f"{optimal_k}"
)


# ============================================================
# 6. 使用最佳 K 执行最终谱聚类
# ============================================================

clustering = SpectralClustering(

    n_clusters=optimal_k,

    assign_labels="kmeans",

    random_state=0,

    n_jobs=-1,

    affinity="nearest_neighbors"
)


labels = clustering.fit_predict(
    embeddings_reduced
)


# ============================================================
# 7. 读取文本数据
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

    "text":
        dataset,

    "cluster":
        labels
})


# ============================================================
# 8. 统计各个类的样本数
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
# 9. 提取高频词
# ============================================================

vectorizer = CountVectorizer(
    stop_words="english"
)


X = vectorizer.fit_transform(
    df["text"]
)


# ------------------------------------------------------------
# sklearn 新旧版本兼容
# ------------------------------------------------------------

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

    # SpectralClustering 正常不会生成 -1，
    # 保留这一判断以兼容你的原代码逻辑
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
# 10. 构建最终聚类摘要
# ============================================================

final_cluster_summary = []


for cluster_id in sorted(
    cluster_high_freq_words.keys()
):

    count = int(
        cluster_counts.loc[
            cluster_id
        ]
    )


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
                count / len(labels)
            ),

        "top_20_high_frequency_words":
            top_words
    })


# ============================================================
# 11. 最终聚类结果加入 JSON
# ============================================================

final_silhouette_score = float(
    silhouette_score(
        embeddings_reduced,
        labels
    )
)


visualization_json[
    "final_clustering"
] = {

    "algorithm":
        "SpectralClustering",

    "input_space":
        "UMAP-reduced embeddings",

    "selected_by":
        "Maximum Silhouette Score",

    "optimal_k":
        int(
            optimal_k
        ),

    "requested_number_of_clusters":
        int(
            optimal_k
        ),

    "actual_number_of_nonempty_clusters":
        int(
            len(
                np.unique(
                    labels
                )
            )
        ),

    "assign_labels":
        "kmeans",

    "affinity":
        "nearest_neighbors",

    "random_state":
        0,

    "n_jobs":
        -1,

    "number_of_samples":
        int(
            len(labels)
        ),

    "silhouette_score":
        float(
            final_silhouette_score
        ),


    # --------------------------------------------------------
    # 每个簇的大小
    # --------------------------------------------------------

    "cluster_counts": [

        {
            "cluster_id":
                int(cluster_id),

            "count":
                int(count),

            "proportion":
                float(
                    count / len(labels)
                )
        }

        for cluster_id, count
        in cluster_counts.items()
    ],


    # --------------------------------------------------------
    # 每个簇的详细信息
    # --------------------------------------------------------

    "clusters":
        final_cluster_summary
}


# ============================================================
# 12. 文件信息
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
# 13. 保存 JSON
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
    f"✅ Spectral Clustering 参数搜索、"
    f"可视化及最终聚类信息已保存到："
    f"{json_output_path}"
)


# ============================================================
# 14. 输出 Excel
# ============================================================

with pd.ExcelWriter(
    output_path
) as writer:


    # --------------------------------------------------------
    # Sheet 1
    # 文本与聚类标签
    # --------------------------------------------------------

    df.to_excel(
        writer,
        index=False,
        sheet_name="Clustered Texts"
    )


    # --------------------------------------------------------
    # Sheet 2
    # 每个簇的样本数
    # --------------------------------------------------------

    cluster_counts_df = (
        cluster_counts
        .rename("count")
        .to_frame()
    )


    cluster_counts_df.to_excel(
        writer,
        sheet_name="Cluster Counts"
    )


    # --------------------------------------------------------
    # Sheet 3
    # 高频词
    # --------------------------------------------------------

    high_freq_words_df = (
        pd.DataFrame.from_dict(
            cluster_high_freq_words,
            orient="index"
        )
    )


    high_freq_words_df.columns = [
        f"Word_{i + 1}"
        for i in range(
            high_freq_words_df.shape[1]
        )
    ]


    high_freq_words_df.to_excel(
        writer,
        sheet_name="High Frequency Words"
    )


print(
    f"Spectral Clustering 聚类结果已保存到 "
    f"{output_path}"
)