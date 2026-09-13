import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan
import umap


# ============================================================
# 0. 文件路径
# ============================================================

embedding_file_path = r"embeddings.npy"
text_file_path = r"all.txt"

output_path = r"HDBSCAN聚类结果_UMAP降维.xlsx"

json_output_path = r"HDBSCAN_UMAP聚类评估可视化完整数据.json"


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
# 2. JSON / Matplotlib 辅助函数
# ============================================================

def get_tick_information(ax, orientation):
    """
    获取 Matplotlib 最终实际显示的刻度位置与标签。
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
    获取折线实际绘图属性和数据。
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

        "alpha":
            (
                None
                if line.get_alpha() is None
                else float(line.get_alpha())
            ),

        "marker":
            str(line.get_marker()),

        "marker_size":
            float(line.get_markersize())
    }


def get_scatter_information(scatter):
    """
    获取散点图的实际坐标及基本绘图属性。
    """

    offsets = scatter.get_offsets()

    points = []

    for i, point in enumerate(offsets):

        points.append({
            "point_index": int(i + 1),
            "x": float(point[0]),
            "y": float(point[1])
        })

    sizes = scatter.get_sizes()

    return {

        "points":
            points,

        "marker_sizes": [
            float(v)
            for v in sizes
        ],

        "number_of_points":
            int(len(offsets))
    }


# ============================================================
# 3. UMAP 降维
# ============================================================

def umap_dimensionality_reduction(
        embeddings,
        n_components=50,
        n_neighbors=15,
        min_dist=0.1
):

    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )

    embeddings_umap = umap_model.fit_transform(
        embeddings
    )

    print(
        f"UMAP reduced dimensions: "
        f"{embeddings_umap.shape}"
    )

    return embeddings_umap


# ============================================================
# 4. 搜索最佳 HDBSCAN 参数
# ============================================================

def search_best_hdbscan(
        embeddings,
        min_cluster_size_values
):

    best_score = -1.0
    best_param = None
    best_labels = None

    # 只保存成功计算 silhouette 的点
    successful_results = []

    # 保存所有测试参数，包括失败/跳过情况
    all_parameter_results = []


    # ========================================================
    # 4.1 遍历所有 min_cluster_size
    # ========================================================

    for min_cluster_size in min_cluster_size_values:

        print(
            f"Testing min_cluster_size = "
            f"{min_cluster_size}"
        )


        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )


        labels = clusterer.fit_predict(
            embeddings
        )


        # ----------------------------------------------------
        # 实际簇数
        # ----------------------------------------------------

        unique_labels = set(
            labels
        )

        n_clusters = (
            len(unique_labels)
            - (1 if -1 in unique_labels else 0)
        )


        # ----------------------------------------------------
        # 噪声信息
        # ----------------------------------------------------

        noise_count = int(
            np.sum(
                labels == -1
            )
        )

        non_noise_count = int(
            np.sum(
                labels != -1
            )
        )

        total_count = int(
            len(labels)
        )

        noise_ratio = float(
            noise_count / total_count
        )


        # ----------------------------------------------------
        # 各簇样本量
        # ----------------------------------------------------

        cluster_sizes = []

        for cluster_id in sorted(
            unique_labels
        ):

            cluster_sizes.append({

                "cluster_id":
                    int(cluster_id),

                "is_noise":
                    bool(cluster_id == -1),

                "count":
                    int(
                        np.sum(
                            labels == cluster_id
                        )
                    )
            })


        # ----------------------------------------------------
        # 少于 2 个簇无法计算 silhouette
        # ----------------------------------------------------

        if n_clusters < 2:

            all_parameter_results.append({

                "min_cluster_size":
                    int(min_cluster_size),

                "status":
                    "skipped",

                "reason":
                    "Fewer than 2 non-noise clusters",

                "number_of_clusters":
                    int(n_clusters),

                "noise_count":
                    int(noise_count),

                "noise_ratio":
                    float(noise_ratio),

                "non_noise_count":
                    int(non_noise_count),

                "silhouette_score":
                    None,

                "cluster_sizes":
                    cluster_sizes
            })

            continue


        # ----------------------------------------------------
        # Silhouette Score
        # 仅针对非噪声点
        # ----------------------------------------------------

        try:

            score = float(
                silhouette_score(
                    embeddings[
                        labels != -1
                    ],
                    labels[
                        labels != -1
                    ]
                )
            )


            successful_results.append(
                (
                    min_cluster_size,
                    n_clusters,
                    score,
                    noise_count,
                    noise_ratio
                )
            )


            all_parameter_results.append({

                "min_cluster_size":
                    int(min_cluster_size),

                "status":
                    "success",

                "reason":
                    None,

                "number_of_clusters":
                    int(n_clusters),

                "noise_count":
                    int(noise_count),

                "noise_ratio":
                    float(noise_ratio),

                "non_noise_count":
                    int(non_noise_count),

                "silhouette_score":
                    float(score),

                "cluster_sizes":
                    cluster_sizes
            })


            print(
                f"min_cluster_size={min_cluster_size}, "
                f"clusters={n_clusters}, "
                f"noise={noise_count}, "
                f"silhouette={score:.4f}"
            )


            # ------------------------------------------------
            # 更新最佳参数
            # ------------------------------------------------

            if score > best_score:

                best_score = score
                best_param = min_cluster_size
                best_labels = labels.copy()


        except Exception as e:

            all_parameter_results.append({

                "min_cluster_size":
                    int(min_cluster_size),

                "status":
                    "failed",

                "reason":
                    str(e),

                "number_of_clusters":
                    int(n_clusters),

                "noise_count":
                    int(noise_count),

                "noise_ratio":
                    float(noise_ratio),

                "non_noise_count":
                    int(non_noise_count),

                "silhouette_score":
                    None,

                "cluster_sizes":
                    cluster_sizes
            })

            continue


    # ========================================================
    # 4.2 如果没有任何有效结果
    # ========================================================

    if not successful_results:

        raise RuntimeError(
            "没有任何 HDBSCAN 参数能够产生有效的 "
            "Silhouette Score。"
        )


    # ========================================================
    # 4.3 构建 DataFrame
    # ========================================================

    df_results = pd.DataFrame(

        successful_results,

        columns=[
            "min_cluster_size",
            "n_clusters",
            "silhouette",
            "noise_count",
            "noise_ratio"
        ]
    )


    # ========================================================
    # 4.4 绘制参数搜索结果
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    # 散点
    scatter = ax.scatter(
        df_results["min_cluster_size"],
        df_results["silhouette"],
        c="blue",
        marker="o"
    )


    # 折线
    line, = ax.plot(
        df_results["min_cluster_size"],
        df_results["silhouette"],
        linestyle="-",
        alpha=0.6
    )


    ax.set_xlabel(
        "min_cluster_size"
    )

    ax.set_ylabel(
        "Silhouette Score"
    )

    ax.set_title(
        "HDBSCAN Parameter Search (Euclidean Metric)"
    )

    ax.grid(
        False
    )


    plt.tight_layout()

    # 强制 Matplotlib 完成最终布局
    fig.canvas.draw()


    # ========================================================
    # 4.5 最佳参数对应信息
    # ========================================================

    best_result = next(
        item
        for item in all_parameter_results
        if (
            item["min_cluster_size"]
            == best_param
            and item["status"] == "success"
        )
    )


    # ========================================================
    # 4.6 构建完整可视化 JSON
    # ========================================================

    visualization_info = {

        # ====================================================
        # Figure
        # ====================================================

        "figure": {

            "figure_type":
                "scatter_and_line_chart",

            "purpose":
                "HDBSCAN min_cluster_size parameter search",

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

            "umap_embedding_shape": [
                int(v)
                for v in embeddings.shape
            ]
        },


        # ====================================================
        # UMAP
        # ====================================================

        "umap_settings": {

            "n_components":
                50,

            "n_neighbors":
                15,

            "min_dist":
                0.1,

            "random_state":
                42
        },


        # ====================================================
        # HDBSCAN 固定参数
        # ====================================================

        "hdbscan_settings": {

            "metric":
                "euclidean",

            "cluster_selection_method":
                "eom",

            "prediction_data":
                True,

            "tested_min_cluster_size_values": [
                int(v)
                for v in min_cluster_size_values
            ],

            "minimum_tested_value":
                int(
                    min(min_cluster_size_values)
                ),

            "maximum_tested_value":
                int(
                    max(min_cluster_size_values)
                ),

            "number_of_tested_values":
                int(
                    len(min_cluster_size_values)
                )
        },


        # ====================================================
        # 所有参数搜索结果
        # ====================================================

        "parameter_search": {

            "total_tested":
                int(
                    len(
                        min_cluster_size_values
                    )
                ),

            "successful_silhouette_calculations":
                int(
                    len(
                        successful_results
                    )
                ),

            "all_results":
                all_parameter_results,

            "plotted_results": [

                {
                    "point_index":
                        int(i + 1),

                    "min_cluster_size":
                        int(row["min_cluster_size"]),

                    "number_of_clusters":
                        int(row["n_clusters"]),

                    "silhouette_score":
                        float(row["silhouette"]),

                    "noise_count":
                        int(row["noise_count"]),

                    "noise_ratio":
                        float(row["noise_ratio"]),

                    "is_best_parameter":
                        bool(
                            int(
                                row["min_cluster_size"]
                            )
                            == best_param
                        )
                }

                for i, (_, row)
                in enumerate(
                    df_results.iterrows()
                )
            ]
        },


        # ====================================================
        # 最佳参数
        # ====================================================

        "best_parameter": {

            "selection_rule":
                "Maximum silhouette score among non-noise samples",

            "min_cluster_size":
                int(best_param),

            "silhouette_score":
                float(best_score),

            "number_of_clusters":
                int(
                    best_result[
                        "number_of_clusters"
                    ]
                ),

            "noise_count":
                int(
                    best_result[
                        "noise_count"
                    ]
                ),

            "noise_ratio":
                float(
                    best_result[
                        "noise_ratio"
                    ]
                ),

            "non_noise_count":
                int(
                    best_result[
                        "non_noise_count"
                    ]
                )
        },


        # ====================================================
        # 图表本身
        # ====================================================

        "plot": {

            "chart_type":
                "scatter_with_connecting_line",

            "title":
                ax.get_title(),

            "axes_position":
                get_axes_position(
                    ax
                ),

            "x_axis": {

                "label":
                    ax.get_xlabel(),

                "semantic_variable":
                    "HDBSCAN min_cluster_size",

                "data_values": [
                    int(v)
                    for v in df_results[
                        "min_cluster_size"
                    ].tolist()
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


            "y_axis": {

                "label":
                    ax.get_ylabel(),

                "semantic_variable":
                    "Silhouette Score",

                "data_values": [
                    float(v)
                    for v in df_results[
                        "silhouette"
                    ].tolist()
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


            "data_points": [

                {
                    "point_index":
                        int(i + 1),

                    "min_cluster_size":
                        int(row["min_cluster_size"]),

                    "silhouette_score":
                        float(row["silhouette"]),

                    "number_of_clusters":
                        int(row["n_clusters"]),

                    "noise_count":
                        int(row["noise_count"]),

                    "noise_ratio":
                        float(row["noise_ratio"]),

                    "is_best_parameter":
                        bool(
                            int(
                                row["min_cluster_size"]
                            )
                            == best_param
                        )
                }

                for i, (_, row)
                in enumerate(
                    df_results.iterrows()
                )
            ],


            "scatter_properties":
                get_scatter_information(
                    scatter
                ),


            "line_properties":
                get_line_information(
                    line
                ),


            "grid_displayed":
                False
        }
    }


    # ========================================================
    # 4.7 显示图
    # ========================================================

    plt.show()


    return (
        best_param,
        best_score,
        best_labels,
        visualization_info
    )


# ============================================================
# 5. UMAP 降维
# ============================================================

# 保留原始 embeddings 引用
embeddings_original = embeddings


embeddings_umap = umap_dimensionality_reduction(
    embeddings,
    n_components=50,
    n_neighbors=15,
    min_dist=0.1
)


# ============================================================
# 6. 搜索最佳 HDBSCAN 参数
# ============================================================

min_cluster_size_range = list(
    range(
        10,
        220,
        1
    )
)


(
    best_param,
    best_score,
    labels,
    visualization_json
) = search_best_hdbscan(
    embeddings_umap,
    min_cluster_size_range
)


print(
    f"\nBest HDBSCAN param: "
    f"min_cluster_size={best_param}"
)

print(
    f"Best silhouette score: "
    f"{best_score:.4f}"
)


# ============================================================
# 7. 读取文本
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
# 8. 各类样本数量
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
# 9. 高频词
# ============================================================

vectorizer = CountVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(
    df["text"]
)


# sklearn 新旧版本兼容
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

    # HDBSCAN 中 -1 是噪声
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
# 10. 最终聚类结果写入 JSON
# ============================================================

final_unique_labels = sorted(
    set(labels)
)

final_cluster_ids = [
    int(v)
    for v in final_unique_labels
    if v != -1
]


final_noise_count = int(
    np.sum(
        labels == -1
    )
)

final_non_noise_count = int(
    np.sum(
        labels != -1
    )
)


final_cluster_summary = []


for cluster_id in final_cluster_ids:

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

        "proportion_of_non_noise_samples":
            float(
                count / final_non_noise_count
            ),

        "top_20_high_frequency_words":
            top_words
    })


visualization_json[
    "final_clustering"
] = {

    "algorithm":
        "HDBSCAN",

    "input_space":
        "UMAP-reduced embeddings",

    "best_min_cluster_size":
        int(best_param),

    "metric":
        "euclidean",

    "cluster_selection_method":
        "eom",

    "number_of_clusters_excluding_noise":
        int(
            len(
                final_cluster_ids
            )
        ),

    "total_number_of_samples":
        int(
            len(labels)
        ),

    "noise_label":
        -1,

    "noise_count":
        int(
            final_noise_count
        ),

    "noise_ratio":
        float(
            final_noise_count
            / len(labels)
        ),

    "non_noise_count":
        int(
            final_non_noise_count
        ),

    "silhouette_score_non_noise_only":
        float(
            best_score
        ),

    "cluster_counts_including_noise": [

        {
            "cluster_id":
                int(cluster_id),

            "is_noise":
                bool(
                    cluster_id == -1
                ),

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

    "clusters_excluding_noise":
        final_cluster_summary
}


# ============================================================
# 11. 文件信息写入 JSON
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
# 12. 保存 JSON
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
    f"✅ HDBSCAN 参数搜索、可视化及最终聚类信息 "
    f"已保存到：{json_output_path}"
)


# ============================================================
# 13. 输出 Excel
# ============================================================

with pd.ExcelWriter(
    output_path
) as writer:


    # --------------------------------------------------------
    # Sheet 1
    # 文本及聚类标签
    # --------------------------------------------------------

    df.to_excel(
        writer,
        index=False,
        sheet_name="Clustered Texts"
    )


    # --------------------------------------------------------
    # Sheet 2
    # 聚类数量，包括 -1 噪声
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
    f"HDBSCAN 聚类结果已保存到 "
    f"{output_path}"
)