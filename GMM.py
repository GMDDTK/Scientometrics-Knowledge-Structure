import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# 0. 文件路径
# ============================================================

embedding_file_path = r"embeddings.npy"
text_file_path = r"all.txt"

output_path = r"GMM聚类结果.xlsx"

# 新增：完整可视化和聚类信息 JSON
json_output_path = r"GMM聚类评估可视化完整数据.json"


# ============================================================
# 1. 读取保存的语义嵌入文件
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
    获取 Matplotlib 最终实际显示的刻度位置与刻度标签。
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
    获取子图在整个 Figure 中的最终位置。
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
    获取折线在 Matplotlib 中的实际绘图属性。
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
            ),

        "alpha":
            (
                None
                if line.get_alpha() is None
                else float(line.get_alpha())
            )
    }


# ============================================================
# 3. 选择最佳聚类数
# ============================================================

def choose_best_k_gmm(
        embeddings,
        min_k=80,
        max_k=130,
        step=1
):

    # 转成 list，便于 JSON 保存以及后面索引
    k_values = list(
        range(
            min_k,
            max_k + 1,
            step
        )
    )

    silhouette_scores = []
    bic_scores = []
    aic_scores = []

    # 额外保存每一个 k 的完整结果
    evaluation_points = []


    # ========================================================
    # 3.1 遍历所有 K
    # ========================================================

    for k in k_values:

        print(
            f"Testing k={k}"
        )


        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0
        )


        labels = gmm.fit_predict(
            embeddings
        )


        # ----------------------------------------------------
        # BIC / AIC
        # ----------------------------------------------------

        bic = float(
            gmm.bic(
                embeddings
            )
        )

        aic = float(
            gmm.aic(
                embeddings
            )
        )


        bic_scores.append(
            bic
        )

        aic_scores.append(
            aic
        )


        # ----------------------------------------------------
        # 当前实际簇数量
        # ----------------------------------------------------

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        actual_cluster_count = int(
            len(unique_labels)
        )


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
                f"silhouette={score:.4f}, "
                f"BIC={bic:.2f}, "
                f"AIC={aic:.2f}"
            )


        except Exception as e:

            score = -1.0

            silhouette_scores.append(
                score
            )

            silhouette_status = "failed"
            silhouette_error = str(e)


        # ----------------------------------------------------
        # 当前 k 的簇大小
        # ----------------------------------------------------

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
        # 当前 K 的完整 GMM 信息
        # ----------------------------------------------------

        evaluation_points.append({

            "k":
                int(k),

            "requested_number_of_components":
                int(k),

            "actual_number_of_nonempty_clusters":
                int(actual_cluster_count),

            "silhouette_score":
                float(score),

            "silhouette_status":
                silhouette_status,

            "silhouette_error":
                silhouette_error,

            "bic":
                float(bic),

            "aic":
                float(aic),

            "converged":
                bool(
                    gmm.converged_
                ),

            "n_iter":
                int(
                    gmm.n_iter_
                ),

            "lower_bound":
                float(
                    gmm.lower_bound_
                ),

            "cluster_sizes":
                cluster_sizes
        })


    # ========================================================
    # 3.2 三种标准分别确定最佳 K
    # ========================================================

    best_silhouette_index = int(
        np.argmax(
            silhouette_scores
        )
    )

    best_bic_index = int(
        np.argmin(
            bic_scores
        )
    )

    best_aic_index = int(
        np.argmin(
            aic_scores
        )
    )


    best_k = int(
        k_values[
            best_silhouette_index
        ]
    )

    best_k_bic = int(
        k_values[
            best_bic_index
        ]
    )

    best_k_aic = int(
        k_values[
            best_aic_index
        ]
    )


    # ========================================================
    # 3.3 绘制三张评估曲线
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    ax1, ax2, ax3 = axes


    # --------------------------------------------------------
    # Silhouette
    # --------------------------------------------------------

    line_silhouette, = ax1.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )

    ax1.set_xlabel(
        "Number of Clusters (k)"
    )

    ax1.set_ylabel(
        "Silhouette Score"
    )

    ax1.set_title(
        "Silhouette Scores"
    )


    # --------------------------------------------------------
    # BIC
    # --------------------------------------------------------

    line_bic, = ax2.plot(
        k_values,
        bic_scores,
        marker="o",
        color="orange"
    )

    ax2.set_xlabel(
        "Number of Clusters (k)"
    )

    ax2.set_ylabel(
        "BIC"
    )

    ax2.set_title(
        "Bayesian Information Criterion"
    )


    # --------------------------------------------------------
    # AIC
    # --------------------------------------------------------

    line_aic, = ax3.plot(
        k_values,
        aic_scores,
        marker="o",
        color="green"
    )

    ax3.set_xlabel(
        "Number of Clusters (k)"
    )

    ax3.set_ylabel(
        "AIC"
    )

    ax3.set_title(
        "Akaike Information Criterion"
    )


    # ========================================================
    # 3.4 Final Layout
    # ========================================================

    plt.tight_layout()

    # 强制 Matplotlib 完成最终坐标、
    # Tick 和 layout 计算
    fig.canvas.draw()


    # ========================================================
    # 3.5 构建完整 JSON
    # ========================================================

    visualization_json = {

        # ====================================================
        # Figure
        # ====================================================

        "figure": {

            "figure_type":
                "three-panel line chart",

            "purpose":
                "GMM cluster-number evaluation using Silhouette Score, BIC and AIC",

            "layout": {
                "rows": 1,
                "columns": 3,
                "number_of_subplots": 3
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
        # GMM 参数
        # ====================================================

        "gmm_settings": {

            "algorithm":
                "GaussianMixture",

            "covariance_type":
                "full",

            "random_state":
                0,

            "minimum_k":
                int(min_k),

            "maximum_k":
                int(max_k),

            "step":
                int(step),

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

            "silhouette_scores": [
                float(v)
                for v in silhouette_scores
            ],

            "bic_scores": [
                float(v)
                for v in bic_scores
            ],

            "aic_scores": [
                float(v)
                for v in aic_scores
            ],

            "points":
                evaluation_points
        },


        # ====================================================
        # 最佳 K 汇总
        # ====================================================

        "model_selection": {

            "final_selection_rule":
                "Maximum Silhouette Score",

            "selected_optimal_k":
                int(best_k),


            "silhouette_criterion": {

                "rule":
                    "Maximum Silhouette Score",

                "best_point_index_zero_based":
                    int(
                        best_silhouette_index
                    ),

                "best_k":
                    int(best_k),

                "best_score":
                    float(
                        silhouette_scores[
                            best_silhouette_index
                        ]
                    ),

                "bic_at_this_k":
                    float(
                        bic_scores[
                            best_silhouette_index
                        ]
                    ),

                "aic_at_this_k":
                    float(
                        aic_scores[
                            best_silhouette_index
                        ]
                    )
            },


            "bic_criterion": {

                "rule":
                    "Minimum BIC",

                "best_point_index_zero_based":
                    int(
                        best_bic_index
                    ),

                "best_k":
                    int(
                        best_k_bic
                    ),

                "minimum_bic":
                    float(
                        bic_scores[
                            best_bic_index
                        ]
                    ),

                "silhouette_at_this_k":
                    float(
                        silhouette_scores[
                            best_bic_index
                        ]
                    ),

                "aic_at_this_k":
                    float(
                        aic_scores[
                            best_bic_index
                        ]
                    )
            },


            "aic_criterion": {

                "rule":
                    "Minimum AIC",

                "best_point_index_zero_based":
                    int(
                        best_aic_index
                    ),

                "best_k":
                    int(
                        best_k_aic
                    ),

                "minimum_aic":
                    float(
                        aic_scores[
                            best_aic_index
                        ]
                    ),

                "silhouette_at_this_k":
                    float(
                        silhouette_scores[
                            best_aic_index
                        ]
                    ),

                "bic_at_this_k":
                    float(
                        bic_scores[
                            best_aic_index
                        ]
                    )
            }
        },


        # ====================================================
        # Subplot 1: Silhouette
        # ====================================================

        "subplot_1_silhouette": {

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
                    "Number of GMM components",

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
                    "Silhouette Score",

                "data_values": [
                    float(v)
                    for v in silhouette_scores
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

                    "silhouette_score":
                        float(score),

                    "is_best_by_silhouette":
                        bool(
                            i
                            == best_silhouette_index
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
                False
        },


        # ====================================================
        # Subplot 2: BIC
        # ====================================================

        "subplot_2_bic": {

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
                    "Number of GMM components",

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
                    "Bayesian Information Criterion",

                "data_values": [
                    float(v)
                    for v in bic_scores
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

                    "bic":
                        float(bic),

                    "is_minimum_bic":
                        bool(
                            i
                            == best_bic_index
                        )
                }

                for i, (k, bic)
                in enumerate(
                    zip(
                        k_values,
                        bic_scores
                    )
                )
            ],


            "line_properties":
                get_line_information(
                    line_bic
                ),

            "grid_displayed":
                False
        },


        # ====================================================
        # Subplot 3: AIC
        # ====================================================

        "subplot_3_aic": {

            "subplot_index":
                3,

            "grid_position": {
                "row": 1,
                "column": 3
            },

            "chart_type":
                "line_chart_with_markers",

            "title":
                ax3.get_title(),

            "axes_position":
                get_axes_position(
                    ax3
                ),


            "x_axis": {

                "label":
                    ax3.get_xlabel(),

                "semantic_variable":
                    "Number of GMM components",

                "data_values": [
                    int(v)
                    for v in k_values
                ],

                "displayed_ticks":
                    get_tick_information(
                        ax3,
                        "x"
                    ),

                "axis_limits": {
                    "min":
                        float(
                            ax3.get_xlim()[0]
                        ),

                    "max":
                        float(
                            ax3.get_xlim()[1]
                        )
                }
            },


            "y_axis": {

                "label":
                    ax3.get_ylabel(),

                "semantic_variable":
                    "Akaike Information Criterion",

                "data_values": [
                    float(v)
                    for v in aic_scores
                ],

                "displayed_ticks":
                    get_tick_information(
                        ax3,
                        "y"
                    ),

                "axis_limits": {
                    "min":
                        float(
                            ax3.get_ylim()[0]
                        ),

                    "max":
                        float(
                            ax3.get_ylim()[1]
                        )
                }
            },


            "data_points": [

                {
                    "point_index":
                        int(i + 1),

                    "k":
                        int(k),

                    "aic":
                        float(aic),

                    "is_minimum_aic":
                        bool(
                            i
                            == best_aic_index
                        )
                }

                for i, (k, aic)
                in enumerate(
                    zip(
                        k_values,
                        aic_scores
                    )
                )
            ],


            "line_properties":
                get_line_information(
                    line_aic
                ),

            "grid_displayed":
                False
        }
    }


    # ========================================================
    # 3.6 显示图片
    # ========================================================

    plt.show()


    return (
        best_k,
        visualization_json
    )


# ============================================================
# 4. 运行 GMM 参数搜索
# ============================================================

optimal_k, visualization_json = choose_best_k_gmm(
    embeddings,
    min_k=5,
    max_k=200,
    step=5
)


print(
    f"Optimal number of clusters (GMM): "
    f"{optimal_k}"
)


# ============================================================
# 5. 使用最佳 K 运行最终 GMM
# ============================================================

gmm = GaussianMixture(
    n_components=optimal_k,
    covariance_type="full",
    random_state=0
)


labels = gmm.fit_predict(
    embeddings
)


# ============================================================
# 6. 读取文本数据
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
# 7. 统计各个类的样本数
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
# 8. 高频词
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

    # GMM 正常不会产生 -1，
    # 这里保留与你原始代码一致的逻辑
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
# 9. 最终 GMM 聚类摘要
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
# 10. 最终模型的统计信息写入 JSON
# ============================================================

visualization_json[
    "final_clustering"
] = {

    "algorithm":
        "GaussianMixture",

    "selected_by":
        "Maximum Silhouette Score",

    "n_components":
        int(optimal_k),

    "covariance_type":
        "full",

    "random_state":
        0,

    "number_of_samples":
        int(
            len(labels)
        ),

    "actual_number_of_nonempty_clusters":
        int(
            len(
                np.unique(
                    labels
                )
            )
        ),

    "converged":
        bool(
            gmm.converged_
        ),

    "n_iter":
        int(
            gmm.n_iter_
        ),

    "lower_bound":
        float(
            gmm.lower_bound_
        ),

    "final_bic":
        float(
            gmm.bic(
                embeddings
            )
        ),

    "final_aic":
        float(
            gmm.aic(
                embeddings
            )
        ),

    "final_silhouette_score":
        float(
            silhouette_score(
                embeddings,
                labels
            )
        ),

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

    "clusters":
        final_cluster_summary
}


# ============================================================
# 11. GMM mixture 权重写入 JSON
# ============================================================

visualization_json[
    "final_clustering"
][
    "mixture_weights"
] = [

    {
        "component_id":
            int(i),

        "weight":
            float(weight)
    }

    for i, weight
    in enumerate(
        gmm.weights_
    )
]


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
    f"✅ GMM 参数搜索、可视化及最终聚类信息 "
    f"已保存到：{json_output_path}"
)


# ============================================================
# 14. 输出到 Excel
# ============================================================

with pd.ExcelWriter(
    output_path
) as writer:


    # --------------------------------------------------------
    # Sheet 1
    # --------------------------------------------------------

    df.to_excel(
        writer,
        index=False,
        sheet_name="Clustered Texts"
    )


    # --------------------------------------------------------
    # Sheet 2
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
    f"GMM 聚类结果已保存到 "
    f"{output_path}"
)