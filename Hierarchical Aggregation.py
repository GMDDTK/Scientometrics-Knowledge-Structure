import os
import re
import ast
import json
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Embedding & clustering
import torch
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import pdist, squareform


# ============================================================
# 0. 配置区
# ============================================================

EXCEL_PATH = (
    r"E:\AAA＿论文\AAA博士\0各章重点实验与分析"
    r"\主题结构\聚类结果\0-Spectral Clustering.xlsx"
)

SHEET_INDEX = 2

MODEL_PATH = r"E:\NLP_data\all-roberta-large-v1"

USE_CUDA = True

BATCH_SIZE = 32

# 原来的图片
OUTPUT_DENDROGRAM = "hierarchy_dendrogram.png"

# 原来的层次结构 JSON
OUTPUT_TREE_JSON = "hierarchy_tree.json"

# 新增：专门保存 dendrogram 可视化的完整数据
OUTPUT_DENDROGRAM_JSON = (
    "hierarchy_dendrogram_visualization.json"
)


# ============================================================
# 1. JSON / Matplotlib 辅助函数
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
    获取坐标轴在 Figure 中的最终位置。
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


# ============================================================
# 2. 解析 Excel 单元格
# ============================================================

def parse_cell_to_word(cell_str: str):
    """
    稳健解析：
    ('index', 108)
    ('index',108)
    ('word', 12)
    index, 108
    index

    返回词，解析失败则返回 None。
    """

    if not isinstance(cell_str, str):
        cell_str = str(cell_str)

    s = cell_str.strip()

    if not s:
        return None

    # --------------------------------------------------------
    # 优先尝试 Python literal
    # --------------------------------------------------------

    try:

        val = ast.literal_eval(s)

        if (
            isinstance(val, tuple)
            and len(val) >= 1
        ):
            return str(val[0])

    except Exception:
        pass


    # --------------------------------------------------------
    # 正则 fallback
    # --------------------------------------------------------

    m = re.search(
        r"['\"]([^'\"]+)['\"]\s*,\s*\d+",
        s
    )

    if m:
        return m.group(1)


    m2 = re.search(
        r"^([^,()\[\]\"']+)\s*(?:,|\)|$)",
        s
    )

    if m2:
        return m2.group(1).strip()

    return None


# ============================================================
# 3. 读取 Excel
#    同时记录每个聚类来自 Excel 的哪一行
# ============================================================

def read_clusters_from_excel(
        path: str,
        sheet_idx: int
):

    df = pd.read_excel(
        path,
        sheet_name=sheet_idx,
        header=None,
        dtype=str
    )

    clusters = []

    source_row_indices = []


    for row_i, row in df.iterrows():

        words = []

        for cell in row.dropna().tolist():

            w = parse_cell_to_word(
                str(cell)
            )

            if w:
                words.append(w)


        # ----------------------------------------------------
        # 去重并保持原始顺序
        # ----------------------------------------------------

        seen = set()

        uniq_words = []

        for w in words:

            if w not in seen:

                seen.add(w)

                uniq_words.append(w)


        # ----------------------------------------------------
        # 空行跳过
        # ----------------------------------------------------

        if len(uniq_words) == 0:
            continue


        clusters.append(
            uniq_words
        )

        source_row_indices.append(
            int(row_i)
        )


    return (
        clusters,
        source_row_indices
    )


# ============================================================
# 4. 读取 Excel
# ============================================================

print(
    "读取并解析 Excel ..."
)


if not os.path.isfile(
    EXCEL_PATH
):

    raise FileNotFoundError(
        f"找不到 Excel 文件："
        f"{EXCEL_PATH}"
    )


(
    clusters_words,
    cluster_source_rows
) = read_clusters_from_excel(
    EXCEL_PATH,
    SHEET_INDEX
)


n_clusters = len(
    clusters_words
)


print(
    f"解析到 {n_clusters} 个簇（非空行）。"
)


# ============================================================
# 5. 打印每个簇的词
# ============================================================

print(
    "\n每个簇的高频词集合（仅词，不含频次）：\n"
)


for i, words in enumerate(
    clusters_words
):

    display_words = [
        word
        for word in words
        if not word.isdigit()
    ]

    print(
        f"{{{', '.join(display_words)}}}"
    )


# ============================================================
# 6. 加载 SentenceTransformer
# ============================================================

device = "cpu"

if (
    USE_CUDA
    and torch.cuda.is_available()
):

    device = "cuda"


print(
    f"\n尝试加载 embedding 模型"
    f"（路径：{MODEL_PATH}），"
    f"device={device} ..."
)


try:

    model = SentenceTransformer(
        MODEL_PATH,
        device=device
    )

except Exception as e:

    raise RuntimeError(
        f"加载模型失败：{e}\n"
        f"请确认 MODEL_PATH 指向 "
        f"sentence-transformers 可识别的模型目录。"
    )


# ============================================================
# 7. 构建全局唯一词表
# ============================================================

all_words = []

for wlist in clusters_words:
    all_words.extend(wlist)


# 保持首次出现顺序
unique_words = list(
    dict.fromkeys(
        all_words
    )
)


print(
    f"\n共有 {len(unique_words)} 个不重复词，"
    f"将对它们做 embedding（分批处理）。"
)


# ============================================================
# 8. Batch Encoding
# ============================================================

def batch_encode(
        model,
        items: List[str],
        batch_size: int = 64
):

    embeddings_list = []


    for i in tqdm(
        range(
            0,
            len(items),
            batch_size
        ),
        desc="Encoding batches"
    ):

        batch = items[
            i:i + batch_size
        ]


        emb = model.encode(

            batch,

            convert_to_numpy=True,

            batch_size=len(batch),

            show_progress_bar=False
        )


        embeddings_list.append(
            emb
        )


    return np.vstack(
        embeddings_list
    )


# ============================================================
# 9. 编码词向量
# ============================================================

word_embeddings = batch_encode(
    model,
    unique_words,
    batch_size=BATCH_SIZE
)


word2emb = {

    w:
        word_embeddings[i]

    for i, w
    in enumerate(
        unique_words
    )
}


emb_dim = int(
    word_embeddings.shape[1]
)


print(
    f"词向量维度: {emb_dim}"
)


# ============================================================
# 10. 生成每个簇的 centroid
# ============================================================

cluster_embeddings = []


for idx, wlist in enumerate(
    clusters_words
):

    embs = [

        word2emb[w]

        for w in wlist

        if w in word2emb
    ]


    if len(embs) == 0:

        fallback = model.encode(

            " ".join(wlist),

            convert_to_numpy=True
        )

        centroid = fallback

    else:

        centroid = np.mean(
            np.vstack(embs),
            axis=0
        )


    cluster_embeddings.append(
        centroid
    )


cluster_embeddings = np.vstack(
    cluster_embeddings
)


print(
    f"得到 {cluster_embeddings.shape[0]} "
    f"个簇向量。"
)


# ============================================================
# 11. 层次聚类
# ============================================================

print(
    "\n进行层次聚类"
    "（average linkage, cosine distance）..."
)


# ------------------------------------------------------------
# 11.1 两两 cosine distance
# ------------------------------------------------------------

dist_condensed = pdist(

    cluster_embeddings,

    metric="cosine"
)


# 完整距离矩阵
distance_matrix = squareform(
    dist_condensed
)


# ------------------------------------------------------------
# 11.2 Average Linkage
# ------------------------------------------------------------

Z = linkage(

    dist_condensed,

    method="average"
)


# ============================================================
# 12. 绘制 Dendrogram
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 8)
)


labels = [

    f"T{i:02d}"

    for i in range(
        n_clusters
    )
]


dn = dendrogram(

    Z,

    labels=labels,

    leaf_rotation=45,

    leaf_font_size=8,

    color_threshold=None,

    ax=ax
)


# 原代码没有总标题
# ax.set_title(...)


ax.set_xlabel(
    "Topic Label"
)


ax.set_ylabel(
    "Cosine Distance"
)


plt.tight_layout()


# ------------------------------------------------------------
# 强制计算最终坐标轴、刻度、标签位置
# ------------------------------------------------------------

fig.canvas.draw()


# ============================================================
# 13. 保存 Dendrogram 图片
# ============================================================

fig.savefig(

    OUTPUT_DENDROGRAM,

    dpi=300
)


print(
    f"已保存 dendrogram 到："
    f"{OUTPUT_DENDROGRAM}"
)


# ============================================================
# 14. 构建 dendrogram 完整可视化 JSON
# ============================================================


# ------------------------------------------------------------
# 14.1 每个叶节点基础信息
# ------------------------------------------------------------

leaf_metadata = []


for cluster_id in range(
    n_clusters
):

    leaf_metadata.append({

        "cluster_id":
            int(cluster_id),

        "topic_label":
            str(
                labels[
                    cluster_id
                ]
            ),

        "source_excel_row_zero_based":
            int(
                cluster_source_rows[
                    cluster_id
                ]
            ),

        "source_excel_row_one_based":
            int(
                cluster_source_rows[
                    cluster_id
                ]
                + 1
            ),

        "number_of_words":
            int(
                len(
                    clusters_words[
                        cluster_id
                    ]
                )
            ),

        "words": [
            str(w)
            for w
            in clusters_words[
                cluster_id
            ]
        ],

        "representative_first_6_words": [

            str(w)

            for w
            in clusters_words[
                cluster_id
            ][:6]
        ]
    })


# ------------------------------------------------------------
# 14.2 dendrogram 最终显示叶节点顺序
# ------------------------------------------------------------

displayed_leaf_order = []


# dn["leaves"] 是原始 cluster id 的显示顺序
for display_index, cluster_id in enumerate(
    dn["leaves"]
):

    displayed_leaf_order.append({

        "display_index_zero_based":
            int(display_index),

        "display_index_one_based":
            int(display_index + 1),

        "cluster_id":
            int(cluster_id),

        "topic_label":
            str(
                labels[
                    cluster_id
                ]
            ),

        # scipy dendrogram 默认叶节点中心通常为
        # 5, 15, 25, ...
        # 这里同时从 Matplotlib 实际 tick 中再取
        "expected_dendrogram_x_position":
            float(
                5 + 10 * display_index
            ),

        "words": [
            str(w)
            for w
            in clusters_words[
                cluster_id
            ]
        ]
    })


# ------------------------------------------------------------
# 14.3 每一个 dendrogram U 型 branch
# ------------------------------------------------------------

branches = []


for i, (
    icoord,
    dcoord,
    color
) in enumerate(
    zip(
        dn["icoord"],
        dn["dcoord"],
        dn["color_list"]
    )
):

    branches.append({

        "branch_index":
            int(i + 1),

        "color":
            str(color),

        "x_coordinates": [
            float(v)
            for v in icoord
        ],

        "y_coordinates": [
            float(v)
            for v in dcoord
        ],

        "points": [

            {
                "x":
                    float(x_val),

                "y":
                    float(y_val)
            }

            for x_val, y_val
            in zip(
                icoord,
                dcoord
            )
        ],

        # U 型分支最高点就是该次可视化合并的高度
        "merge_height":
            float(
                max(
                    dcoord
                )
            )
    })


# ------------------------------------------------------------
# 14.4 Linkage Matrix
#
# Z 每一行:
# [left_child, right_child, distance, sample_count]
# ------------------------------------------------------------

linkage_steps = []


for i, row in enumerate(
    Z
):

    left_child = int(
        row[0]
    )

    right_child = int(
        row[1]
    )

    merge_distance = float(
        row[2]
    )

    member_count = int(
        row[3]
    )

    new_node_id = int(
        n_clusters + i
    )


    linkage_steps.append({

        "merge_step":
            int(i + 1),

        "new_node_id":
            new_node_id,

        "left_child_id":
            left_child,

        "right_child_id":
            right_child,

        "cosine_distance":
            merge_distance,

        "number_of_original_clusters_in_new_node":
            member_count
    })


# ------------------------------------------------------------
# 14.5 两两 cosine distance
# ------------------------------------------------------------

pairwise_distances = []


for i in range(
    n_clusters
):

    for j in range(
        i + 1,
        n_clusters
    ):

        pairwise_distances.append({

            "cluster_1_id":
                int(i),

            "cluster_1_label":
                str(
                    labels[i]
                ),

            "cluster_2_id":
                int(j),

            "cluster_2_label":
                str(
                    labels[j]
                ),

            "cosine_distance":
                float(
                    distance_matrix[
                        i,
                        j
                    ]
                )
        })


# ------------------------------------------------------------
# 14.6 实际 X Tick
# ------------------------------------------------------------

actual_x_ticks = (
    get_tick_information(
        ax,
        "x"
    )
)


# 为实际 tick 补充对应 cluster
actual_leaf_ticks = []


for i, tick in enumerate(
    actual_x_ticks
):

    entry = {

        "position":
            float(
                tick["position"]
            ),

        "displayed_label":
            str(
                tick["label"]
            )
    }


    # 若这个 tick 确实属于 dendrogram leaf
    if i < len(
        dn["leaves"]
    ):

        cluster_id = int(
            dn["leaves"][i]
        )

        entry.update({

            "cluster_id":
                cluster_id,

            "topic_label":
                labels[
                    cluster_id
                ],

            "words": [
                str(w)
                for w
                in clusters_words[
                    cluster_id
                ]
            ]
        })


    actual_leaf_ticks.append(
        entry
    )


# ============================================================
# 15. 完整 Dendrogram JSON
# ============================================================

dendrogram_visualization = {


    # ========================================================
    # Figure
    # ========================================================

    "figure": {

        "figure_type":
            "hierarchical_clustering_dendrogram",

        "purpose":
            "Hierarchical organization of topic clusters based on semantic similarity",

        "figure_size_inches": {

            "width":
                float(
                    fig.get_size_inches()[0]
                ),

            "height":
                float(
                    fig.get_size_inches()[1]
                )
        },

        "dpi_saved":
            300,

        "image_output":
            OUTPUT_DENDROGRAM,

        "tree_json_output":
            OUTPUT_TREE_JSON,

        "visualization_json_output":
            OUTPUT_DENDROGRAM_JSON,

        "number_of_leaf_clusters":
            int(
                n_clusters
            )
    },


    # ========================================================
    # 数据来源
    # ========================================================

    "source": {

        "excel_path":
            EXCEL_PATH,

        "sheet_index_zero_based":
            int(
                SHEET_INDEX
            ),

        "model_path":
            MODEL_PATH,

        "device_used":
            device,

        "use_cuda_requested":
            bool(
                USE_CUDA
            ),

        "batch_size":
            int(
                BATCH_SIZE
            )
    },


    # ========================================================
    # Embedding 信息
    # ========================================================

    "embedding": {

        "method":
            "SentenceTransformer",

        "model_path":
            MODEL_PATH,

        "number_of_unique_words":
            int(
                len(
                    unique_words
                )
            ),

        "word_embedding_dimension":
            int(
                emb_dim
            ),

        "number_of_cluster_centroids":
            int(
                cluster_embeddings.shape[0]
            ),

        "cluster_embedding_dimension":
            int(
                cluster_embeddings.shape[1]
            ),

        "cluster_vector_method":
            "Mean of word embeddings within each cluster",

        "unique_words_in_encoding_order": [
            str(w)
            for w in unique_words
        ]
    },


    # ========================================================
    # 层次聚类参数
    # ========================================================

    "hierarchical_clustering": {

        "distance_metric":
            "cosine",

        "linkage_method":
            "average",

        "number_of_input_clusters":
            int(
                n_clusters
            ),

        "number_of_linkage_merges":
            int(
                Z.shape[0]
            ),

        "root_merge_distance":
            float(
                Z[-1, 2]
            ),

        "linkage_matrix_shape": [
            int(v)
            for v in Z.shape
        ],

        "linkage_steps":
            linkage_steps
    },


    # ========================================================
    # Dendrogram 参数
    # ========================================================

    "dendrogram_settings": {

        "leaf_rotation_degrees":
            45,

        "leaf_font_size":
            8,

        "color_threshold":
            None,

        "leaf_label_format":
            "T{cluster_id:02d}",

        "xlabel":
            ax.get_xlabel(),

        "ylabel":
            ax.get_ylabel(),

        "title":
            ax.get_title()
    },


    # ========================================================
    # 坐标轴信息
    # ========================================================

    "axes": {

        "axes_position_in_figure":
            get_axes_position(
                ax
            ),


        "x_axis": {

            "label":
                ax.get_xlabel(),

            "semantic_variable":
                "Topic Cluster Label",

            "axis_limits": {

                "min":
                    float(
                        ax.get_xlim()[0]
                    ),

                "max":
                    float(
                        ax.get_xlim()[1]
                    )
            },

            "displayed_ticks":
                actual_leaf_ticks,

            "tick_rotation_degrees":
                45,

            "tick_font_size":
                8
        },


        "y_axis": {

            "label":
                ax.get_ylabel(),

            "semantic_variable":
                "Cosine Distance / Merge Height",

            "axis_limits": {

                "min":
                    float(
                        ax.get_ylim()[0]
                    ),

                "max":
                    float(
                        ax.get_ylim()[1]
                    )
            },

            "displayed_ticks":
                get_tick_information(
                    ax,
                    "y"
                )
        }
    },


    # ========================================================
    # 叶节点原始信息
    # ========================================================

    "leaf_clusters":
        leaf_metadata,


    # ========================================================
    # Dendrogram 最终排序
    # ========================================================

    "displayed_leaf_order": {

        "leaf_ids_in_display_order": [
            int(v)
            for v in dn["leaves"]
        ],

        "labels_in_display_order": [
            str(v)
            for v in dn["ivl"]
        ],

        "details":
            displayed_leaf_order
    },


    # ========================================================
    # 每一个可视化 branch
    # ========================================================

    "dendrogram_branches": {

        "number_of_branches":
            int(
                len(
                    branches
                )
            ),

        "branches":
            branches,

        "branch_colors_in_order": [
            str(v)
            for v in dn[
                "color_list"
            ]
        ],

        "leaf_colors_in_display_order": [
            str(v)
            for v in dn.get(
                "leaves_color_list",
                []
            )
        ]
    },


    # ========================================================
    # 两两 cosine distance
    # ========================================================

    "pairwise_cosine_distances": {

        "number_of_pairs":
            int(
                len(
                    pairwise_distances
                )
            ),

        "pairs":
            pairwise_distances
    },


    # ========================================================
    # 完整距离矩阵
    # ========================================================

    "cosine_distance_matrix": {

        "row_and_column_order": [
            str(v)
            for v in labels
        ],

        "cluster_id_order": [
            int(v)
            for v in range(
                n_clusters
            )
        ],

        "matrix": [

            [
                float(value)
                for value in row
            ]

            for row
            in distance_matrix
        ]
    }
}


# ============================================================
# 16. 保存 Dendrogram 可视化 JSON
# ============================================================

with open(
    OUTPUT_DENDROGRAM_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dendrogram_visualization,
        f,
        ensure_ascii=False,
        indent=2,
        allow_nan=False
    )


print(
    f"已保存 dendrogram 完整可视化 JSON："
    f"{OUTPUT_DENDROGRAM_JSON}"
)


# ============================================================
# 17. 显示图
# ============================================================

plt.show()


# ============================================================
# 18. 树结构辅助函数
# ============================================================

def get_leaves_ids(node):

    if node.is_leaf():

        return [
            node.id
        ]

    else:

        left = node.get_left()
        right = node.get_right()

        return (
            get_leaves_ids(left)
            +
            get_leaves_ids(right)
        )


# ============================================================
# 19. 构造层次树字典
# ============================================================

def node_to_dict(
        node,
        depth=0,
        top_k_rep=5
):

    if node.is_leaf():

        leaf_id = node.id

        words = clusters_words[
            leaf_id
        ]

        return {

            "name":
                f"簇_{leaf_id:02d}",

            "topic_label":
                f"T{leaf_id:02d}",

            "leaf":
                True,

            "members_indices": [
                int(
                    leaf_id
                )
            ],

            "source_excel_row_zero_based":
                int(
                    cluster_source_rows[
                        leaf_id
                    ]
                ),

            "representative":
                words[
                    :top_k_rep
                ],

            "words":
                words
        }


    else:

        left = node.get_left()

        right = node.get_right()


        left_d = node_to_dict(
            left,
            depth + 1,
            top_k_rep
        )


        right_d = node_to_dict(
            right,
            depth + 1,
            top_k_rep
        )


        leaf_ids = get_leaves_ids(
            node
        )


        merged_words = []


        for lid in leaf_ids:

            merged_words.extend(
                clusters_words[
                    lid
                ]
            )


        c = Counter(
            merged_words
        )


        rep = [

            w
            for w, _
            in c.most_common(
                top_k_rep
            )
        ]


        return {

            "name":
                f"node_{node.id}",

            "node_id":
                int(
                    node.id
                ),

            "leaf":
                False,

            "merge_distance":
                float(
                    node.dist
                ),

            "number_of_leaf_members":
                int(
                    node.count
                ),

            "members_indices": [
                int(v)
                for v in leaf_ids
            ],

            "member_topic_labels": [

                f"T{int(v):02d}"

                for v in leaf_ids
            ],

            "representative":
                rep,

            "children": [
                left_d,
                right_d
            ]
        }


# ============================================================
# 20. 构造树
# ============================================================

root, node_list = to_tree(
    Z,
    rd=True
)


hierarchy = node_to_dict(
    root,
    top_k_rep=6
)


# 最外层根节点命名为“科学计量学”
hierarchy["name"] = (
    "科学计量学"
)


# 补充根节点方法信息
hierarchy[
    "hierarchical_method"
] = {

    "distance_metric":
        "cosine",

    "linkage_method":
        "average",

    "number_of_leaf_clusters":
        int(
            n_clusters
        ),

    "root_merge_distance":
        float(
            root.dist
        )
}


# ============================================================
# 21. 保存树 JSON
# ============================================================

with open(
    OUTPUT_TREE_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        hierarchy,
        f,
        ensure_ascii=False,
        indent=2,
        allow_nan=False
    )


print(
    f"已保存层次结构 JSON："
    f"{OUTPUT_TREE_JSON}"
)


# ============================================================
# 22. 打印文本树
# ============================================================

def print_tree_dict(
        node_dict,
        indent=0,
        max_children_show=10
):

    pad = (
        "  "
        * indent
    )

    name = node_dict.get(
        "name",
        ""
    )

    rep = node_dict.get(
        "representative",
        []
    )

    members_count = len(
        node_dict.get(
            "members_indices",
            []
        )
    )


    if node_dict.get(
        "leaf",
        False
    ):

        words = node_dict.get(
            "words",
            []
        )

        print(

            f"{pad}- {name} "
            f"(leaf, "
            f"idx="
            f"{node_dict['members_indices'][0]}), "
            f"words: "
            f"{', '.join(words)}"
        )


    else:

        print(

            f"{pad}+ {name} "
            f"(members: "
            f"{members_count}), "
            f"rep_words: "
            f"{', '.join(rep)}"
        )


        for child in node_dict.get(
            "children",
            []
        ):

            print_tree_dict(
                child,
                indent=indent + 1
            )


# ============================================================
# 23. 输出树
# ============================================================

print(
    "\n层次结构（文本展示，"
    "根节点为“科学计量学”）：\n"
)


print_tree_dict(
    hierarchy
)


# ============================================================
# 24. 最终提示
# ============================================================

print(
    "\n脚本运行结束。"
)


print(
    " - cluster words 已打印在控制台；"
)


print(
    f" - dendrogram 图片保存在: "
    f"{os.path.abspath(OUTPUT_DENDROGRAM)}"
)


print(
    f" - 层次结构 JSON 保存为: "
    f"{os.path.abspath(OUTPUT_TREE_JSON)}"
)


print(
    f" - dendrogram 完整可视化 JSON 保存为: "
    f"{os.path.abspath(OUTPUT_DENDROGRAM_JSON)}"
)