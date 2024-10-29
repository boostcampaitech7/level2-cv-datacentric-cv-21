from map_boxes import mean_average_precision_for_boxes
from pycocotools.coco import COCO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def process_batch(
    detections, labels, num_classes=10, CONF_THRESHOLD=0.8, IOU_THRESHOLD=0.5
):
    # https://github.com/kaanakan/object_detection_confusion_matrix?tab=readme-ov-file
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        None, updates confusion matrix accordingly
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1))
    gt_classes = labels[:, 0].astype(np.int16)

    try:
        detections = detections[detections[:, 4] > CONF_THRESHOLD]
    except IndexError or TypeError:
        # detections are empty, end of process
        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            matrix[num_classes, gt_class] += 1
        return

    detection_classes = detections[:, 5].astype(np.int16)

    all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
    want_idx = np.where(all_ious > IOU_THRESHOLD)

    all_matches = [
        [want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
        for i in range(want_idx[0].shape[0])
    ]

    all_matches = np.array(all_matches)
    if all_matches.shape[0] > 0:  # if there is match
        all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

        all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

        all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

        all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

    for i, label in enumerate(labels):
        gt_class = gt_classes[i]
        if (
            all_matches.shape[0] > 0
            and all_matches[all_matches[:, 0] == i].shape[0] == 1
        ):
            detection_class = detection_classes[
                int(all_matches[all_matches[:, 0] == i, 1][0])
            ]
            matrix[detection_class, gt_class] += 1
        else:
            matrix[num_classes, gt_class] += 1

    for i, detection in enumerate(detections):
        if not all_matches.shape[0] or (
            all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0
        ):
            detection_class = detection_classes[i]
            matrix[detection_class, num_classes] += 1
    return matrix


def meanAveragePrecision(submission, args):
    classes = [
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    ]
    new_pred = []
    file_names = submission["image_id"].values.tolist()
    bboxes = submission["PredictionString"].values.tolist()

    # check variable type
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f"{file_names[i]} empty box")

    for file_name, bbox in zip(file_names, bboxes):
        boxes = np.array(str(bbox).strip().split(" "))

        # boxes - class ID confidence score xmin ymin xmax ymax
        if len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        elif isinstance(bbox, float):
            print(f"{file_name} empty box")
            continue
        else:
            raise Exception("error", "invalid box count")
        for box in boxes:
            # [file_name label_index confidence_score x_min x_max y_min y_max]
            new_pred.append(
                [
                    file_name,
                    box[0],
                    box[1],
                    float(box[2]),
                    float(box[4]),
                    float(box[3]),
                    float(box[5]),
                ]
            )

    gt = []
    coco = COCO(args.root + args.annotation)
    image_infos = coco.loadImgs(coco.getImgIds())
    for image_info in image_infos:
        annIds = coco.getAnnIds(imgIds=image_info["id"])
        annotation_info_list = coco.loadAnns(annIds)
        for annotation in annotation_info_list:
            """
            기존 bbox annotation
                x_min, y_min, width, height

            -bbox annotation -> gt 재구성-
                class_id, x_min, x_max, y_min, y_max
            """
            # [file_name label_index confidence_score x_min x_max y_min y_max]
            gt.append(
                [
                    image_info["file_name"],
                    annotation["category_id"],
                    float(annotation["bbox"][0]),
                    float(annotation["bbox"][0] + annotation["bbox"][2]),
                    float(annotation["bbox"][1]),
                    float(annotation["bbox"][1] + annotation["bbox"][3]),
                ]
            )

    mAP, average_precisions = mean_average_precision_for_boxes(
        gt, new_pred, iou_threshold=args.iou_threshold
    )

    # PLOT mAP
    x = list(map(int, list(average_precisions.keys())))
    y = list(average_precisions.values())
    y = [i[0] for i in y]

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        x=classes, y=y, palette=sns.color_palette("coolwarm", n_colors=len(x))
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    ax.axhline(mAP, color="green")
    text_properties = dict(
        x=0.9,
        y=mAP,
        s=f"mAP@50: {mAP:.4f}",
        ha="center",
        va="center",
        fontweight="bold",
        color="white",
        fontsize=12,
    )
    bbox_properties = dict(boxstyle="round", facecolor="green", alpha=1)

    # Add text with background to the plot
    ax.text(**text_properties, bbox=bbox_properties)

    for idx, patch in enumerate(ax.patches):
        ax.text(
            x=patch.get_x() + patch.get_width() / 2 - 0.03,
            y=patch.get_height() + 0.001,
            s=f"{y[idx]: .4f}",
            ha="center",
        )
    plt.ylim(0, 1)
    plt.title("mean Average Precisions")
    plt.xlabel("Classes")
    plt.ylabel("Average Precisions")
    plt.tight_layout()
    plt.savefig(args.output + "/mean_Average_Precisions_vertical.png")

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        x=y, y=classes, palette=sns.color_palette("coolwarm", n_colors=len(x))
    )

    # ax.axhline(mAP,color='darkblue')
    ax.axvline(mAP, color="green")
    text_properties = dict(
        y=0.2,
        x=mAP,
        s=f"mAP@50: {mAP:.4f}",
        ha="center",
        va="center",
        fontweight="bold",
        color="white",
        fontsize=12,
    )
    bbox_properties = dict(boxstyle="round", facecolor="green", alpha=1)

    # Add text with background to the plot
    ax.text(**text_properties, bbox=bbox_properties)

    for idx, patch in enumerate(ax.patches):
        ax.text(
            x=patch.get_width() + 0.04,
            y=patch.get_y() + patch.get_height() / 2 + 0.1,
            s=f"{y[idx]: .4f}",
            ha="center",
        )
    plt.xlim(0, 1)
    plt.title("mean Average Precisions")
    plt.xlabel("Average Precisions")
    plt.ylabel("Classes")
    plt.tight_layout()
    plt.savefig(args.output + "/mean_Average_Precisions_horizontal.png")

    return mAP


def confusion_matrix(submission, args):
    classes = [
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    ]
    num_classes = len(classes)
    matrix = np.zeros((num_classes + 1, num_classes + 1))
    coco = COCO(args.root + args.annotation)
    image_infos = coco.loadImgs(coco.getImgIds())
    for image_info in image_infos:
        # GT PART
        # GT is COCO DATASET
        annIds = coco.getAnnIds(imgIds=image_info["id"])
        gt_annotation_info_list = coco.loadAnns(annIds)
        gt = []
        for annotation in gt_annotation_info_list:
            category_id = annotation["category_id"]
            gt_bbox = [
                int(annotation["category_id"]),
                float(annotation["bbox"][0]),
                float(annotation["bbox"][1]),
                float(annotation["bbox"][0] + annotation["bbox"][2]),
                float(annotation["bbox"][1] + annotation["bbox"][3]),
            ]
            gt.append(gt_bbox)

        img_file_name = image_info["file_name"]

        out_annotation_info_list = submission[submission["image_id"] == img_file_name][
            "PredictionString"
        ].values[0]
        boxes = np.array(str(out_annotation_info_list).strip().split(" "))

        if len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        elif isinstance(out_annotation_info_list, float):
            print("empty box")
            continue
        else:
            raise Exception("error", "invalid box count")

        predicted = []
        for box in boxes:
            predicted.append(
                [
                    float(box[2]),
                    float(box[3]),
                    float(box[4]),
                    float(box[5]),
                    float(box[1]),
                    int(box[0]),
                ]
            )

        matrix += process_batch(
            np.array(predicted),
            np.array(gt),
            num_classes,
            args.conf_threshold,
            args.iou_threshold,
        )

    # plot confusion matrix
    percentages = (matrix / np.sum(matrix, axis=1)[:, np.newaxis]).round(2)
    percentages = np.where(
        np.isnan(percentages), 0, percentages
    )  # Handle division by zero

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap=sns.color_palette("light:b", as_cmap=True),
        xticklabels=classes + ["Ghost_prediction-FN"],
        yticklabels=classes + ["Ghost_prediction-FP"],
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(args.output + f"/conf_{args.conf_threshold}_confusion_matrix_cnt.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        percentages,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("light:b", as_cmap=True),
        xticklabels=classes[:10] + ["Ghost_prediction-FN"],
        yticklabels=classes[:10] + ["Ghost_prediction-FP"],
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(args.output + f"/conf_{args.conf_threshold}_confusion_matrix_per.png")

    return matrix
