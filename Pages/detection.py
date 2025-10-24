import glob
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
import torch
import os

# 禁用 PyTorch 的“仅权重加载”模式，确保模型加载兼容性
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

"""--------------------------------------picture uploader-------------------------------------"""

def uploaded_img_viewer(upload_title, image_types, accept_multiple_files):
    """
    图片查看器组件：上传、预览、切换图片

    :param upload_title: 上传组件标题
    :param image_types: 允许上传的图片类型列表，如 ['png', 'jpg']
    :param accept_multiple_files: 是否允许多文件上传
    :return: 上传的图片文件列表（可能为空）
    """
    # 初始化会话状态
    if 'img_index' not in st.session_state:
        st.session_state.img_index = 0
    if 'uploaded_img' not in st.session_state:
        st.session_state.uploaded_img = []

    # 上传组件
    uploaded_img = st.file_uploader(
        upload_title,
        image_types,
        accept_multiple_files
    )

    # 新上传文件触发重置索引
    if uploaded_img and uploaded_img != st.session_state.uploaded_img:
        st.session_state.uploaded_img = [uploaded_img] if not isinstance(uploaded_img, list) else uploaded_img
        st.session_state.img_index = 0

    # 显示当前图片
    if st.session_state.uploaded_img:
        img = Image.open(st.session_state.uploaded_img[st.session_state.img_index])
        st.image(
            img,
            caption=f"{st.session_state.img_index + 1}/{len(st.session_state.uploaded_img)} {img.filename}"
        )
        st.write(st.session_state.uploaded_img[st.session_state.img_index].name)

    # 上一张/下一张按钮
    column_l, column_m = st.columns([1, 1])
    with column_l:
        if st.button("◀ 上一张", key="prev"):
            if st.session_state.img_index > 0:
                st.session_state.img_index -= 1
            else:
                st.warning('This is the first image!')
    with column_m:
        if st.button("下一张▶", key="next"):
            if st.session_state.img_index < len(st.session_state.uploaded_img) - 1:
                st.session_state.img_index += 1
            else:
                st.warning('This is the last image!')

    # 返回上传的文件列表
    if st.session_state.uploaded_img:
        return st.session_state.uploaded_img

"""--------------------------------------detector-------------------------------------"""

def model_param_selector():
    """
    侧边栏选择检测模型及超参数

    :return: dict，包含 model、conf_threshold、iou_threshold
    """
    st.sidebar.markdown("\n")
    model = st.sidebar.selectbox(
        label='choose expected model',
        options=("yolo", "待添加...")
    )
    model_params = dict()
    if model == 'yolo':
        conf_threshold = st.sidebar.slider("conf_threshold", 0.1, 1.0, 0.1)
        iou_threshold = st.sidebar.slider("iou_threshold", 0.1, 1.0, 0.1)

    model_params['model'] = 'yolo'
    model_params['conf_threshold'] = conf_threshold
    model_params['iou_threshold'] = iou_threshold
    return model_params

@st.cache_data()
def load_model(weights_path):
    """
    加载 YOLOv5 模型（缓存避免重复加载）

    :param weights_path: .pt 权重文件路径
    :return: YOLOv5 模型对象
    """
    model = torch.hub.load('/mnt/dfc_data1/home/linyusen/project/84_patheye_gui/yolov5-7.0', 'custom',
                           path=weights_path,
                           source='local')
    return model

def detect_images(model, image_files):
    """
    对上传图片批量推理

    :param model: 已加载的 YOLOv5 模型
    :param image_files: 上传的图片文件列表
    :return: list of (PIL.Image, DataFrame)，每张图片及对应检测结果
    """
    results = []
    for img_file in image_files:
        img = Image.open(img_file)
        result = model(img)
        data = result.pandas().xyxy[0]
        data.insert(0, 'image', img_file.name)
        data['area'] = (data['xmax'] - data['xmin']) * (data['ymax'] - data['ymin'])
        results.append((img, data))
    return results

def draw_boxes(img, data):
    """
    在图片上绘制检测框

    :param img: PIL.Image 原图
    :param data: DataFrame，包含 xmin/ymin/xmax/ymax
    :return: 绘制后的 PIL.Image
    """
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for _, row in data.iterrows():
        draw.rectangle(
            [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
            outline="red",
            width=3
        )
    return img

def app():
    """
    Streamlit 主应用：上传图片 -> 选择模型/参数 -> 运行检测 -> 展示结果
    """
    st.header('目标检测')

    # 上传图片
    uploaded_imgs = uploaded_img_viewer(
        "choose directory to upload",
        ['png', 'jpg', 'jpeg'],
        True
    )

    # 上传完成后显示侧边栏参数与运行按钮
    if uploaded_imgs:
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M")

        # 获取检测参数
        model_params = model_param_selector()
        conf_thresh = model_params['conf_threshold']
        iou_thresh = model_params['iou_threshold']

        # 扫描本地模型权重
        scan_dir = Path("runs/train/model_path")
        pt_files = sorted(glob.glob(str(scan_dir / "*.pt"))) if scan_dir.exists() else []
        file_labels = [Path(f).name for f in pt_files]
        selected_label = st.sidebar.selectbox(
            "选择模型文件",
            file_labels,
            index=0 if file_labels else 0,
            disabled=not file_labels
        )
        MODEL_PATH = pt_files[file_labels.index(selected_label)] if file_labels else None

        # 运行检测
        if st.button("run detection"):
            with st.spinner('加载模型中...'):
                model = load_model(MODEL_PATH)
                model.conf = conf_thresh
                model.iou = iou_thresh
            with st.spinner('检测中...'):
                results = detect_images(model, uploaded_imgs)
            st.success("检测完成!")
            # 缓存结果与索引
            st.session_state.detection_results = [
                {
                    "original": orig_img,
                    "annotated": draw_boxes(orig_img.copy(), data),
                    "data": data
                } for orig_img, data in results
            ]
            st.session_state.result_index = 0
            st.session_state.raw_results = results

        # 结果展示区域
        if 'detection_results' in st.session_state:
            res = st.session_state.detection_results[st.session_state.result_index]

            # 原图 vs 检测结果
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(res["original"], caption="原图")
            with col2:
                st.image(res["annotated"], caption="检测结果")

            # 上一张/下一张
            btn_col1, btn_col2, _ = st.columns([1, 1, 4])
            with btn_col1:
                if st.button("◀ 上一个", disabled=st.session_state.result_index == 0):
                    st.session_state.result_index -= 1
                    st.rerun()
            with btn_col2:
                if st.button("下一个 ▶",
                             disabled=st.session_state.result_index == len(st.session_state.detection_results) - 1):
                    st.session_state.result_index += 1
                    st.rerun()

            # 当前图片检测表格
            st.dataframe(res["data"])

            # 汇总所有图片检测结果
            results = st.session_state.raw_results
            all_df = []
            for _, data in results:
                all_df.append(data)
            summary_df = pd.concat(all_df, ignore_index=True)
            st.subheader("汇总检测结果")
            st.dataframe(summary_df)