# -*- coding: utf-8 -*-
"""
Streamlit 端到端 YOLOv5 训练流水线
包括：图片/标注上传 → 数据集划分 → 图像增强/归一化 → 超参选择 → 模型训练 → 结果可视化
作者：YourName
日期：2025-06
"""

import re
import sys
import time
import yaml
import shutil
import random
import logging
import datetime
import tempfile
import subprocess
from PIL import Image
import streamlit as st
from pathlib import Path
from datetime import datetime
from wsi_patch_enhancer import ImageEnhancer
from wsi_patch_normalizer import StainNormalizer
from typing import List, Tuple, Dict, Optional, Union

# -------------------- 日志配置 --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='training_system.log'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. 通用组件
# ------------------------------------------------------------------
def uploaded_img_viewer(upload_title: str,
                        image_types: List[str],
                        accept_multiple_files: bool) -> Optional[List]:
    """
    带预览/翻页功能的图片上传组件

    :param upload_title: 上传区域标题
    :param image_types: 允许后缀列表，如 ['png','jpg']
    :param accept_multiple_files: 是否多文件
    :return: 上传文件列表 or None
    """
    # 会话状态初始化
    if 'img_index' not in st.session_state:
        st.session_state.img_index = 0
    if 'images' not in st.session_state:
        st.session_state.images = 0

    uploaded_img = st.file_uploader(upload_title,
                                    type=image_types,
                                    accept_multiple_files=accept_multiple_files)

    # 新上传触发重置
    if uploaded_img and uploaded_img != st.session_state.images:
        st.session_state.images = [uploaded_img] if not isinstance(uploaded_img, list) else uploaded_img
        st.session_state.img_index = 0

    # 当前图片显示
    if st.session_state.images:
        try:
            img = Image.open(st.session_state.images[st.session_state.img_index])
            st.image(img,
                     caption=f"{st.session_state.img_index + 1}/{len(st.session_state.images)} {img.filename}")
            st.write(st.session_state.images[st.session_state.img_index].name)
        except Exception as e:
            logger.error(f"图片打开失败: {str(e)}")
            st.error("图片显示失败，请检查文件格式")

    # 翻页按钮
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("◀ 上一张", key="prev"):
            if st.session_state.img_index > 0:
                st.session_state.img_index -= 1
            else:
                st.warning('已经是第一张图片!')
    with col2:
        if st.button("下一张▶", key="next"):
            if st.session_state.img_index < len(st.session_state.images) - 1:
                st.session_state.img_index += 1
            else:
                st.warning('已经是最后一张图片!')

    return st.session_state.images if st.session_state.images else None


def safe_file_copy(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    健壮的文件复制（含异常提示）

    :param src: 源文件
    :param dst: 目标文件
    :return: 成功 True / 失败 False
    """
    try:
        shutil.copy2(str(src), str(dst))
        return True
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}")
        st.error(f"源文件不存在: {src}")
    except PermissionError as e:
        logger.error(f"权限错误: {str(e)}")
        st.error(f"无权限访问文件: {dst}")
    except Exception as e:
        logger.error(f"文件操作出错: {str(e)}")
        st.error(f"文件复制失败: {str(e)}")
    return False


# ------------------------------------------------------------------
# 2. 数据集划分
# ------------------------------------------------------------------
def split_dataset(img_dir: Path,
                  lbl_dir: Path,
                  out_dir: Path,
                  ratios: Tuple[float, float, float],
                  seed: int) -> Optional[Dict[str, int]]:
    """
    将图片+标注按指定比例划分为 train/val/test

    :param img_dir: 图片目录
    :param lbl_dir: YOLO 格式 txt 标注目录
    :param out_dir: 输出根目录
    :param ratios: (train, val, test) 比例，和需为 1.0
    :param seed: 随机种子
    :return: 划分后各集样本数 dict or None
    """
    # 参数验证
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print("输入目录无效")
        return None
    if abs(sum(ratios) - 1.0) > 1e-6:
        print("比例之和须为1.0")
        return None

    try:
        # 收集图片并过滤有标注的样本
        img_list = [p for p in img_dir.glob('*')
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        names = [p.stem for p in img_list]
        names = [n for n in names if (lbl_dir / f'{n}.txt').exists()]
        if not names:
            print('未找到任何"图片-标注"成对的样本')
            return None

        # 随机划分
        random.seed(seed)
        random.shuffle(names)
        n = len(names)
        s1, s2 = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        splits = {'train': names[:s1], 'val': names[s1:s2], 'test': names[s2:]}

        # 复制文件到对应集
        for split, lst in splits.items():
            (out_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (out_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

            for stem in lst:
                # 查找图片（支持多种后缀）
                img_src = None
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    potential_path = img_dir / f'{stem}{ext}'
                    if potential_path.exists():
                        img_src = potential_path
                        break
                if img_src is None:
                    print(f"未找到图片文件: {stem}")
                    return None

                # 复制图片（保持原格式）
                img_dst = out_dir / split / 'images' / f'{stem}{img_src.suffix}'
                if not safe_file_copy(img_src, img_dst):
                    return None

                # 复制标注
                lbl_src = lbl_dir / f'{stem}.txt'
                lbl_dst = out_dir / split / 'labels' / f'{stem}.txt'
                if not safe_file_copy(lbl_src, lbl_dst):
                    return None

        return {k: len(v) for k, v in splits.items()}

    except Exception as e:
        logger.error(f"数据集划分失败: {str(e)}")
        print(f"数据集划分过程中出错: {str(e)}")
        return None


# ------------------------------------------------------------------
# 3. 预处理：增强 + 归一化
# ------------------------------------------------------------------
def overwrite_with_aug(original_dir: Path, aug_dir: Path) -> int:
    """
    用增强后的图片覆盖原图（原地替换）

    :param original_dir: 原图目录
    :param aug_dir: 增强图目录
    :return: 实际覆盖文件数
    """
    if aug_dir is None:
        return 0
    overwrite_cnt = 0
    aug_list = list(aug_dir.glob("*"))
    prog = st.progress(0)
    for i, aug_file in enumerate(aug_list):
        if aug_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        orig_file = original_dir / aug_file.name
        if orig_file.exists():
            shutil.copy2(aug_file, orig_file)
            overwrite_cnt += 1
        prog.progress((i + 1) / len(aug_list))
    prog.empty()
    return overwrite_cnt


# ------------------------------------------------------------------
# 4. 训练参数选择
# ------------------------------------------------------------------
def training_param_selector() -> Dict[str, Union[str, int, List[str]]]:
    """
    侧边栏收集 YOLOv5 训练超参

    :return: 超参 dict
    """
    model = st.sidebar.selectbox(
        label='选择模型',
        options=("yolov5s", "yolov5m", "yolov5l", "yolov5x")
    )

    training_params = {
        'model': model,
        'data_cfg': st.sidebar.text_input("数据集配置文件路径", 'data/mydata.yaml'),
        'epochs': st.sidebar.slider("训练轮次", 1, 300, 300),
        'batch_size': st.sidebar.selectbox("批大小", [8, 16, 32, 64], index=1),
        'image_size': st.sidebar.slider("输入尺寸", 320, 1280, 640, 32),
        'weights_patch': st.sidebar.text_input("预训练权重", "yolov5s.pt"),
        'cfg_path': st.sidebar.text_input("配置文件", "models/yolov5s.yaml")
    }
    return training_params


# ------------------------------------------------------------------
# 5. 训练进度解析
# ------------------------------------------------------------------
def parse_training_progress(line: str) -> Dict[str, Union[str, List[str]]]:
    """
    从训练日志单行提取关键信息

    :param line: 日志行
    :return: 解析结果 dict
    """
    patterns = {
        'epoch': r"Epoch\s+(\d+)/\d+",
        'metrics': r"box_loss:\s+(\d+\.\d+).*cls_loss:\s+(\d+\.\d+)",
        'time': r"time=\s+(\d+:\d+)",
        'gpu_mem': r"GPU mem:\s+(\d+\.\d+)GB"
    }

    progress = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            progress[key] = match.group(1) if key != 'metrics' else [match.group(1), match.group(2)]
    return progress


# ------------------------------------------------------------------
# 6. 增强版训练函数（无实时前端日志）
# ------------------------------------------------------------------
def enhanced_run_yolov5_training(
        data_yaml: str,
        epochs: int,
        image_size: int,
        batch_size: int,
        weights_path: str,
        cfg_path: str) -> Optional[Tuple[str, str]]:
    """
    启动 YOLOv5 训练并后台落盘日志

    :param data_yaml: 数据集 yaml 路径
    :param epochs: 训练轮数
    :param image_size: 输入分辨率
    :param batch_size: 批大小
    :param weights_path: 预训练权重
    :param cfg_path: 模型 yaml 配置
    :return: (exp_dir, best_pt_path) or None
    """
    # 文件存在检查
    required_files = [data_yaml, weights_path, cfg_path]
    for f in required_files:
        if not Path(f).exists():
            st.error(f"必要文件缺失: {f}")
            return None

    # 日志目录
    log_dir = Path("training_logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    status = st.empty()
    status.info("训练中 ⏳ …")
    progress_bar = st.empty()

    try:
        with open(log_file, "w", buffering=1) as f_log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "train.py",
                 "--img", str(image_size),
                 "--batch", str(batch_size),
                 "--epochs", str(epochs),
                 "--data", str(data_yaml),
                 "--weights", str(weights_path),
                 "--cfg", str(cfg_path),
                 "--project", "runs/train",
                 "--name", "exp",
                 "--exist-ok"],
                stdout=f_log,
                stderr=subprocess.STDOUT
            )
            # 轮询进度
            while proc.poll() is None:
                time.sleep(2)
                if not log_file.exists():
                    continue
                txt = log_file.read_text(encoding="utf-8", errors="ignore")
                progress = parse_training_progress("\n".join(txt.splitlines()[-50:]))
                if 'epoch' in progress:
                    current_epoch = int(progress['epoch'])
                    progress_bar.progress(min(current_epoch / epochs, 1.0))
            proc.wait()

        if proc.returncode != 0:
            status.error("训练失败 ❌")
            st.button("查看出错日志", on_click=lambda: st.code(log_file.read_text()))
            return None

        status.success("训练完成 ✅")
        # 查找最新 exp
        exp_dirs = sorted(Path("runs/train").glob("exp*"),
                          key=lambda x: x.stat().st_mtime,
                          reverse=True)
        if not exp_dirs:
            st.error("未生成训练结果")
            return None

        best_pt = str(exp_dirs[0] / "weights" / "best.pt")
        # 额外拷贝一份带时间戳的权重
        save_dir = Path("runs/train/model_path")
        save_dir.mkdir(parents=True, exist_ok=True)
        train_name = f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{Path(data_yaml).stem}"
        dst_path = save_dir / f"{train_name}_best.pt"
        shutil.copy2(best_pt, dst_path)
        logger.info(f"best.pt 已额外保存至 {dst_path}")
        return str(exp_dirs[0]), best_pt

    except Exception as e:
        logger.error(f"训练过程异常: {str(e)}")
        status.error(f"训练过程中出错: {str(e)}")
        return None


# ------------------------------------------------------------------
# 7. Streamlit 页面组装
# ------------------------------------------------------------------
def app():
    """
    Streamlit 主页面：按顺序完成上传→划分→预处理→训练→可视化
    """
    st.title("模型训练")
    base_path = Path(__file__).parent.parent
    UPLOAD_DIR = base_path / "dataset"
    yolov5_path = base_path / 'yolov5-7.0'
    sys.path.insert(0, str(yolov5_path))

    # 会话状态初始化
    if 'pending_path' not in st.session_state:
        st.session_state.pending_path = None
    if 'split_config' not in st.session_state:
        st.session_state.split_config = {
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'random_seed': 42
        }
    if 'split_path' not in st.session_state:
        st.session_state.split_path = None
    if 'preprocessing_config' not in st.session_state:
        st.session_state.preprocessing_config = {
            'enhancement': ['gaussian', 'median', 'hist_eq', 'usm_sharpen'],
            'normalization': 'Reinhard_default',
            'reference_img': 'reference.png'
        }
    if 'enhance_cfg_done' not in st.session_state:
        st.session_state.enhance_cfg_done = False
    if 'enhancement_done' not in st.session_state:
        st.session_state.enhancement_done = False
    if 'normalization_done' not in st.session_state:
        st.session_state.normalization_done = False
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    if "training_config" not in st.session_state:
        st.session_state.training_config = {}
    if "result_dir" not in st.session_state:
        st.session_state.result_dir = None
    if "best_pt" not in st.session_state:
        st.session_state.best_pt = None

    # 1. 上传数据
    st.subheader("1. 上传训练数据")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_imgs = uploaded_img_viewer(
            "上传图片文件",
            ['png', 'jpg', 'jpeg'],
            True
        )
    with col2:
        uploaded_txts = st.file_uploader(
            '上传标注文件',
            type=['txt'],
            accept_multiple_files=True
        )

    # 保存到服务器
    if uploaded_imgs and uploaded_txts:
        ts = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        root = UPLOAD_DIR / f'yolo_{ts}'
        img_save_dir = root / 'raw_upload' / 'images'
        lbl_save_dir = root / 'raw_upload' / 'labels'
        img_save_dir.mkdir(parents=True, exist_ok=True)
        lbl_save_dir.mkdir(parents=True, exist_ok=True)

        try:
            for f in uploaded_imgs:
                (img_save_dir / f.name).write_bytes(f.getbuffer())
            for f in uploaded_txts:
                # 统一替换后缀为 .txt
                clean_name = f.name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                (lbl_save_dir / clean_name).write_bytes(f.getbuffer())

            st.success(f'已保存 {len(uploaded_imgs)} 张图片 + {len(uploaded_txts)} 个标注')
            st.session_state.pending_path = root

        except Exception as e:
            logger.error(f"文件保存失败: {str(e)}")
            st.error(f"文件保存过程中出错: {str(e)}")

    # 2. 数据集划分
    if st.session_state.pending_path and not st.session_state.split_path:
        st.subheader("2. 数据集划分")
        with st.form('数据集划分配置'):
            cfg = st.session_state.split_config
            tr = st.slider('训练集比例', 0.1, 0.9, cfg['train_ratio'], 0.05)
            va = st.slider('验证集比例', 0.1, 0.9, cfg['val_ratio'], 0.05)
            te = st.slider('测试集比例', 0.1, 0.9, cfg['test_ratio'], 0.05)
            if abs(tr + va + te - 1.0) > 0.01:
                st.warning('三项之和须为 1.0')
            seed = st.number_input('随机种子', value=cfg['random_seed'])
            submitted_1 = st.form_submit_button('保存配置', key='scfg')
            if submitted_1:
                cfg.update({'train_ratio': tr, 'val_ratio': va, 'test_ratio': te, 'random_seed': seed})

        if st.session_state.split_config and st.button('确认划分数据集', key='split'):
            root = Path(st.session_state.pending_path)
            img_dir = root / 'raw_upload' / 'images'
            lbl_dir = root / 'raw_upload' / 'labels'
            out_path = root / 'split'
            ratios = (cfg['train_ratio'], cfg['val_ratio'], cfg['test_ratio'])
            cnt = split_dataset(img_dir, lbl_dir, out_path, ratios, cfg['random_seed'])
            if cnt:
                st.success(f'划分完成！训练集: {cnt["train"]}, 验证集: {cnt["val"]}, 测试集: {cnt["test"]}')
                st.session_state.split_path = out_path

    # 3. 预处理配置
    if st.session_state.split_path and not st.session_state.enhance_cfg_done:
        with st.form("图像预处理设置"):
            preprocessing_cfg = st.session_state.preprocessing_config
            en = st.multiselect(
                "图像增强方法",
                ['gaussian', 'median', 'hist_eq', 'usm_sharpen'],
                default=['gaussian', 'median', 'hist_eq', 'usm_sharpen']
            )

            nor = st.selectbox(
                "归一化方法",
                ('Reinhard_default', 'Reinhard_reference', 'Macenko')
            )

            reference_path = st.text_input("参考图像路径", value=preprocessing_cfg['reference_img'])
            submitted_2 = st.form_submit_button('保存配置', key='pcfg')
            if submitted_2:
                preprocessing_cfg.update({'enhancement': en,
                                          'normalization': nor,
                                          'reference_img': reference_path})
                st.session_state.enhance_cfg_done = True

    # 4. 执行预处理
    if st.session_state.enhance_cfg_done and not st.session_state.preprocessing_done:
        raw_dir = Path(st.session_state.split_path) / 'train' / 'images'
        en_dir = Path(st.session_state.split_path) / 'train' / 'images_enhanced'
        nor_dir = Path(st.session_state.split_path) / 'train' / 'images_normalized'
        en_dir.mkdir(parents=True, exist_ok=True)
        nor_dir.mkdir(parents=True, exist_ok=True)
        preprocessing_cfg = st.session_state.preprocessing_config
        enhan_options = preprocessing_cfg['enhancement']
        norm_options = preprocessing_cfg['normalization']
        reference_img = preprocessing_cfg['reference_img']

        # 图像增强
        if enhan_options:
            pipeline = enhan_options
            enhancer = ImageEnhancer(raw_dir, en_dir)
            enhancer.run(pipeline)
            st.session_state.enhancement_done = True
        else:
            en_dir = raw_dir

        # 颜色归一化
        sn = StainNormalizer()
        if norm_options == "Reinhard_default":
            sn.batch_normalize(en_dir, nor_dir, method='reinhard')
        elif norm_options == 'Reinhard_reference':
            ref_path = Path(reference_img)
            if ref_path.exists():
                sn.set_reference(ref_path)
            sn.batch_normalize(en_dir, nor_dir, method='reinhard')
        elif norm_options == "Macenko":
            sn.batch_normalize(en_dir, nor_dir, method='macenko')

        st.session_state.normalization_done = True
        st.session_state.preprocessing_done = True
        st.success("预处理完成！")

    # 5. 训练配置与启动
    if st.session_state.preprocessing_done:
        st.session_state.training_config = training_param_selector()
        training_params = st.session_state.training_config
        if training_params and st.button("开始训练", key="training"):
            data_yaml = training_params["data_cfg"]
            epochs = training_params["epochs"]
            batch_size = training_params["batch_size"]
            image_size = training_params["image_size"]
            weights_path = training_params["weights_patch"]
            cfg_path = training_params["cfg_path"]

            # 动态更新 data.yaml 中的 path 字段
            with open(data_yaml, "r") as f:
                data_cfg = yaml.safe_load(f)
            data_cfg["path"] = str(st.session_state.split_path)
            with open(data_yaml, "w") as f:
                yaml.dump(data_cfg, f, sort_keys=False)

            st.success(f"data.yaml 已更新，path -> {st.session_state.split_path}")

            result = enhanced_run_yolov5_training(
                data_yaml,
                epochs,
                image_size,
                batch_size,
                weights_path,
                cfg_path
            )
            if result is None:
                st.error("训练失败，未返回结果")
            else:
                result_dir, best_pt = result
                st.session_state.result_dir = result_dir
                st.session_state.best_pt = best_pt

    # 6. 结果可视化
    if st.session_state.get("result_dir"):
        st.subheader("训练结果展示")
        result_dir = st.session_state.result_dir
        best_pt = st.session_state.best_pt
        try:
            cols = st.columns(3)
            with cols[0]:
                try:
                    result_img = Image.open(f"{result_dir}/results.png")
                    st.image(result_img, caption="训练指标")
                except Exception:
                    st.error("训练指标图加载失败")

            with cols[1]:
                try:
                    conf_img = Image.open(f"{result_dir}/confusion_matrix.png")
                    st.image(conf_img, caption="混淆矩阵")
                except Exception:
                    st.error("混淆矩阵图加载失败")

            with cols[2]:
                try:
                    pr_img = Image.open(f"{result_dir}/PR_curve.png")
                    st.image(pr_img, caption="PR曲线")
                except Exception:
                    st.error("PR曲线图加载失败")

        except Exception as e:
            st.error(f"加载训练结果图表失败: {str(e)}")