# 导入 shutil 模块用于文件和文件夹操作
import shutil
# 导入 tempfile 模块用于创建临时文件和文件夹
import tempfile
# 导入 streamlit 用于构建 Web 应用
import streamlit as st
# 导入 datetime 用于获取时间戳
from datetime import datetime
# 导入 os 模块用于文件系统操作
import os
# 从 wsi_to_patch 模块导入 histolab_func 函数，用于病理图像切片
from wsi_to_patch import histolab_func
# 导入 glob 模块用于匹配文件路径
import glob

# 定义上传目录常量
UPLOAD_DIR = "uploads"
# 如果上传目录不存在则创建
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_timestamp():
    """
    返回精确到分钟的时间戳
    :return: 字符串格式的时间戳，精确到分钟
    """
    return datetime.now().strftime("%Y_%m_%d_%H_%M")

def get_timestamp_dir(category):
    """
    生成带时间戳的目录
    :param category: 目录类别前缀
    :return: 创建的目录路径
    """
    ts = get_timestamp()
    save_dir = os.path.join(UPLOAD_DIR, f"{category}_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def patch_images(patch_input_dir, patch_output_dir):
    """
    病理图像分割，不是计算机视觉的分割，是把大图切成小图
    :param patch_input_dir: 输入病理图像目录
    :param patch_output_dir: 输出小图块目录
    :return: 无
    """
    os.makedirs(patch_output_dir, exist_ok=True)
    wsi_paths = glob.glob(os.path.join(patch_input_dir, '*.ndpi'))
    for i, wsi_path in enumerate(wsi_paths):
        histolab_func(wsi_path, patch_output_dir, i + 1)

def he_uploader(label,type,accept_multiple_files):
    """
    streamlit自带上传组件
    :param label: 上传组件标签
    :param type: 允许上传的文件类型
    :param accept_multiple_files: 是否允许多文件上传
    :return: 上传的文件对象
    """
    upload_slide=st.file_uploader(
        label,
        type,
        accept_multiple_files
    )
    return upload_slide

def make_zip_from_dir(dir_path, zip_name="output.zip"):
    """
    打包文件
    :param dir_path: 需要打包的目录路径
    :param zip_name: 输出的zip文件名
    :return: 生成的zip文件路径
    """
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, zip_name)
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', dir_path)
    return zip_path

def app():
    """
    以上函数的实现，上传、保存到服务器、分割处理、打包、下载到本地
    :return: 无
    """
    st.header('病理图像切割')

    # 初始化会话状态变量
    if "pending_ndpi_files" not in st.session_state:
        st.session_state.pending_ndpi_files = {}
    if "ndpi_save_dir" not in st.session_state:
        st.session_state.ndpi_save_dir = None
    if "patch_done" not in st.session_state:
        st.session_state.patch_done = False
    if "patch_output_dir" not in st.session_state:
        st.session_state.patch_output_dir = None

    # 文件上传组件
    uploaded_ndpi = st.file_uploader("选择待处理文件",
                                     type=[
                                         "ndpi", "svs", "tif", "tiff", "mrxs", "scn", "vms", "vmu", "jpg",
                                         "jpeg", "png", "bmp"
                                     ],
                                     accept_multiple_files=True)

    # 将上传的文件加入待处理列表
    if uploaded_ndpi:
        for f in uploaded_ndpi:
            st.session_state.pending_ndpi_files[f.name] = f

    # 显示待上传文件列表
    if st.session_state.pending_ndpi_files:
        st.subheader("待上传文件列表")
        for name in st.session_state.pending_ndpi_files:
            st.write(name)

    # 上传按钮：保存文件到服务器
    if st.button("确定上传"):
        if st.session_state.pending_ndpi_files:
            ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
            ndpi0_dir = os.path.join(UPLOAD_DIR, "resourse", f"ndpi_{ts}")
            os.makedirs(ndpi0_dir, exist_ok=True)
            for name, f in st.session_state.pending_ndpi_files.items():
                with open(os.path.join(ndpi0_dir, name), "wb") as out:
                    out.write(f.getvalue())
            st.success(f"已上传 {len(st.session_state.pending_ndpi_files)} 个文件到 {ndpi0_dir}")
            st.session_state.ndpi_save_dir = ndpi0_dir
            st.session_state.pending_ndpi_files = {}
            st.session_state.patch_done = False

    # 如果文件已上传，准备切割
    if st.session_state.ndpi_save_dir:
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
        patch_output_dir = os.path.join(UPLOAD_DIR, "preprocessing_output", f"patch_results_{ts}")
        os.makedirs(patch_output_dir, exist_ok=True)

        # 开始切割按钮
        if st.button("开始切割 "):
            with st.spinner("正在切割 Patch..."):
                patch_images(st.session_state.ndpi_save_dir, patch_output_dir)
            st.success(f"Patch 切割完成: {patch_output_dir}")
            st.session_state.patch_done = True
            st.session_state.patch_output_dir = patch_output_dir

        # 切割完成后提供下载
        if st.session_state.get("patch_done"):
            patch_zip = make_zip_from_dir(st.session_state.patch_output_dir, "patch_results.zip")
            with open(patch_zip, "rb") as f:
                st.download_button("下载结果 (zip)", f, file_name=f"patch_results_{ts}.zip")