# -*- coding: utf-8 -*-
"""
网页功能介绍页，有关视频的部分设计个人信息没有提供，可以先注释掉
作者：shannon yxn
日期：2025-10
"""
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2

def load_lottie_image(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载本地文件失败: {str(e)}")
        return None


def load_css(css_file):
     with open(css_file) as f:
         st.markdown('<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    load_css("style/style.css")
    lottie_animation = load_lottie_image("home_res/Movement.json")
    if lottie_animation:
      st_lottie(lottie_animation, height=200)


    #Part Ⅰ
    with st.container():
        st.subheader('Hi!Welcome to HE Processor! :shark:')
        st.write(
              "以下是本网页功能页面的简介，让我们一起开启吧！"
        )


    #Part Ⅱ
    with st.container():
        st.write('---')
        st.subheader("H&E To Patches")
        l_column,r_column = st.columns(2)
        with l_column:
            video_file=open("home_res/patches.mp4","rb")
            video_bytes=video_file.read()
            st.video(video_bytes)
        with r_column:
            st.write(
                """
                简介：该模块用于分割H&E图像以解决深度学习的显存限制问题\n
                操作步骤：
                - 通过拖拽或点击 Browse files 的按钮上传待处理病理图像
                - 点击确认上传，后台保存图像
                -点击开始切割按键，右上角出现闪烁图标，闪烁停止并且出现成功提醒，图像切割完成
                - 点击下载按钮即可下载切割好的图像
                !!! 请注意：该步骤处理速度较慢，每张病理图像处理时间约为5-15分钟，请合理安排时间。
                """)

    #Part Ⅲ
    with st.container():
        st.write('---')
        st.subheader("Labeling")
        l_column,r_column = st.columns(2)
        with l_column:
            video_file=open("home_res/labeling.mp4","rb")
            video_bytes=video_file.read()
            st.video(video_bytes)
        with r_column:
            st.write(
                """
                简介：该部分使用开源平台Make Sense进行图像标注\n
                步骤：
                - 左侧导航切换至labeling页面
                -点击立即使用即可跳转页面
                - 使用方法（待补充）
                """
            )


    #Part Ⅳ
    with st.container():
        st.write('---')
        st.subheader("Training")
        l_column,r_column = st.columns(2)
        with l_column:
            video_file=open("home_res/training.mp4","rb")
            video_bytes=video_file.read()
            st.video(video_bytes)
        with r_column:
            st.write(
                """
                简介：训练模型，包含数据集划分、图像预处理以及训练三个组件\n
                步骤：
                - 同样是拖拽或浏览本地文件夹上传图像以及标注
                - 看见图像成功保存提醒后点击划分数据集（参数已设置好，不用调整）
                - 划分数据集成功后自动进行图像预处理，无需调整参数
                - 提示图像预处理完成后即可点击训练模型按钮进行训练
                - 训练成果展示训练结果，训练好的模型权重自动保存在服务器，可直接在检测页面调用
                """
                    )

        # Part Ⅳ
        with st.container():
            st.write('---')
            st.subheader("Detection")
            l_column, r_column = st.columns(2)
            with l_column:
                video_file=open("home_res/detection.mp4","rb")
                video_bytes = video_file.read()
                st.video(video_bytes)
            with r_column:
                st.write(
                    """
                    简介：该部分用于已经切割好的图像进行检测\n
                    步骤：
                    - 同样是拖拽或浏览本地文件夹上传图像
                    - 看见图像成功显示后即可在侧边栏选择合适的阈值进行目标检测
                    - 划分数据集成功后自动进行图像预处理，无需调整参数
                    - 统计结果可以通过下载按钮进行下载
                    """
                )

        with st.container():
            st.write('---')
            st.subheader("🔴Gentle reminder")
            st.write("""
                     由于技术限制，在使用不同功能页面时，页面切换之前最好刷新一下网页，或者同一个功能重复使用之前也需要刷新。否则可能会引起进程堵塞或者页面瘫痪:smile:
                     """)








