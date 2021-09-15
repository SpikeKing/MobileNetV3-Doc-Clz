#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 31.8.21
"""

import argparse
from multiprocessing.pool import Pool

from myutils.cv_utils import *
from myutils.make_html_page import make_html_page
from myutils.project_utils import *
from x_utils.vpf_sevices import get_vpf_service_np


class ServiceTester(object):
    def __init__(self, in_folder, service, out_folder):
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.service = service
        print('[Info] 输入文件夹: {}'.format(self.in_folder))
        print('[Info] 服务: {}'.format(self.service))
        print('[Info] 输出文件夹: {}'.format(self.out_folder))

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Image-Rotation/imgs-tmp/{}".format(get_current_day_str())
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_img_path(img_idx, img_path, service,  out_file):
        img_bgr = cv2.imread(img_path)
        res_dict = get_vpf_service_np(img_np=img_bgr, service_name=service)  # 表格
        p_label = res_dict["data"]["label"]
        p_label = int(p_label)
        r_label = int(img_path.split("/")[-2])

        if p_label != r_label:
            img_name = "{}-{}.jpg".format(get_current_time_str(), time.time())
            img_url = ServiceTester.save_img_path(img_bgr, img_name)
            write_line(out_file, "{}\t{}\t{}".format(img_url, r_label, p_label))
        print('[Info] 处理完成: {}, right: {}'.format(img_idx, p_label == r_label))

    def process_folder(self):
        paths_list, names_list = traverse_dir_files(self.in_folder)
        time_str = get_current_time_str()
        out_file = os.path.join(self.out_folder, "val_{}.txt".format(time_str))
        out_html = os.path.join(self.out_folder, "val_{}.html".format(time_str))
        pool = Pool(processes=100)
        for img_idx, img_path in enumerate(paths_list):
            # ServiceTester.process_img_path(img_idx, img_path, self.service, out_file)
            pool.apply_async(ServiceTester.process_img_path, (img_idx, img_path, self.service, out_file))
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(out_file))
        data_lines = read_file(out_file)
        print('[Info] 正确率: {}'.format(safe_div(len(paths_list) - len(data_lines), len(paths_list))))
        out_list = []
        label_str_list = ["纸质文档", "非纸质文档"]
        for data_line in data_lines:
            img_url, r_label, p_label = data_line.split("\t")
            out_list.append([img_url, label_str_list[int(r_label)], label_str_list[int(p_label)]])
        make_html_page(out_html, out_list)
        print('[Info] 处理完成: {}'.format(out_html))


def parse_args():
    """
    处理脚本参数，支持相对路径
    """
    parser = argparse.ArgumentParser(description='服务测试')
    parser.add_argument('-i', dest='in_folder', required=False, help='测试文件夹', type=str)
    parser.add_argument('-s', dest='service', required=False, help='服务', type=str)
    parser.add_argument('-o', dest='out_folder', required=False, help='输出文件夹', type=str)

    args = parser.parse_args()

    arg_in_folder = args.in_folder
    print("测试文件夹: {}".format(arg_in_folder))

    # ysu362VFeRZkizhfBkfbck
    arg_service = args.service
    print("服务: {}".format(arg_service))

    arg_out_folder = args.out_folder
    print("输出文件夹: {}".format(arg_out_folder))
    mkdir_if_not_exist(arg_out_folder)

    return arg_in_folder, arg_service, arg_out_folder


def main():
    arg_in_folder, arg_service, arg_out_folder = parse_args()
    st = ServiceTester(arg_in_folder, arg_service, arg_out_folder)
    st.process_folder()


if __name__ == '__main__':
    main()
