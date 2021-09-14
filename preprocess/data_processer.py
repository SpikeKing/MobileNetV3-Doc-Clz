#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 26.8.21
"""
import os
import sys
from multiprocessing.pool import Pool

import cv2
import xml.dom.minidom

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from root_dir import DATA_DIR


class SampleLabeledParser(object):
    """
    简单样本解析
    """
    def __init__(self):
        self.image_urls = os.path.join(DATA_DIR, 'files', "doc_clz_urls.txt")
        self.label_path = os.path.join(DATA_DIR, 'files', 'annotations.xml')
        self.out_labeled = os.path.join(DATA_DIR, 'files', 'out_labeled_{}.txt'.format(get_current_time_str()))
        self.out_label_list = os.path.join(DATA_DIR, 'files', 'out_label_list.txt')

    @staticmethod
    def split_boxes(pnt_list, box):
        """
        根据点列表拆分box
        """
        if not pnt_list:
            return [box]
        x_min, y_min, x_max, y_max = box
        x_list = []
        for pnt in pnt_list:
            x_list.append(pnt[0])
        x_list = sorted(x_list)
        sub_boxes = []
        x_s = x_min
        for x in x_list:
            sub_boxes.append([x_s, y_min, x, y_max])
            x_s = x
        sub_boxes.append([x_s, y_min, x_max, y_max])
        return sub_boxes

    @staticmethod
    def parse_pnt_and_box(box_pnt_dict, box_list, img_bgr=None):
        """
        解析点和box
        """
        sub_boxes_list = []

        for idx in box_pnt_dict.keys():
            pnt_list = box_pnt_dict[idx]
            # print('[Info] pnt_list: {}'.format(pnt_list))
            box = box_list[idx]
            sub_boxes = SampleLabeledParser.split_boxes(pnt_list, box)
            sub_boxes_list.append(sub_boxes)

        sub_boxes_list = unfold_nested_list(sub_boxes_list)  # 双层list变成单层list

        # 划掉文字的区域需要区分对待
        for x_idx in range(len(box_list)):
            if x_idx not in box_pnt_dict.keys():
                sub_boxes_list.append(box_list[x_idx])

        # tmp_path = os.path.join(DATA_DIR, 'tmps', 'sub_boxes.jpg')
        # draw_box_list(img_bgr, sub_boxes_list, is_text=False, color=(255, 0, 0), save_name=tmp_path)
        return sub_boxes_list

    def parse_urls_dict(self):
        """
        解析urls
        """
        data_lines = read_file(self.image_urls)
        url_dict = dict()
        for data_line in data_lines:
            items = data_line.split("/")
            img_name = items[-1]
            url_dict[img_name] = data_line
        return url_dict

    def process_annotations(self):
        """
        处理解析标签
        """
        label_dict = {"纸质文档": 0, "拍摄电脑屏幕": 1, "精美生活照": 2, "不确定的类别": 3, "手机截屏": 4, "卡证": 5}
        DOMTree = xml.dom.minidom.parse(self.label_path)
        collection = DOMTree.documentElement
        meta = collection.getElementsByTagName("meta")
        # print('[Info] meta: {}'.format(meta))
        image_data = collection.getElementsByTagName("image")
        print('[Info] 样本数: {}'.format(len(image_data)))
        url_dict = self.parse_urls_dict()

        anno_list = []  # 标签信息列表
        label_list = []
        for image in image_data:
            image_name = image.getAttribute("name")
            # print('[Info] image: {}'.format(image_name))
            img_url = url_dict[image_name]
            # print("[Info] img_url: {}".format(img_url))
            points_data = image.getElementsByTagName("points")
            # print('[Info] points_data: {}'.format(len(points_data)))
            img_label = ""
            for points in points_data:
                img_label = points.getAttribute("label")
            # print('[Info] img_url: {}, img_label: {}'.format(img_url, img_label))
            # img_anno_dict = {
            #     "image_name": img_url,
            #     "image_label": label_dict[img_label]
            # }
            # label_list.append(img_label)
            # img_anno_str = json.dumps(img_anno_dict)
            anno_list.append("{}\t{}".format(img_url, label_dict[img_label]))

        label_list = list(set(label_list))
        write_list_to_file(self.out_label_list, label_list)
        print('[Info] 标签数量: {}'.format(len(anno_list)))
        write_list_to_file(self.out_labeled, anno_list)
        print('[Info] 标签文本写入完成: {}'.format(self.out_labeled))

    def analyze_dataset(self):
        file_name = os.path.join(DATA_DIR, "files", "out_labeled_urls.txt")
        label_dict = {"纸质文档": 0, "拍摄电脑屏幕": 1, "精美生活照": 2, "不确定的类别": 3, "手机截屏": 4, "卡证": 5}
        label_dict = invert_dict(label_dict)

        data_lines = read_file(file_name)
        data_dict = collections.defaultdict(int)
        for data_line in data_lines:
            url, label = data_line.split("\t")
            data_dict[label_dict[int(label)]] += 1
        print("data_dict: {}".format(data_dict))

    @staticmethod
    def process_data(data_idx, url, label, dataset_dir, data_type=""):
        # 根据数据集，设置数据量
        label_str = str(str(label).zfill(3))
        if data_type:
            out_label_dir = os.path.join(dataset_dir, label_str)
        else:
            out_label_dir = dataset_dir
        mkdir_if_not_exist(out_label_dir)

        _, img_bgr = download_url_img(url)
        if data_type:
            out_name = "{}_{}_{}.jpg".format(data_type, str(data_idx).zfill(6), str(str(label).zfill(3)))
        else:
            out_name = "{}_{}.jpg".format(str(data_idx).zfill(6), str(str(label).zfill(3)))

        out_path = os.path.join(out_label_dir, out_name)
        cv2.imwrite(out_path, img_bgr)
        print('[Info] label: {}, idx: {}'.format(label_str, data_idx))

    def make_dataset(self):
        train_file_name = os.path.join(DATA_DIR, "files", "out_labeled_urls_train_balanced.txt")
        print('[Info] label文件: {}'.format(train_file_name))
        train_dir = os.path.join(DATA_DIR, "document_dataset_mini", "train")
        mkdir_if_not_exist(train_dir)
        train_lines = read_file(train_file_name)
        print('[Info] 样本数: {}'.format(len(train_lines)))

        val_file_name = os.path.join(DATA_DIR, "files", "out_labeled_urls_val_balanced.txt")
        print('[Info] label文件: {}'.format(val_file_name))
        val_dir = os.path.join(DATA_DIR, "document_dataset_mini", "val")
        mkdir_if_not_exist(val_dir)
        val_lines = read_file(val_file_name)
        print('[Info] 样本数: {}'.format(len(val_lines)))

        train_label_dict = collections.defaultdict(int)
        val_label_dict = collections.defaultdict(int)
        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(train_lines):
            url, label = data_line.split("\t")
            if train_label_dict[label] == 16:
                continue
            pool.apply_async(SampleLabeledParser.process_data, (data_idx, url, label, train_dir, "train"))
            train_label_dict[label] += 1

        for data_idx, data_line in enumerate(val_lines):
            url, label = data_line.split("\t")
            if val_label_dict[label] == 16:
                continue
            pool.apply_async(SampleLabeledParser.process_data, (data_idx, url, label, val_dir, "val"))
            val_label_dict[label] += 1

        pool.close()
        pool.join()
        print('[Info] 全部写入完成: {}'.format(train_dir))

    def make_dataset_v2(self):
        out_file_name = os.path.join(DATA_DIR, "files", "out_labeled_urls_balanced.txt")
        print('[Info] label文件: {}'.format(out_file_name))
        out_dir = os.path.join(DATA_DIR, "document_dataset_v2_2")
        mkdir_if_not_exist(out_dir)
        train_lines = read_file(out_file_name)
        print('[Info] 样本数: {}'.format(len(train_lines)))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(train_lines):
            url, label = data_line.split("\t")
            pool.apply_async(SampleLabeledParser.process_data, (data_idx, url, label, out_dir))
        pool.close()
        pool.join()
        print('[Info] 全部写入完成: {}'.format(out_dir))

    def balance_samples(self):
        file_path = os.path.join(DATA_DIR, "files", "out_labeled_urls.txt")
        train_file_path = os.path.join(DATA_DIR, "files", "out_labeled_urls_train_balanced.txt")
        val_file_path = os.path.join(DATA_DIR, "files", "out_labeled_urls_val_balanced.txt")
        print("[Info] 文件路径: {}".format(file_path))
        data_lines = read_file(file_path)
        print("[Info] 样本数: {}".format(file_path))
        img_label_dict = collections.defaultdict(list)
        for data_line in data_lines:
            img_url, img_label = data_line.split("\t")
            img_label_dict[img_label].append(img_url)
        print_data_dict(img_label_dict)

        train_dict = collections.defaultdict(list)
        val_dict = collections.defaultdict(list)

        for img_label in img_label_dict.keys():
            samples = img_label_dict[img_label]
            gap = len(samples) // 10
            val_dict[img_label] = samples[0:gap]
            train_dict[img_label] = samples[gap:]

        num = 20000
        for img_label in train_dict.keys():
            samples = train_dict[img_label]
            samples = expand_sample_list(samples, num)
            train_dict[img_label] = samples

        num = 2000
        for img_label in val_dict.keys():
            samples = val_dict[img_label]
            samples = expand_sample_list(samples, num)
            val_dict[img_label] = samples

        print_data_dict(train_dict)
        out_list = []
        for img_label in train_dict.keys():
            samples = train_dict[img_label]
            for sample in samples:
                out_list.append("{}\t{}".format(sample, img_label))
        print('[Info] 样本数: {}'.format(len(out_list)))
        write_list_to_file(train_file_path, out_list)
        print('[Info] 写入完成: {}'.format(train_file_path))

        print_data_dict(val_dict)
        out_list = []
        for img_label in val_dict.keys():
            samples = val_dict[img_label]
            for sample in samples:
                out_list.append("{}\t{}".format(sample, img_label))
        print('[Info] 样本数: {}'.format(len(out_list)))
        write_list_to_file(val_file_path, out_list)
        print('[Info] 写入完成: {}'.format(val_file_path))


def main():
    slp = SampleLabeledParser()
    slp.make_dataset()


if __name__ == '__main__':
    main()
