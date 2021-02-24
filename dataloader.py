import operator
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from blackai.datamanager import FetchDataManager
import json
from logging import getLogger
from blackai.utils import fast_parse_fn
import requests as orig_requests
from threading import Thread
import glob

logger = getLogger()
import asyncio

def generate_labelled_local_track_list_threaded(config, output_path):
	os.makedirs(output_path, exist_ok=True)

	if "label_dataset" in config:
		pkl_path = os.path.join(output_path, "all_track_point_linked_local_tracks" + config["database_name"] + "_" + config["label_dataset"] + ".pkl")
	else:
		pkl_path = os.path.join(output_path,"all_track_point_linked_local_tracks" + config["database_name"] + ".pkl")

	merge_track_dict = {}

	if not os.path.isfile(pkl_path):

		if "vendor_labelled" in config and config["vendor_labelled"]:
			link = "http://supervisor.black.ai.lan:4200/api/{dbName}/vendor/getCleanMerges".format(
				dbName=config['database_name'])
			data = {"dataset": config["label_dataset"]}
			if "vendor_page_range" in config and  config["vendor_page_range"] is not None:
				data["pageRange"] = config["vendor_page_range"]
			if "filter_staff" in config and config["filter_staff"] is not None:
				data["filterStaff"] = config["filter_staff"]
			payload = json.dumps(data)
			headers = {"Content-Type": "application/json"}
			response = orig_requests.post(link, payload, headers=headers)
			if not response.ok:
				print("Error retrieving results for query: " + response.text)
			# parsed = json.loads(response["body"])
			parsed = json.loads(response.content)
			merged_track_uuids = np.unique([data["mergeUuid"] for data in parsed])
			merged_local_track_uuids = np.unique([data["localTrackUuid"] for data in parsed])

			local_track_to_merge_track_uuid_dict = {}
			for merge_uuid in tqdm(merged_track_uuids, desc="Generating merge_uuid_dict"):
				merge_track_dict[merge_uuid] = np.unique([data["localTrackUuid"] for data in parsed if data["mergeUuid"] == merge_uuid]).tolist()

	return merge_track_dict



def processing_img(merged_local_track_list, data_manager, output_path, cameralist):



	for merged_uuid in tqdm(merged_local_track_list, desc="Generating dataset."):
		if merged_uuid == '00000000-0000-0000-0000-000000000000':
			continue


		local_track_uuids = merged_local_track_list[merged_uuid]

		detections = data_manager.get_detections_by_localtrack_uuid(local_track_uuids, hide_progress=False)

		for camera_id in cameralist:
			temppath = os.path.join(output_path, camera_id)
			os.makedirs(temppath, exist_ok=True)

			imagepath = os.path.join(temppath, "img1")
			os.makedirs(imagepath, exist_ok=True)


			filterdetections = [it for it in detections  if it.source_node_uuid == camera_id]

			for detection in filterdetections:
				imagefile = os.path.basename(data_manager.get_frame_url_for_trackpoint_uuid(detection.track_point_uuid)[0]['frame'])

				imagefilepath = os.path.join(imagepath, imagefile)



				if not os.path.exists(imagefilepath):
					image = data_manager.get_full_frame_for_detection(detection)
					if image is not None:
						image.save(imagefilepath)
						print(imagefilepath)
				else:
					print("pass")





def processing_txt(merged_local_track_list, data_manager, output_path, cameralist):
	for merged_uuid in tqdm(merged_local_track_list, desc="Generating dataset."):
		if merged_uuid == '00000000-0000-0000-0000-000000000000':
			continue

		print("\n merge_uuid = {}".format(merged_uuid))

		local_track_uuids = merged_local_track_list[merged_uuid]

		detections = data_manager.get_detections_by_localtrack_uuid(local_track_uuids, hide_progress=False)

		for camera_id in cameralist:

			imagepath = os.path.join(output_path, camera_id, "img1")
			os.makedirs(imagepath, exist_ok=True)
			textpath = os.path.join(output_path, camera_id, "labels_with_ids")
			os.makedirs(textpath, exist_ok=True)

			filterdetections = [it for it in detections if it.source_node_uuid == camera_id]

			for indx, detection in enumerate(filterdetections):
				print("detection index = {}".format(indx))
				print("text path = {}".format(textpath))
				imagefile = os.path.basename(
					data_manager.get_frame_url_for_trackpoint_uuid(detection.track_point_uuid)[0]['frame'])

				imagefilepath = os.path.join(imagepath, imagefile)
				textfilepath = os.path.join(textpath, imagefile).replace('.png', '.txt')

				if os.path.exists(imagefilepath):
					class_ = 0
					identity_ = merged_uuid
					x_center = ((detection.body_box['min_x'] + detection.body_box['max_x']) / 2) / 1920
					y_center = ((detection.body_box['min_y'] + detection.body_box['max_y']) / 2) / 1080
					width = ((detection.body_box['max_x'] - detection.body_box['min_x']) / 2) / 1920
					height = ((detection.body_box['max_y'] - detection.body_box['min_y']) / 2) / 1080
					detection_uuid = detection.detection_uuid
					local_track_uuid = detection.local_track_uuid


					str_ = os.path.join(str(class_), str(identity_), str(x_center), str(y_center), str(width),
										str(height), str(detection_uuid), str(local_track_uuid)).replace("/", " ")

					if os.path.exists(textfilepath):
						fp = open(textfilepath, "r")
						word = fp.read()
						if str_ in word:
							print("pass")
							continue
						temp = word.split(" ")
						if merged_uuid not in temp:
							fp.close()
							fp = open(textfilepath, "w")
							fp.write(word + str_ + " \n")
							fp.close()
					else:
						fp = open(textfilepath, "w")
						fp.write(str_ + " \n")
						fp.close()


def sort_dataset(image_root, cameralist):

	for camera_id in cameralist:
		frame_indx = 0
		temppath = os.path.join(image_root, camera_id, 'img1')

		gt_path = os.path.join(image_root, camera_id, 'gt')
		os.makedirs(gt_path, exist_ok=True)
		gt_path = os.path.join(image_root, camera_id, 'gt' + "/gt.txt")


		imagelist = glob.glob(temppath + "/*.png")
		imagelist = sorted(imagelist, key=lambda x: (int(x.split(".")[1]), int(x.split(".")[6])))

		all_list = []

		for image_path in imagelist:
			text_path = image_path.replace("img1", "labels_with_ids").replace(".png", ".txt")
			frame_indx = frame_indx + 1
			with open(text_path, 'r') as f:
				for line in f.readlines():
					line_list = line.split(" ")
					line_list[0] = str(frame_indx)
					x_center = float(line_list[2]) * 1920
					y_center = float(line_list[3]) * 1080
					w = float(line_list[4]) * 1920
					h = float(line_list[5]) * 1080

					x_min = x_center - w
					y_min = y_center - h


					line_list[2] = str(x_min)
					line_list[3] = str(y_min)
					line_list[4] = str(w * 2)
					line_list[5] = str(h * 2)

					str1 = ','.join(line_list)


					all_list.append(str1)


		str1 = ''.join(all_list)

		fp = open(gt_path, "w")
		fp.write(str1 + " \n")
		fp.close()

		print()










def divide_dict(dictionary, chunk_size):

	import numpy, collections

	count_ar = numpy.linspace(0, len(dictionary), chunk_size+1, dtype= int)
	group_lst = []
	temp_dict = collections.defaultdict(lambda : None)
	i = 1
	for key, value in dictionary.items():
		temp_dict[key] = value
		if i in count_ar:
			group_lst.append(temp_dict)
			temp_dict = collections.defaultdict(lambda : None)
		i += 1
	return group_lst


if __name__ == '__main__':
	config ={
			"kitchensync_host": "http://mars.black.ai.lan",
			"kitchensync_port": "9980",
			"kitchensync_from_time": "2020-08-26T00:00:00+00:00",
			"kitchensync_to_time": "2020-08-28T00:00:00+00:00",
			"kitchensync_sitecode": "hfm1-20200827-20min-01",
			"database_host": "mars",
			"vendor_labelled": True,
			"vendor_page_range": None,
			"filter_staff": False, # Only turn off for benchmarks
			"label_dataset": "2020-08-27",
			"database_port": 36101,
			"database_name": "hfm1data2020cloud_samasource",
			"database_user": "sequelize",
			"database_password": "Q6bbFXiPy9TX9VXKXnuzLxab7",
			"data_cache_host": "mars",
			"enable_feature_cache": False,
			"data_cache_port": 3306,
			"data_cache_name": "hfm1data2020cloud_samasource_cache_1280",
			"data_cache_user": "sequelize",
			"data_cache_password": "Q6bbFXiPy9TX9VXKXnuzLxab7",

		}


	parser = argparse.ArgumentParser()
	output_path = '/root/data/FairMOT/black_ai_evaluation/images/train'
	parser.add_argument("--output_path", required=False, type=str, default=output_path,
	                    help="where to save the tracker data. use default('./data')")
	args = parser.parse_args()

	#local_track_uuids = generate_labelled_local_track_list_threaded(config, args.output_path)
	#data_manager = FetchDataManager(config,	labels=False if "vendor_labelled" in config and config["vendor_labelled"] else True)

	cameralist = ['hfm1-8', 'hfm1-76', 'hfm1-80']



	#temp = divide_dict(local_track_uuids, 24)

	# threads = []
	# for i in range(24):
	# 	threads.append(Thread(target=processing_img, args=(temp[i],data_manager, args.output_path, cameralist )))
	#
	# for thread in threads:
	# 	thread.start()
	# for thread in threads:
	# 	thread.join()

	#processing_txt(local_track_uuids, data_manager, args.output_path, cameralist)
	# threads = []
	# for i in range(24):
	# 	threads.append(Thread(target=processing_txt, args=(temp[i], data_manager, args.output_path, cameralist)))
	#
	# for thread in threads:
	# 	thread.start()
	# for thread in threads:
	# 	thread.join()


	sort_dataset(output_path, cameralist)

