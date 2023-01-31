# 실험 순서
# 01-wav_to_mfcc.py : wav 파일을 mel이미지로 변환
# 02-class_train.py : mel 이미지들을 VGG16으로 분류 하여 .h5 모델 파일 생성
# 02 번 후 정확도 높은 순으로 tasks 선택
# 03-class_feature_extractor.py : 선택된 태스크의 mel 이미지들을 .h5 모델 파일을 이용하여 flatten 파일 생성
# task 별로 생성된 flatten 파일을 train, val, test 파일들을 모두 합치고 파일명으로 sort
# 모든 task에 다 있는 파일 선택해서 별도로 저장
# 04-feature_extraction_tasks_append.py : task 파일들 컬럼으로 합치기  ==> 자동으로 k-fold 실험할 때 사용
# k-fold 파일로 분류해서 실험  ==> 강제로 5-fold 만들어서 실험하기 위함
# 08-MakeFold.py 사용해서 fold별 파일 나누기, 그 전에 5개의 fold에 포함되는 파일명 정리하기
# 05-Final-Classifier~~.py 로 최종 학습 돌리기

import os
import numpy as np
import pandas as pd
import csv


#base_path = 'E:/01-DATASET/0-EXP-IB_치매음성-MEDAUTH/01-SCIvsOTHERS-EXP/3-FlattenFiles-FourTasks-SvsO'
#save_path = 'E:/01-DATASET/0-EXP-IB_치매음성-MEDAUTH/01-SCIvsOTHERS-EXP/3-FlattenFiles-FourTasks-SvsO/02-fourtasks-SvsO.'
D:\aible\flatten\MCI_AD_dense1
base_path = 'D:/aible/flatten/MCI_AD_dense1/'
save_path = 'D:/aible/flatten/MCI_AD_dense1/final'

file_name_01 = base_path + '/01_test_1_flattenfeatures_SvsA_75.7_76.6-medauth-sort.csv'
file_name_02 = base_path + '/01_test_3_flattenfeatures_SvsA_63.1_68.2-medauth-sort.csv'
file_name_03 = base_path + '/01_test_4_flattenfeatures_SvsA_66.0_68.2-medauth-sort.csv'
file_name_04 = base_path + '/01_test_5_flattenfeatures_SvsA_68.0_72.0-medauth-sort.csv'
file_name_05 = base_path + '/01_test_6_flattenfeatures_SvsA_75.2_78.3-medauth-sort.csv'
file_name_06 = base_path + '/01_test_8_flattenfeatures_SvsA_68.9_79.4-medauth-sort.csv'
file_name_07 = base_path + '/01_test_1_flattenfeatures_SvsA_75.7_76.6-medauth-sort.csv'
file_name_08 = base_path + '/01_test_3_flattenfeatures_SvsA_63.1_68.2-medauth-sort.csv'
file_name_09 = base_path + '/01_test_4_flattenfeatures_SvsA_66.0_68.2-medauth-sort.csv'
file_name_10 = base_path + '/01_test_5_flattenfeatures_SvsA_68.0_72.0-medauth-sort.csv'
file_name_11 = base_path + '/01_test_6_flattenfeatures_SvsA_75.2_78.3-medauth-sort.csv'



dataset_array_all = []

dataset_array = []
data_pd = pd.read_csv(file_name_01)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = dataset_array
print('dataset_array_all 1',  dataset_array_all.shape)

dataset_array = []
data_pd = pd.read_csv(file_name_03)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
print('dataset_array_all 3',  dataset_array_all.shape)


dataset_array = []
data_pd = pd.read_csv(file_name_04)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
print('dataset_array_all 4',  dataset_array_all.shape)


dataset_array = []
data_pd = pd.read_csv(file_name_05)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
print('dataset_array_all 5',  dataset_array_all.shape)



dataset_array = []
data_pd = pd.read_csv(file_name_06)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
print('dataset_array_all 6',  dataset_array_all.shape)


dataset_array = []
data_pd = pd.read_csv(file_name_08)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
print('dataset_array_all 8', dataset_array_all.shape)


dataset_array = np.array(data_pd.iloc[:, 4610:4611])
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)

df = pd.DataFrame(dataset_array_all)
df.to_csv(save_path + 'csv', index=False)

"""
cnt = 1
for file_name in file_list:
    dataset_array = []
    if file_name.find('csv') is not -1:
        file_path = base_path + '/' + file_name
        data_pd = pd.read_csv(file_path)
        #data_pd.iloc[:, 0:4607]
        dataset_array = np.array(data_pd.iloc[:, 0:4608])
        print(dataset_array.shape)
        if cnt == 1:
            dataset_array_all = dataset_array
            print('dataset_array_all ',cnt, ' ',  dataset_array_all.shape)
        else :
            dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
            print('dataset_array_all ',cnt, ' ',  dataset_array_all.shape)
        cnt += 1
dataset_array = np.array(data_pd.iloc[:, 4608:4611])
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)

df = pd.DataFrame(dataset_array_all)
df.to_csv(save_path + 'csv', index=False)
"""


