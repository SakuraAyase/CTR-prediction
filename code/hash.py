# -*- coding: utf-8 -*-
import hashlib,csv,collections

#hash到100w维
NR_BINS = 1000000
def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

id_cnt = collections.defaultdict(int)
ip_cnt = collections.defaultdict(int)
#hour_cnt = collections.defaultdict(int)

def scan(path):
    for row in csv.DictReader(open(path)):
        id_cnt[row['device_id']] += 1
        ip_cnt[row['device_ip']] += 1

tr_src_path = 'Data/train.csv'
va_src_path = 'Data/test.csv'

scan(tr_src_path)
scan(va_src_path)

#说明:使用21维原始特征+1维特征hour.原始的hour不能直接用,需要进行格式处理
#总共的特征数22维
#trs=[2,16,4,11,22,3,20,6,19,7,8,14,18,21,9,5,15,10,13,12,1,17]

fields = ['C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']


print ("样本特征个数:",len(fields)+1)

#device_ip,device_id为长尾特征的阈值
T1 = 1000
T2 = 1000
H1 = 14102800
H2 = 14102900

def convert(src_path, dst_path, is_train):
    with open(dst_path, 'w') as f:
        for row in csv.DictReader(open(src_path)):
            feats = []
            for field in fields:
                if field=='device_ip':
                    if (ip_cnt[row['device_ip']])>T1:
                        feats.append(hashstr('device_ip-'+str(row['device_ip'])))
                    else:
                        feats.append(hashstr('device_ip-less-'+str(ip_cnt[row['device_ip']])))
                elif field=='device_id':
                    if (id_cnt[row['device_id']])>T2:
                        feats.append(hashstr('device_id-'+str(row['device_id'])))
                    else:
                        feats.append(hashstr('device_id-less-'+str(id_cnt[row['device_id']])))
                else:
                    feats.append(hashstr(field+'-'+row[field]))
            feats.append(hashstr('hour-'+row['hour'][-2:]))
            if is_train:
                click = row['click']
            else:
                click = str(0)
            line = ','.join(feats)+','+click+'\n'
            f.write(line)


def convert_1(src_path, dst_path1, dst_path2):
    L1 = 14102100
    for row in csv.DictReader(open(src_path)):
        if (int(row['hour'])) != L1:
            print("courrent hour:", int(row['hour']))
            L1 = int(row['hour'])
        if (int(row['hour']))<H1:
            with open(dst_path1, 'a+') as f1:
                feats = []
                for field in fields:
                    if field=='device_ip':
                        if (ip_cnt[row['device_ip']])>T1:
                            feats.append(hashstr('device_ip-'+str(row['device_ip'])))
                        else:
                            feats.append(hashstr('device_ip-less-'+str(ip_cnt[row['device_ip']])))
                    elif field=='device_id':
                        if (id_cnt[row['device_id']]) > T2:
                            feats.append(hashstr('device_id-'+str(row['device_id'])))
                        else:
                            feats.append(hashstr('device_id-less-'+str(id_cnt[row['device_id']])))
                    else:
                        feats.append(hashstr(field+'-'+row[field]))
                feats.append(hashstr('hour-'+row['hour'][-2:]))
                click = row['click']
                line = ','.join(feats)+','+click+'\n'
                f1.write(line)
        elif (int(row['hour'])) < H2:
            with open(dst_path2, 'a+') as f2:
                feats = []
                for field in fields:
                    if field=='device_ip':
                        if (ip_cnt[row['device_ip']])>T1:
                            feats.append(hashstr('device_ip-'+str(row['device_ip'])))
                        else:
                            feats.append(hashstr('device_ip-less-'+str(ip_cnt[row['device_ip']])))
                    elif field=='device_id':
                        if (id_cnt[row['device_id']]) > T2:
                            feats.append(hashstr('device_id-'+str(row['device_id'])))
                        else:
                            feats.append(hashstr('device_id-less-'+str(id_cnt[row['device_id']])))
                    else:
                        feats.append(hashstr(field+'-'+row[field]))
                feats.append(hashstr('hour-'+row['hour'][-2:]))
                click = row['click']
                line = ','.join(feats)+','+click+'\n'
                f2.write(line)

tr_dst_path = 'Data/data_train_hash7.csv'
te_dst_path = 'Data/data_validation_hash8.csv'
#va_dst_path = 'Data/val_hash.csv'
convert_1(tr_src_path, tr_dst_path, te_dst_path)
#convert(va_src_path, va_dst_path, False)

