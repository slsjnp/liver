import os
import datetime
import boto3
import botocore
from boto3.session import Session


class CephS3BOTO3():

    def __init__(self):
        access_key = 'admin'
        secret_key = '12345678'
        self.session = Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        self.url = 'http://minio:9000'
        # self.url = 'http://127.0.0.1:9000'
        self.s3_client = self.session.client('s3', endpoint_url=self.url)

        self.s3_client1 = self.session.resource('s3', endpoint_url=self.url)
    def get_bucket(self):
        buckets = [bucket['Name'] for bucket in self.s3_client.list_buckets()['Buckets']]
        print(buckets)
        s = self.s3_client.list_buckets()
        return buckets

    def get_list_object(self, bucket, prefix):
        list = self.s3_client.list_objects(Bucket=bucket, Prefix=prefix)
        # for i in list['Contents']:
        #     print(i['Key'])
        s = list['Contents']
        print(len(s))
        return s

    def create_bucket(self):
        # 默认是私有的桶
        self.s3_client.create_bucket(Bucket='hy_test')
        # 创建公开可读的桶
        # ACL有如下几种"private","public-read","public-read-write","authenticated-read"
        self.s3_client.create_bucket(Bucket='hy_test', ACL='public-read')

    def upload(self, bucket, key, path):
        resp = self.s3_client.put_object(
            Bucket=bucket,  # 存储桶名称
            Key=key,  # 上传到
            Body=open(path, 'rb').read()
        )
        print(resp)
        return resp

    def download(self, bucket, key, save_path=None):
        resp = self.s3_client.get_object(
            Bucket=bucket,
            Key=key
        )

        # print(resp['Body'].read())
        if save_path is not None:
            with open(save_path, 'wb') as f:
                f.write(resp['Body'].read())
        return resp

    def delete(self, bucket, obj):
        # del_list = self.get_list_object(bucket, obj)
        # self.s3_client.remove_object('DELETE', bucket_name=bucket,
        #                          object_name=obj)
        bucket = self.s3_client1.Bucket(bucket)
        for obj in bucket.objects.filter(Prefix=obj):
            obj.delete()


def getInformation(identify_card):
    sex = int(identify_card[-2])
    birth_year = int(identify_card[6:10])
    birth_month = int(identify_card[10:12])
    birth_day = int(identify_card[12:14])
    if (len(identify_card) == 15):
        birth_year = int(identify_card[6:8])
        birth_month = int(identify_card[8:10])
        birth_day = int(identify_card[10:12])
    sex = sex % 2
    now = (datetime.datetime.now() + datetime.timedelta(days=1))
    year = now.year
    month = now.month
    day = now.day
    if year == birth_year:
        return sex, 0
    else:
        if birth_month > month or (birth_month == month and birth_day > day):
            return sex, year - birth_year - 1
        else:
            return sex, year - birth_year


# if __name__ == "__main__":
    # cephs3_boto3 = CephS3BOTO3()
    # cephs3_boto3.delete('bucket', '12d98a9cc19411eabfa30242ac130002/')
    # cephs3_boto3.get_bucket()
#     cephs3_boto3.get_list_object('bucket', '4117ae1cbc0511eaa3569cb6d0c05faa/dcm')
#     cephs3_boto3.upload('bucket', '4117ae1cbc0511eaa3569cb6d0c05faa/dcm/0001.dcm', '0001.dcm')
#     cephs3_boto3.download()
