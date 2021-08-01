import sys
import io
import os
import struct
import time 
import numpy as np
import logging
import csv
import random
import base64

from threading import local 

from . import yt_example_pb2

from io import BytesIO, StringIO
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, io

csv.field_size_limit(sys.maxsize)

def tfrecord2idx(tfrecord, idxfile):
    '''
    refer :  https://github.com/NVIDIA/DALI/blob/master/tools/tfrecord2idx
    '''
    try:
        #check idxfile exist and size large than 0
        st = os.stat(idxfile)
        if st.st_size > 0:
            return idxfile
    except:
        # no found or size is zero regenerate
        pass

    samples  = 0
    with open(tfrecord, 'rb') as f:
        with open(idxfile, 'w') as idx :
            while True:
                current = f.tell()
                byte_len_crc = f.read(12)
                # eof 
                if len(byte_len_crc) == 0:
                    break
                if len(byte_len_crc) != 12:
                    logging.error("read byte_len_crc failed, file:%s, num:%d pos:%s byte_len_crc:%s" % (tfrecord, samples, f.tell(), len(byte_len_crc)))
                    break
                proto_len = struct.unpack('L', byte_len_crc[:8])[0]
                buffer = f.read(proto_len + 4)
                if len(buffer) != proto_len + 4:
                    logging.error("read proto_len failed, file:%s, num:%d pos:%s proto_len:%s" % (tfrecord, samples, f.tell(), proto_len))
                    break                
                idx.write(str(current) + ' ' + str(f.tell() - current) + '\n')
                samples += 1
    if samples == 0:
        logging.error("no idx found,  file:%s" % tfrecord)
        os.remove(idxfile)
        return None
    logging.info("idx generate done, samples:%s file:%s" %(samples, idxfile))
    return idxfile

class TFRecordDataSet(Dataset):
    def __init__(self, tfrecords):

        tfindexs = [tfrecord2idx(f, f.replace('.tfrecord', '.idx')) for f in tfrecords]
        self.idxs = []
        self.thread_local = local()
        self.thread_local.cache = {}
        self.samples = 0
        for index, tffile in zip(tfindexs, tfrecords):
            idx = []
            with open(index) as idxf:
                for line in idxf:
                    offset, _ = line.split(' ')
                    idx.append(offset)
            self.samples += len(idx)
            print("load %s, samples:%s" %(tffile,  len(idx)))
            self.idxs.append((idx, tffile))

    def __len__(self):
        return self.samples
    
    def parser(self, feature_list):
        raise NotImplementedError("Must Implement parser")
    
    def get_record(self, f, offset, name_only=False):
        f.seek(offset)

        # length,crc
        byte_len_crc = f.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        # proto,crc
        pb_data = f.read(proto_len)
        if len(pb_data) < proto_len:
            print("read pb_data err,proto_len:%s pb_data len:%s"%(proto_len, len(pb_data)))
            return None
        
        example = yt_example_pb2.Example()
        example.ParseFromString(pb_data)
        #keep key value in order
        feature = sorted(example.features.feature.items())
     
        record = self.parser(feature)
        #print(record)
        return record

    def __getitem__(self, index):
        for idx, tffile in self.idxs:
            if index >= len(idx):
                index -= len(idx)
                continue
            # every thread keep a f instace 
            f = self.thread_local.cache.get(tffile, None)
            if f is None:
                f = open(tffile, 'rb')
                self.thread_local.cache[tffile] = f

            offset = int(idx[index])
            record = self.get_record(f, offset)
            if record:
                return record
            else:
                print("bad data index %d" % index)
                return self[random.randint(0, self.samples-1)]

        print("bad index,", index)


class ImageTFRecordDataSet(TFRecordDataSet):
    def __init__(self, tfrecords, transforms):
        super(ImageTFRecordDataSet, self).__init__(tfrecords)
        self.transforms = transforms
    def parser(self, feature_list):
        '''
        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        '''
        for key, feature in feature_list:
            try:
                #for image file col
                if key == 'image':
                    image_raw = feature.bytes_list.value[0]
                    image = Image.open(BytesIO(image_raw))
                    image = image.convert('RGB')
                    image = self.transforms(image)
                #for int col
                elif key == 'label':
                    label = feature.int64_list.value[0] - 1 ## imagenet classes are from 1 ~ 1000
            except:
                return False
        return image, label

