from __future__ import print_function

from AnalysisModule.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process


import requests
import json
import torch
import numpy as np
import h5py
import ast
import os
import base64

@worker_init.connect
def model_load_info(**__):
    print("====================")
    print("Worker Analyzer Initialize")
    print("====================")


@worker_process_init.connect
def module_load_init(**__):
    global analyzer
    worker_index = current_process().index

    print("====================")
    print(" Worker Id: {0}".format(worker_index))
    print("====================")

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    # from Modules.dummy.main import Dummy
    # analyzer=Dummy()
    #from Modules.Extractor.main import Extractor
    #analyzer = Extractor()


@app.task
def extract_and_search(image_path,options,url):
    feature=send_request_to_extractor(image_path,options,url)
    result=similarity_search(feature,options)
    return result


def send_request_to_extractor(image,options,url):
      json_image=open(image,'rb')
      json_files={'image':json_image}
      json_data=dict()

      response=requests.post(url=url,data=json_data,files=json_files)
      response_data=json.loads(response.text)['result']

      for l in response_data:
          if l['desc']==options['feature']:
              response_data=np.array(ast.literal_eval(l['feature']))

      print(response_data.shape)
      return response_data

def load_feature(path,feat):
    files=os.listdir(path)
    #feature={'Pool5':[],'MAC':[],'SPoC':[],'RMAC':[],'RAMAC':[],'names':[]}
    feature={'MAC':[],'SPoC':[],'RMAC':[],'RAMAC':[],'names':[]}
    
    for i in files:
        f=h5py.File(os.path.join(path,i),'r')
        for k,v in feature.items():
            v.append(f[k][()])
        f.close()

    for k, v in feature.items():
        arr=np.concatenate(v,axis=0)
        feature[k]=arr
    
    names=feature['names']
    feature=feature[feat]

    print(feature.shape)
    print(names)

    return feature,names


def similarity_search(feature,options):
    featPATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','features')
    imgPATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','images')
    topN=int(options['topN'])
    dataset=options['dataset']
    feat=options['feature']

    #READ DB
    f,name=load_feature(os.path.join(featPATH,dataset),feat)

    #GET SIMILARITY
    f = torch.tensor(f.reshape([f.shape[0],-1]))
    feature=torch.tensor(feature.reshape([feature.shape[0],-1]),dtype=torch.float32)

    #CPU/GPU
    f=f.cuda()
    feature=feature.cuda()

    norm_feature=torch.div(feature,torch.norm(feature,2,1,True))
    norm_f= torch.div(f,torch.norm(f,2,1,True))
    cos=torch.t(torch.mm(norm_f,torch.t(norm_feature)))

    ret,idx=torch.sort(cos)
    ret=ret[0,-topN:].cpu().numpy()
    idx=idx[0,-topN:].cpu().numpy()

    print(name[idx])

    ret=[{'name':i.decode(),'similarity':str(ret[topN-n-1]),'image':str(base64.b64encode(open(os.path.join(imgPATH,dataset,i.decode()),'rb').read()))} for n,i in enumerate(reversed(name[idx]))]
    return ret

  
