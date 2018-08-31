# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
from rest_framework import exceptions
from WebAnalyzer.tasks import extract_and_search
from WebAnalyzer.utils import filename
import ast


class ImageModel(models.Model):
    image = models.ImageField(upload_to=filename.uploaded_date)
    token = models.AutoField(primary_key=True)
    options=models.TextField(default="")
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    url=models.TextField(default="")

    URLS=["http://mleagles.sogang.ac.kr:8001/"]


    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)

        opts=ast.literal_eval(self.options)
        self.set_url(opts)

        ret=ast.literal_eval(str(extract_and_search.delay(self.image.path,opts,self.url).get()))
        for n,i in enumerate(ret):
            self.result.create(rank=n,similarity=i['similarity'],image=i['image'],name=i['name'])
        super(ImageModel, self).save()

    def set_url(self,opts):
        self.url=self.URLS[0]


class ResultModel(models.Model):
    result_model = models.ForeignKey(ImageModel, related_name='result', on_delete=models.CASCADE)
    rank = models.IntegerField()
    similarity=models.TextField()
    image = models.TextField()
    name= models.TextField()

