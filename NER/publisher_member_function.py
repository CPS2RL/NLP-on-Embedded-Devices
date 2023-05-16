# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

import json
# External imports
import json
import sklearn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from model import tokenization_with_bert, BertClassifier
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import pickle
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from memory_profiler import profile
import requests
import psutil

from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
                
        BERT_LOC='/home/pi/ros2_foxy/src/ner/ner/bert/Layer2_No Masking'
        
        
        def load_model(BERT_LOC):
            tokenizer = AutoTokenizer.from_pretrained(BERT_LOC, add_prefix_space=True)
            model = AutoModelForTokenClassification.from_pretrained(BERT_LOC,ignore_mismatched_sizes=True)
            return {'tokenizer':tokenizer,'model':model}


        def load_ner_model(model_name):

            if "bert" == model_name:
                return load_model(BERT_LOC)
            else:
                print("No NER technique detected.")
            return



        def get_labels(text, ner):
            ner_model = pipeline('ner', model=ner["model"], tokenizer=ner["tokenizer"], grouped_entities=True)
            # Test the sequence
            return ner_model(text)



        def annotate(text, active_feature, ner):
            label_sequence = get_labels(text, ner)
            return label_sequence
    
        def annotate_text(text, active_feature, ner_name, ner):
            # if "dialogflow" == ner_name:
            #     return dialogflow.annotate(text, schema, active_feature, ner)
            # el
            if "bert" == ner_name:
                return annotate(text, active_feature, ner)
            elif "roberta" == ner_name:
                return annotate(text, active_feature, ner)
            elif "xlnet" == ner_name:
                return annotate(text, active_feature, ner)
            elif "distilbert" == ner_name:
                return annotate(text, active_feature, ner)
            return text
        
        @profile
        def fa():
            feature_annotation = annotate_text(line, 'ORG', model_name, model)
        
        dataset_path='/home/pi/ros2_foxy/src/ner/ner/test.txt'
        model_name='bert'
        model = load_ner_model(model_name)
        t1=time.time()
        for i in range(100):
            x=2
        t1=(time.time()-t1)/100
        arr=[]
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
            #line='Indiana is 13th in the Eastern Conference with a 25-55 record. At the trade deadline, the Pacers reset their competitive timeline by sending All-Star Domantas Sabonis, Justin Holiday, Jeremy Lamb, and a second-round pick in 2023 to the Sacramento Kings for Tyrese Haliburton, Buddy Hield, and Tristan Thompson.'
                t2=time.time()
                for i in range(100):  
                    count=0
                    feature_annotation = fa()
            
                t2=(time.time()-t2)/100
                arr.append(t2-t1)
                print(t2-t1)
                print(feature_annotation)
                msg = String()
                msg.data = str(feature_annotation)
                self.publisher_.publish(msg)
                self.get_logger().info('Publishing: "%s"' % msg.data)
                self.i += 1
        
        print(arr)
        file = open("timing_data.txt", "w+")
        file.write(arr)
        file.close()


    
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
