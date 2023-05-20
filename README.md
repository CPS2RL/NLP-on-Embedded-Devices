# NLP-on-Embedded-Devices

#How to use the code?  
We used the ROS2 platform for conducting Intent Classification (IC) and Name Entity Recognition (NER) tasks. Our implementation employed based on the pub-sub model of robot operation system (ROS). You can find more information about the ROS pub-sub model in the provided link: https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html  
We have trained the Bert model offline and put it on the BERT folder inside Intent Classification or NER folder.  
Terminal command for running pub-sub model:    
IC:  `ros2 run intenc talker`    
NER:  `ros2 run ner talker`

To measure the energy we used UM25C energy meter.     
To measure system memory consumption, we used @profile method of python.   
Deatils can be found by analyzing the code of `publisher_member_function.py`

#Data-set link:   
HuRic Dataset: https://github.com/crux82/huric  
Go Emotions Dataset: https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset  
WNUT'17 Dataset: https://github.com/leondz/emerging_entities_17  
