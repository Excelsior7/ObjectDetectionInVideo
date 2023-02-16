# AIVE
OBJECT DETECTION PROJECT

### Object detection in video

**Note: TSML means To See Model Limitations.**

The goal of this little project was to create an object detection algorithm that takes as input a video along with the names of objects the user wants the algorithm to detect in the video, and a threshold of confidence T. Then, the algorithm outputs a new video with bounding boxes around each occurrence of each object of interest to the user if the object appears in the video and the algorithm detects it with confidence greater or equal than T.

More specifically, how to use the code in order to have the expected results as described above:

1. You instantiate one object **objectDetectorInVideo**.
2. From your instance you call the function **def objectDetectionInVideo** with the following parameters:  
    2.1 The path of the input video. (e.g. “./path/to/input/video/input.mp4”)  
    2.2 The path of the output video. (e.g. “./path/to/output/video/output.mp4”) TSLM  
    2.3 A threshold T, such that T is a real number between 0 and 1. (e.g. 0.9)  
    2.4 The objects of interest as a list of strings. (e.g. [“person”, “umbrella”]) To see all the objects manipulated by the code and the correct way to write them for input, please refer to [COCO Explorer](https://cocodataset.org/#explore).  
3. Done.


### Object detection in video algorithm.

The purpose of this section is to give an overview of my approach to the problem of object detection in videos.

Like a lot of problems in computer science and engineering in general, the key is to divide the big problem into subproblems whose solutions exist or can be found more easily and be combined to form the solution of the big problem.

But sometimes, the task of subdividing a problem into subproblems is the most difficult task because it requires you to have a good understanding of the structure of the problem.

Here for the object detection in video algorithm, the essential fact is that a video consists of a sequence of frames that is nothing more (if the video is not encoded using an interframe video compression algorithm) than a sequence of individual images. (See [What is the difference between an image and a video frame ?](https://blog.chiariglione.org/what-is-the-difference-between-an-image-and-a-video-frame/#:~:text=The%20question%20looks%20innocent%20enough,i.e.%20an%20image%2C%20is%20obtained.))

From this point of having an understanding of the structure of a video, and armed with the fact that there are algorithms for the detection of objects in images, here is the structure of my subproblems presented in the form of questions:

1. How to iterate over the frames (images) of a video ?
2. In the set of algorithms for object detection in image, which one has the following four qualities:  
    2.1 Not too difficult to set up (time constraint: 3 days).  
    2.2 Requires an amount of resources (memory, execution time) that fits in my constraints.  
    2.3 Good performance.  
    2.4 Able to identify objects of interest to me.  
3. How to draw the boxes around each predicted object in an image ?
4. How to assemble the different processed images into a new video ?
5. How to extract the audio from the initial video and set it to the final assembled processed video ? (Obviously, the last question arises if the audio is important to the project)

So now that these questions had been asked, if I could answer each of them by finding the right tool or by developing a homemade tool, the problem was solved, the last and simplest step was to simply put the pieces together. So after the questions, here are the answers associated to each of them:

1. Using the read() method of openCV VideoCapture object.
2. Using the DETR model (End-to-End Object Detection) from facebook.
3. Using the rectangle() function of openCV.
4. Using the write() method of openCV VideoWriter object.
5. Using the methods and properties of Moviepy VideoFileClip object.


### Model Limitations.

The goal of this section is to present the various limitations of the algorithm.

These limitations are due to the time constraint of the project and were not necessary (at least personally judged as such) for its accomplishment. 

These limitations can be more or less easily resolved. 

* Display the model's confidence in its predictions.
* Be able to modify certain hyperparameters of the model (e.g. the number of object queries).
* Be able to specify a time interval in the video that will be the only one processed by the algorithm.
* The set of objects that a model can identify is limited by its training dataset, and the training dataset of the * model used (DETR) is [COCO](https://cocodataset.org/#home) of 2017.
* The number of different colors to identify the objects is limited to 7.
* Offers only the format mp4 as output.

### Various links that have been relevant or interesting to the project.
    
* [DETR](https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/detr#transformers.DetrFeatureExtractor)
* [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50)
* [Haar Cascasde Classifier](https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08)
* [Haar Cascasde Classifier with openCV](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
* [Audio in MoviePy](https://zulko.github.io/moviepy/getting_started/audioclips.html)
