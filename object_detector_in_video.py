import cv2 as cv
import numpy as np
from moviepy.editor import *
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
import os


class objectDetectorInVideo:
    
    # The different colors available for the predicted boxes. 
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255)];
    number_of_colors = len(colors);
    
    # Model for object detection.
    # https://huggingface.co/facebook/detr-resnet-50.
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50");
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50");
    
    
    def __init__(self):
        print("the objectDetectorInVideo instance correctly created");
    
    
    # Main interface between the user and the functionality of the objectDetectorInVideo.
    # This functionality consists of:
    # 1: Taking a video in input.
    # 2: Identify objects of interest and draw boxes around them (if the model detects them).
    # 3: Save the processed video in a new file.
    #
    # Note: The code creates a temporary file "temp_file.mp4" to store the processed video without audio.
    #       Then, the function that handle the audio part takes this "temp_file.mp4", set the audio to it
    #       and save it to the <out_video_path> defined by the user.
    #       Given the conditions of my experiments, using the same memory area for the processed video file, 
    #       and the final video-audio file gave problems, hence the introduction of this temporary file.
    #
    # @param string <in_video_path> - The input video to process.
    # @param string <out_video_path> - The new file that will contain the new processed video.
    # @param float <threshold> - The level of confidence we want the model to have (real number between 0 and 1).
    # @param list of string <objects_of_interest> - String representation of objects of interest to the user.
    #
    def objectDetectionInVideo(self, in_video_path, out_video_path, threshold, objects_of_interest):
        
        if threshold < 0 or threshold > 1:
            raise ValueError("The threshold must be a number between 0 and 1");
            
        if not os.path.exists(in_video_path):
            raise ValueError(f"The path of the video {in_video_path} is incorrect");

        video = cv.VideoCapture(in_video_path);
        frames_width = int(video.get(3));
        frames_height = int(video.get(4));

        fourcc = cv.VideoWriter_fourcc(*'mp4v');
        out = cv.VideoWriter("temp_file.mp4", fourcc,video.get(cv.CAP_PROP_FPS), (frames_width,frames_height));
        
        objects_indices = self.__objectsPreprocessing(objects_of_interest);
        
        while video.isOpened():
            is_frame_read_correctly, frame = video.read();

            if not is_frame_read_correctly:
                break;

            input_frame = self.__framePreprocessing(frame);

            x = self.feature_extractor(input_frame);
            y = self.model(torch.tensor(x.pixel_values), torch.tensor(x.pixel_mask));

            predicted_boxes = self.__processModelOutputs(y, frames_width, frames_height, threshold, objects_indices);
            
            # Draw predicted boxes (rectangle) around each object of interest identified.
            # pt1 are the coordinates (x1,y1) of a corner of the rectangle.
            # pt2 are the coordinates (x2,y2) of the corner of the rectangle at the opposite of pt1.
            for i in range(len(predicted_boxes)):
                pb = predicted_boxes[i];
                pt1 = (int(pb[0]-pb[2]/2), int(pb[1]-pb[3]//2));
                pt2 = (int(pb[0]+pb[2]//2), int(pb[1]+pb[3]//2));
                cv.rectangle(frame, pt1, pt2, self.colors[i % self.number_of_colors], 3);
                
            out.write(frame);

        video.release();
        out.release();
        
        self.__setAudio(in_video_path, out_video_path);
        os.remove("temp_file.mp4");
    
    # Take the processed video (video with predicted boxes around each objects of interest detected),
    # and set up its audio using the audio of the initial unprocessed video.
    #
    # @param string <in_video_path> - The input video to process.
    # @param string <out_video_path> - The new file that will contain the new processed video.
    #
    def __setAudio(self, in_video_path, out_video_path):
        videoclip_in = VideoFileClip(in_video_path);
        audioclip_in = videoclip_in.audio;

        videoclip_out = VideoFileClip("temp_file.mp4");
        videoclip_out_with_audio = videoclip_out.set_audio(audioclip_in);

        videoclip_out_with_audio.write_videofile(out_video_path);
    

    # Convert the objects of interest to the user from string representation to index representation.
    #
    # @param list of string <objects_of_interest> - String representation of objects of interest to the user.
    #
    # @return list of int <object_indices> - Index representation of objects of interest to the user.
    def __objectsPreprocessing(self, objects_of_interest):
        objects_indices = [];
        
        try:
            for obj in objects_of_interest:
                objects_indices.append(self.model.config.label2id[obj]);
        except KeyError:
            print(f"""Object {obj} is not handled by the code, please refer 
                  to the list of handled objects https://cocodataset.org/#explore """);
                
        return list(set(objects_indices));
                
        
    # Pre-processing step on the frame before entering the model.
    #
    # @param numpy.ndarray <frame> Unprocessed video frame.
    #
    # @return torch.tensor - Processed video frame.
    def __framePreprocessing(self, frame):
        return torch.from_numpy(frame).transpose(1,2).transpose(0,1);
    
    
    # The goal here will be to find the predicted boxes of objects of interest in an image (if they appear in it),
    # if the model detects the object with over <threshold> confidence.
    # What determines whether an object is of interest or not is whether its index is in <object_indices>.
    #
    # @param detr.DetrObjectDetectionOutput <y> - Output of the model DetrForObjectDetection.
    # @param int <width> - Width of the frames.
    # @param int <height> - Height of the frames.
    # @param float <threshold> - Level of confidence we want the model to have in its prediction.
    # @param list of int <objects_indices> - Index representation of objects of interest to the user.
    #
    # @return list of float <predicted_boxes> The predicted boxes found composed of 4 elements each
    #                                         (object x center, object y center, object width, object height).
    def __processModelOutputs(self, y, width, height, threshold, objects_indices):
        items = [];
        predicted_boxes = [];

        # The DETR model outputs 100 "object queries" (designation given by the authors) per image.
        # Each object query is an attempt to find one object in the given image.
        # More concretely each object query is a vector of probabilities of size num_classes+1,
        # (+1 here refers to the N/A class). 
        # Each object query is in y.logits whose shape is (batch_size, num_queries, num_classes+1).
        object_queries = torch.nn.functional.softmax(y.logits[0],dim=1);
        predictions = torch.max(object_queries, dim=1);
        
        for index_object_query, (object_index,confidence) in enumerate(zip(predictions.indices, predictions.values)):
            if object_index in objects_indices and confidence >= threshold:
                items.append(index_object_query);
        
        # Each predicted box is composed of four elements:
        # pb[0] = object x center, normalized between [0,1].
        # pb[1] = object y center, normalized between [0,1].
        # pb[2] = object width, normalized between [0,1].
        # pb[3] = object height, normalized between [0,1].
        for index_object_query in items:
            pb = y.pred_boxes[0][index_object_query];
            pb[0] = pb[0]*width;
            pb[1] = pb[1]*height;

            pb[2] = pb[2]*width;
            pb[3] = pb[3]*height;

            predicted_boxes.append(pb);

        return predicted_boxes;


# in_video_path = "./data/miss_dior.mp4";
# out_video_path = "./data/miss_dior_output.mp4";

# obj = objectDetectorInVideo()

# obj.objectDetectionInVideo(in_video_path, out_video_path, 0.9, ["person"])