Since the recognition process need "torch" and some other package, you may need to install or update the packages
You can try to use the command below to install and update the packages.
Installation command: conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

Step 1:
run "Create Custom Dataset Directory Structure.ipynb" to get a custom dataset directory structure.

Step 2:
run "Train Traffic Sign Classifier.ipynb" to train and save the model.
The pretrained model (saved_model.pt) can also be downloaded directly from the github link

Step 3:
run "segmentation_recognition.py" in command prompt to perform segmentation and recognition for single/multiple images in a give folder.
To run "segmentation_recognition.py", please make sure that the saved_model.pt is included in the same folder.
By default, the output of the segmentation and recognition will be saved in a folder called "result".

#test case 1: test for images_without_annotation (without accuracy and groundtruth)
[!] Please make sure that a folder named "image_without_annotation" that included all input images is created
command: python segmentation_recognition.py 

#test case 2: test for 70_test_images (with accuracy and groundtruth)
[!] Please make sure that a folder named "images_with_annotation" that included all input images is created
[!] Please make sure that a "TsignRecgTrain4170Annotation.txt" that included all annotation for train images is included in the folder
command: python segmentation_recognition.py -f 70_test_images -a True -t True
