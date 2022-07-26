# tensorflow-captcha

### Training Constraints
1. Only support 4 digits as both correct answers and contents in images
2. Only support an alphanumeric combination of lowercase English letters and numbers.

### Enviroment Set-up
Using `Anaconda` to set up virtual dev environment is recommended.

### Specified Version Requirements
1. Python 3.8
2. TensorFlow 2.7

### Model Training Steps
1. Collect catpcha-like images and rename them with correct answers.  For example, if the image says '5dc8' in your eyes, it shall be renamed `5dc5.jpeg`.
2. Put a major portion of images in a root directory named `training` for training.
3. Put another batch of images in another root directory named `validation` for validation.
4. Run `python train.py`, a prompt will ask how many epochs do you want.  Insert an integer.
5. Run `python predict.py`, images in the `validation` folder will go through the model.  
   - A series of predictions will show up in your terminal.  
   - At the end, a percentage will tell what is the preciseness of current model. 

### Personal Training Experience
After labeling over 700+ images, after a certain amount of epochs, the correct percentage could have reached around `70%`.  The more images, the better the result is.


### Concepts
#### Inspired by [Andy Wu's Note](https://notes.andywu.tw/2019/%E7%94%A8tensorflowkeras%E8%A8%93%E7%B7%B4%E8%BE%A8%E8%AD%98%E9%A9%97%E8%AD%89%E7%A2%BC%E7%9A%84cnn%E6%A8%A1%E5%9E%8B/)