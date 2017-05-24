# **Behavioral Cloning** 

During this project I trained a deep convolutional neural network to imitate the steering angles of a human driver, around multiple simulated laps of a circuit.

[//]: # (Image References)
[cameraviews]: ./writeup_images/camera_views.png "Simulator cameras"
[datasetbalancing]: ./writeup_images/dataset_balancing.png "Dataset balancing"
[croppedandflipped]: ./writeup_images/cropped_flipped.png "Cropping and flipping"

## Testing

The network was tested by leaving the car to be automously driven around the circuit for 1 hour, which it did without leaving the circuit, crashing or touching the kerb. A video of a single lap is included with the submission (``video.mp4``).

## Data collection

At first I tried to collect more data using the simulator with a keyboard, however, the steering angles recorded in this way were quite extreme, as you can not input "slightly right" to the keyboard for example. I then collected data using an Xbox 360 controller connected to the computer and this smoothed out the steering angles considerably.

I recorded my driving whilst doing a full lap of the circuit. As the course was an anti-clockwise  circuit most of the steering was in the left direction. To try to balance this out I also recorded a lap of driving clockwise, and so mainly turning right. A second track was also available, and so I also recorded two laps of this, one forward and one backward. Finally I recorded some "recovery manoeuvres", where I positioned the car in a place where it would need sharp turning to correct it from driving off the course, then recorded me returning it to the center of the circuit, the aim of this is that if the car starts to veer off course, the network will have learned how to correct for that.

## Code

The code for training is included in ``model.py``. The lessons suggest that if you are running out of RAM whilst processing the training data, a generator should be used which reads each batch preprocesses it and feeds it into the network before opening the next batch of images. This was not necessary in my case, more data was not required and my machine had sufficient RAM for the data used.

### The Network

To start with I experimented with experimented with various network architectures by adding different sized fully connected layers, the performance improved as I added more layers, but the number of parameters and so the size of the resulting model grew out of control quickly. I then added some convolutional layers instead, this allowed the network to learn about patches of the images rather than individual pixels. I finally settled on the network shown in the [Nvidia End-to-End learning paper](https://arxiv.org/pdf/1604.07316v1.pdf), with added relu activation stages to introduce nonlinearity and a dropout stage to reduce overfitting. Below is the full architecture.

1. Input image size was 85x320x3
2. Lambda layer for normalising the image to be floats centered around 0.
3. A cropping layer to remove the top 50 pixels (sky) and bottom 25 pixels (bonnet/hood).
4. Convolution layer with kernel size of 5x5 and stride 2x2 and depth 24
5. Relu activation
6. Convolution layer with kernel size of 5x5 and stride 2x2 and depth 36
7. Convolution layer with kernel size of 3x3 and stride 1x1 and depth 48
8. Relu activation
9. Convolution layer with kernel size of 3x3 and stride 1x1 and depth 64
10. 10% dropout
11. Fully connected layer with 100 nodes
12. Fully connected layer with 50 nodes
13. Fully connected layer with 10 nodes
14. Fully connected layer with 1 node which predicts the steering angle.

I tried altering the shape of the input image to be the same as in the Nvidia paper (66x200), but it didn't show much improvement in using the standard sized image after cropping (85x320).

I did not need to set the learning rate as the Adam optimiser was used, which uses an adaptive learning rate, starting with a large one and exponentially reducing as the epochs progress.

### Data Handling

The simulator simulataneously records the output of left, right and central mounted cameras. Example outputs from these cameras are shown below.

[//]:![cameraviews]
![Simulator camera outputs][cameraviews]

#### Balancing the dataset

When looking at a histogram of steering angles associated with the collected images, we can see that there is substantially more where the steering angle is near 0. When looking more closely the example data provided, more than 50% of the steering angles were exactly 0. It is not good to train neural networks with heavily biased data, and in this case the extreme bias for a zero steering angle will mean that the neural network will be more prone to driving straight than we would like.

![datasetbalancing]

To combat this I randomly discarded a large amount of the training images where the steering angle was 0. At first I was going to deicde to discard a percentage of the 0 angle images. In the end I produced a histogram of the steering angles with 250 equally spaced bins (from min to max value), and reduced the number of zero angle images to the same as the second highest bin. The hope is this should scale better with different data sets. This was all placed into the ``balance_dataset`` function in ``model.py``.

#### Left and right cameras

The steering angle recorded for the sets of center, left and right images all had a single steering angle, and the car in the future would be steering based on the center image. Therefore to use the left and right images for training, a correction factor needed to be applied to the steering angle. Using basic geometry I made an estimate of the correction factor being 0.1 and -0.1 radians for the left and right cameras respectively. After further empirical investigation I decided to use 0.2 and -0.2.

The left and right camera images are very useful in training as they give a continuous example of what angle to use when facing offcenter from the intended direction of travel. However, as with the center images if I used the whole dataset it would have highly overrepresented steering angles as 0.2 and -0.2, so the left and right steering angles were processed after the dataset balancing step.

## Training

The dataset was split into training and validation sets during the training process to reduce overfitting and to ensure that the model was likely to generalise well. 5 epochs was generally sufficient to reach a minimum validation loss, after that the loss didn't decrease even whilst using a dynamic learning rate. Before training the data was shuffled so that continous blocks of images were not all in the same train/validation sets during cross validation.

#### Cropping and flipping the images

A large proportion of each image produced by the simulator is either sky or the bonnet of the car. To handle this, this isn't useful when training the network, and so the images are cropped by Keras before training.

![croppedandflipped]

To create more training data the images were also flipped, and the training data multiplied by -1. 

## Further work

- Further investigate using elu instead of relu: I briefly tried this but it made my car swerve quite violently from side to side, further investigation may have made it successful.
- For most training runs I only used 7 epochs as this minimised the error well, on the final run I used 20 epochs, this didn't improve the loss, so I would not do this again.
- Make more accuracte calculations for the correction factor for the left and right cameras.
- Train exclusively on the second track to see how well the model generalises to the main project track.