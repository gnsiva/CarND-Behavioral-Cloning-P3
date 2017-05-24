#**Behavioral Cloning** 

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
- normalisation
- dropout
- elu/relu
- image input size
- adam optimizer (why that means you didn't set the learning rate)

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

#### Cropping and flipping the images

A large proportion of each image produced by the simulator is either sky or the bonnet of the car. To handle this, this isn't useful when training the network, and so the images are cropped by Keras before training.

![croppedandflipped]

To create more training data the images were also flipped, and the training data multiplied by -1. 

## Further work
- generator
- train for less epochs
- try using elu instead of relu
- collect data using a steering wheel instead of a controller
- calculating the correction factor for left/right camera
- train exclusively on the second track to show it is generalising to the first track


*examples of images from the dataset must be included.*
