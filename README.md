# Car tracking as oriented bounding boxes with a modified CenterNet

In this repo, I propose a multi-object detection network for cars/vehicles seen from a Bird Eye View (BEV). Instead of detecting a basic rectangular bounding box, we can also train the network to detect the angle and orientation of each vehicles.

Then, a basic tracking algorithm is applied with Kalman Filtering for each objects.

This runs at 20fps on my laptop with a Nvidia RTX A2000 GPU.

<p>
<em>Tracking</em></br>
<img src="training/res/demo.gif" width="1000" alt>
</p>

[Youtube video link](https://www.youtube.com/watch?v=Bs9wLnG2jMY&ab_channel=antoinekeller)

## Training

The network is inspired from [CenterNet](https://github.com/xingyizhou/CenterNet). In addition to position, offsets and bounding boxes dimensions, we also predict sine/cosine heatmaps to have the yaw angle.

See [training](training/) repo for more details.

## Tracking

I managed to keep the code as simple as possible, so that it can easily be customized to your needs.

You will maybe need to change the process noise covariance matrix in the Kalman Filter.

```
python multi_object_tracking.py path/to/video.mp4 training/centernet-oriented-bbox.pth
```

You can also:
- display objects ids
- display objects speeds
- change trajectory length to be displayed

## Some nice samples

- [Ta Dream channel](https://www.youtube.com/@tadream)
- https://www.youtube.com/watch?v=BV7bI6R8-0g&ab_channel=TimelessAerialPhotography
- https://www.youtube.com/watch?v=bfc2wsX29zk&ab_channel=FloridaManDrone
- https://www.youtube.com/watch?v=FpSau23yde4&ab_channel=LeonardSuchanek

Enjoy !